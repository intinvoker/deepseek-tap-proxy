"""PrefixLockStage — 三段 hash 前缀匹配 + KV cache 锁定。

匹配 session 的前缀链，锁定 sys/tools/messages 到前缀值以最大化缓存命中。
通过 MemoryStore 读取（0 I/O），SQLite 落盘为 fire-and-forget。
"""
from __future__ import annotations

import json
import logging

from deepseek_tap_proxy.pipeline import Stage, Context, ChainAction
from deepseek_tap_proxy.shared.store import MemoryStore
from deepseek_tap_proxy.shared.stages.helpers import (
    compute_char_count, json_list, chain_hash_view, find_chain_result, create_chain,
)

log = logging.getLogger("deepseek-tap")


class PrefixLockStage(Stage):
    """Prefix lock: match session's prefix chains against canonical form.

    Reads from MemoryStore (0 I/O). All DB writes are fire-and-forget
    via store.enqueue_sqlite_write.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    async def process(self, ctx: Context) -> Context:
        if not self.enabled:
            return ctx

        session_id = ctx.session_id
        if not session_id:
            return ctx

        sys_c, tools_c, msgs_c = ctx.sys, ctx.tools, ctx.messages
        all_msgs_c = sys_c + tools_c + msgs_c
        if not all_msgs_c:
            return ctx

        store: MemoryStore | None = ctx.request.app.get("store")
        if not store:
            return ctx

        # 按分支族过滤：同 session + 同 sys+tools → 只匹配同族链
        if ctx.family_key:
            chains = store.get_family_chains(session_id, ctx.family_key)
        else:
            chains = store.get_session_chains(session_id)

        # ── CREATE：无已有链 ──
        if not chains:
            create_chain(ctx, all_msgs_c, hit_count=0)
            return ctx

        from deepseek_tap_proxy.shared.match import match_prefix

        # ── 三段独立匹配 ──
        _user_positions = [
            i for i, m in enumerate(msgs_c)
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        _disable_ua = ctx.chat_type == "compaction"

        def _match_segment(hashes_key: str, curr_hashes: list[str],
                           user_pos: list[int], disable_ua: bool):
            if not curr_hashes:
                return None
            return match_prefix(
                [chain_hash_view(c, hashes_key) for c in chains],
                curr_hashes, user_pos, disable_user_anchor=disable_ua,
            )

        msg_result = _match_segment("messages_hashes", ctx.messages_hashes, _user_positions, _disable_ua)
        sys_result = _match_segment("sys_hashes", ctx.sys_hashes, [], True)
        tools_result = _match_segment("tools_hashes", ctx.tools_hashes, [], True)

        if msg_result is None:
            create_chain(ctx, all_msgs_c, hit_count=0)
            return ctx

        best_chain_id = msg_result.chain_id
        best_chain = next(c for c in chains if c["chain_id"] == best_chain_id)
        ctx.persist.prefix_chain_id = best_chain["chain_id"]
        ctx.meta["prefix_chain_msg_count"] = best_chain["msg_count"]

        sys_best = find_chain_result(sys_result, best_chain_id)
        tools_best = find_chain_result(tools_result, best_chain_id)

        sys_p = json_list(best_chain.get("sys"))
        tools_p = json_list(best_chain.get("tools"))
        msg_p = json_list(best_chain.get("messages"))
        all_msgs_p = sys_p + tools_p + msg_p

        msgs_n = msg_result.n if msg_result else -1
        sys_n = sys_best.n if sys_best else -1
        tools_n = tools_best.n if tools_best else -1

        msgs_lock = msg_result.lock_type if msg_result else "none"
        sys_lock = sys_best.lock_type if sys_best else "none"
        tools_lock = tools_best.lock_type if tools_best else "none"

        def _effective_lock(lock_str: str, curr_len: int, chain_len: int) -> str:
            if curr_len == 0 and chain_len == 0:
                return "layer1_perfect"
            return lock_str or "none"

        eff_sys = _effective_lock(sys_lock, len(sys_c), len(sys_p))
        eff_tools = _effective_lock(tools_lock, len(tools_c), len(tools_p))
        eff_msgs = _effective_lock(msgs_lock if msg_result else "none", len(msgs_c), len(msg_p))

        if eff_sys == "layer1_perfect" and eff_tools == "layer1_perfect" and eff_msgs == "layer1_perfect":
            lock_type = "layer1_perfect"
        elif msg_result:
            lock_type = "layer2_partial"
        else:
            lock_type = "none"

        locked_sys = sys_p[:sys_n + 1] + sys_c[sys_n + 1:] if sys_n >= 0 else sys_c
        locked_tools = tools_p[:tools_n + 1] + tools_c[tools_n + 1:] if tools_n >= 0 else tools_c
        locked_msgs = msg_p[:msgs_n + 1] + msgs_c[msgs_n + 1:] if msgs_n >= 0 else msgs_c

        # 防御性验证仅对 messages 段（tool_call 边界）
        from deepseek_tap_proxy.shared.prefix import (
            compute_safe_breakpoints, find_safe_truncation, validate_tool_call_chain,
        )
        msgs_n_original = msgs_n
        if not validate_tool_call_chain(locked_msgs):
            safe_bps = None
            if best_chain.get("safe_breakpoints"):
                try:
                    safe_bps = json.loads(best_chain["safe_breakpoints"])
                except (json.JSONDecodeError, TypeError):
                    safe_bps = None
            if safe_bps is None:
                safe_bps = compute_safe_breakpoints(msg_p)
            msgs_n = find_safe_truncation(safe_bps, msgs_n)
            locked_msgs = msg_p[:msgs_n + 1] + msgs_c[msgs_n + 1:]
            log.warning(
                f"[{ctx.req_id}] msgs_n corrected for tool_call boundary: "
                f"{msgs_n_original}→{msgs_n}"
            )

        if lock_type != "layer1_perfect":
            ctx.persist.is_modified = True
        ctx.meta["log_msgs_after"] = len(locked_sys) + len(locked_tools) + len(locked_msgs)

        total_hit = max(0, sys_n + 1) + max(0, tools_n + 1) + max(0, msgs_n + 1)
        ctx.meta["pref_hit_count"] = total_hit
        ctx.meta["pref_sys_hit"] = max(0, sys_n + 1)
        ctx.meta["pref_tools_hit"] = max(0, tools_n + 1)
        ctx.meta["pref_msgs_hit"] = max(0, msgs_n + 1)

        # est 数据 — from MemoryStore (0 I/O)
        pref_hit = total_hit
        chain_tc = store.get_chain_tokens_and_chars(best_chain["chain_id"], pref_hit)
        if chain_tc is None:
            chain_tc_ge = store.get_chain_tokens_ge(best_chain["chain_id"], pref_hit)
            if chain_tc_ge is not None:
                ge_tokens, ge_chars, ge_msg_count = chain_tc_ge
                overflow = ge_msg_count - pref_hit
                if overflow > 0 and ge_chars > 0:
                    overflow_chars = sum(
                        len(json.dumps(all_msgs_p[pref_hit + i], ensure_ascii=False))
                        for i in range(min(overflow, len(all_msgs_p) - pref_hit))
                    )
                    tpc = ge_tokens / ge_chars
                    prefix_tokens_est = int(ge_tokens - overflow_chars * tpc)
                else:
                    prefix_tokens_est = ge_tokens

                from deepseek_tap_proxy.shared.prefix import make_chain_id, compute_safe_breakpoints as _bps
                from deepseek_tap_proxy.shared.colors import green

                _prefix_context = all_msgs_p[:pref_hit]
                _prefix_cc = compute_char_count(_prefix_context)

                _p_sys = _prefix_context[:len(sys_p)]
                _p_tools = _prefix_context[len(sys_p):len(sys_p) + len(tools_p)]
                _p_msgs = _prefix_context[len(sys_p) + len(tools_p):]

                _prefix_hashes = (
                    best_chain["sys_hashes"].split(",")[:sys_n + 1] +
                    best_chain["tools_hashes"].split(",")[:tools_n + 1] +
                    best_chain["messages_hashes"].split(",")[:msgs_n + 1]
                )
                _prefix_chain_id = make_chain_id(_prefix_hashes)

                # MemoryStore write (0 I/O)
                store.upsert_chain(
                    ctx.session_id, _prefix_chain_id,
                    sys=json.dumps(_p_sys), tools=json.dumps(_p_tools),
                    messages=json.dumps(_p_msgs),
                    sys_hashes=",".join(best_chain["sys_hashes"].split(",")[:sys_n + 1]),
                    tools_hashes=",".join(best_chain["tools_hashes"].split(",")[:tools_n + 1]),
                    messages_hashes=",".join(best_chain["messages_hashes"].split(",")[:msgs_n + 1]),
                    msg_count=pref_hit, safe_breakpoints=_bps(_prefix_context), is_frozen=True,
                    family_key=ctx.family_key,
                )
                store.insert_chain_token(_prefix_chain_id, pref_hit, prefix_tokens_est, _prefix_cc)

                # Fire-and-forget SQLite writes
                _schedule_chain_persist(store, ctx.session_id, _prefix_chain_id,
                                        json.dumps(_p_sys), json.dumps(_p_tools),
                                        json.dumps(_p_msgs),
                                        ",".join(best_chain["sys_hashes"].split(",")[:sys_n + 1]),
                                        ",".join(best_chain["tools_hashes"].split(",")[:tools_n + 1]),
                                        ",".join(best_chain["messages_hashes"].split(",")[:msgs_n + 1]),
                                        pref_hit, _bps(_prefix_context), True,
                                        _prefix_chain_id, pref_hit, prefix_tokens_est, _prefix_cc)

                print(green(f"  [chain] + prefix: {_prefix_chain_id[:16]}... "
                            f"(s={sys_n + 1} t={tools_n + 1} m={msgs_n + 1}, frozen, est={prefix_tokens_est}t)"))

                chain_tokens = prefix_tokens_est
                chain_char_count = _prefix_cc
            else:
                chain_tokens = 0
                chain_char_count = 0
        else:
            chain_tokens = chain_tc[0]
            chain_char_count = chain_tc[1]
        ctx.meta["chain_tokens_est"] = chain_tokens
        ctx.meta["chain_char_count"] = chain_char_count
        new_msgs = all_msgs_c[pref_hit:]
        ctx.meta["new_char_count"] = compute_char_count(new_msgs)

        # 写回规范形式 — setter 自动刷新 hash chain
        ctx.sys = locked_sys
        ctx.tools = locked_tools
        ctx.messages = locked_msgs

        ctx.persist.lock_type = lock_type
        correction_note = ""
        if msgs_n != msgs_n_original:
            correction_note = f" msgs_n_corrected: {msgs_n_original}→{msgs_n}"
        prefix_note = (
            f"{lock_type}: "
            f"sys_n={sys_n}, tools_n={tools_n}, msgs_n={msgs_n}, "
            f"prefix=({len(sys_p)},{len(tools_p)},{len(msg_p)}), "
            f"curr=({len(sys_c)},{len(tools_c)},{len(msgs_c)})"
            + correction_note
        )
        existing = ctx.persist.modification_note or ""
        ctx.persist.modification_note = (
            f"{existing} | {prefix_note}" if existing else prefix_note
        )

        from deepseek_tap_proxy.shared.prefix import make_chain_id

        cc = compute_char_count(locked_sys + locked_tools + locked_msgs)

        _matched_is_frozen = bool(best_chain.get("is_frozen"))
        if len(all_msgs_c) > len(all_msgs_p) and not _matched_is_frozen:
            ctx.chain_action = ChainAction(
                action="update", chain_id=best_chain["chain_id"],
                msg_count=len(locked_sys) + len(locked_tools) + len(locked_msgs),
                sys_hashes=ctx.sys_hash_chain, tools_hashes=ctx.tools_hash_chain,
                messages_hashes=ctx.messages_hash_chain,
                sys=ctx.sys, tools=ctx.tools, messages=ctx.messages,
                hit_count=total_hit, sys_hit=max(0, sys_n + 1), tools_hit=max(0, tools_n + 1),
                chain_tokens=chain_tokens, char_count=cc,
            )
        else:
            _all_hashes = ctx.sys_hashes + ctx.tools_hashes + ctx.messages_hashes
            ctx.chain_action = ChainAction(
                action="create", chain_id=make_chain_id(_all_hashes),
                msg_count=len(all_msgs_c),
                sys_hashes=ctx.sys_hash_chain, tools_hashes=ctx.tools_hash_chain,
                messages_hashes=ctx.messages_hash_chain,
                sys=ctx.sys, tools=ctx.tools, messages=ctx.messages,
                hit_count=0, chain_tokens=0, char_count=compute_char_count(all_msgs_c),
            )
        ctx.persist.chain_id = ctx.chain_action.chain_id
        ctx.persist.chain_action = ctx.chain_action.action
        ctx.persist.chain_msg_count = ctx.chain_action.msg_count
        if ctx.chain_action.action == "update":
            ctx.persist.chain_tokens = chain_tokens or 0
        return ctx


def _schedule_chain_persist(store: MemoryStore, session_id: str, chain_id: str,
                            sys: str, tools: str, messages: str,
                            sys_hashes: str, tools_hashes: str, messages_hashes: str,
                            msg_count: int, safe_breakpoints: list[int] | None,
                            is_frozen: bool,
                            cth_chain_id: str, cth_msg_count: int,
                            cth_tokens: int, cth_char_count: int) -> None:
    """Fire-and-forget: persist frozen prefix chain + token to SQLite."""
    from deepseek_tap_proxy.shared.db import upsert_prefix_chain as _upsert, insert_chain_token as _insert

    def _write(conn, sid, cid, s, t, m, sh, th, mh, mc, sb, frozen,
               tcid, tm, ttk, tcc):
        _upsert(conn, sid, cid, sys=s, tools=t, messages=m,
                sys_hashes=sh, tools_hashes=th, messages_hashes=mh,
                msg_count=mc, safe_breakpoints=sb, is_frozen=frozen)
        _insert(conn, tcid, msg_count=tm, tokens=ttk, char_count=tcc)

    store.enqueue_sqlite_write(
        _write, session_id, chain_id, sys, tools, messages,
        sys_hashes, tools_hashes, messages_hashes,
        msg_count, safe_breakpoints, is_frozen,
        cth_chain_id, cth_msg_count, cth_tokens, cth_char_count,
    )
