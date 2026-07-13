"""PrefixLockStage — 三段 hash 前缀匹配 + KV cache 锁定。

匹配 session 的前缀链，锁定 sys/tools/messages 到前缀值以最大化缓存命中。
通过 MemoryStore 读取（0 I/O），SQLite 落盘为 fire-and-forget。

匹配策略 (v2, 2026-06-12):
  - messages 段: 先提取 stable prefix（倒数第二个 user 之前），再双向匹配
  - sys/tools 段: 全量位置化匹配（SysToolsLock 已保证稳定）
  - 锁定: chain[:fc] + curr[fc:n-bc] + chain[n-bc:] + current_round
"""
from __future__ import annotations

import logging

from deepseek_tap_proxy.pipeline import Stage, Context, ChainAction
from deepseek_tap_proxy.shared.chain import Chain
from deepseek_tap_proxy.shared.store import MemoryStore
from deepseek_tap_proxy.shared.stages.helpers import (
    json_list, chain_hash_view, find_chain_result, create_chain,
    _find_anchor_across_chains, _ensure_frozen_chain,
)
from deepseek_tap_proxy.shared.chain import find_anchor_moment

log = logging.getLogger("deepseek-tap")


def _classify_pref_fail(
    chains: list[Chain],
    stable_hashes: list[str],
    family_key: str,
    store: MemoryStore | None,
) -> str:
    """分类 pref=0 的原因，返回简短标签。

    逻辑：
      1. 无链 → 'init'
      2. 有链但 family_key 不匹配任何链 → 'tools_changed'
      3. 有同族链但 position 0 hash 不匹配 → 'context_shift'
      4. 有同族链、position 0 匹配但整体不匹配 → 'multi_conv'
      5. 其他 → 'chain_mismatch'
    """
    if not chains:
        return "init"

    if not stable_hashes:
        return "empty_stable"

    # 检查是否有同 family_key 的链
    same_family = []
    other_family = []
    for c in chains:
        fk = _chain_family_key(c)
        if fk == family_key:
            same_family.append(c)
        else:
            other_family.append(c)
    if not same_family:
        # 没有同族链 → tools 变更导致 family_key 变化
        # 进一步检查：其他族链的 position 0 是否匹配
        first_hash = stable_hashes[0]
        for c in other_family:
            ch = (c.messages_hashes or "").split(",")
            if ch and ch[0] == first_hash:
                return "tools_changed"  # tools 变但 context 相同
        return "tools_changed_context_shift"  # tools 变且 context 也变

    # 检查 position 0 hash 是否匹配任何同族链
    first_hash = stable_hashes[0]
    for c in same_family:
        ch = (c.messages_hashes or "").split(",")
        if ch and ch[0] == first_hash:
            # position 0 匹配但整体不匹配 → 多 conversation
            return "multi_conv"

    # 所有同族链的 position 0 都不匹配 → context 日期变化
    return "context_shift"


def _chain_family_key(chain: Chain) -> str:
    """计算链的 family_key（与 Context._recompute_family_key 一致）。"""
    import xxhash
    h = xxhash.xxh3_64()
    for s in (chain.sys_hashes or "").split(","):
        if s:
            h.update(s.encode())
    for t in (chain.tools_hashes or "").split(","):
        if t:
            h.update(t.encode())
    return h.hexdigest()[:16]


def _find_ref_chain(sys_result, tools_result, chains: list[Chain]) -> Chain | None:
    """从 sys/tools 匹配结果中找到参考链（用于 frozen chain 创建）。"""
    for result in (tools_result, sys_result):
        if result is not None:
            for c in chains:
                if c.chain_id == result.chain_id:
                    return c
    return None


class PrefixLockStage(Stage):
    """Prefix lock: match session's prefix chains against canonical form.

    Reads from MemoryStore (0 I/O). All DB writes are fire-and-forget
    via store.enqueue_sqlite_write.
    """

    def __init__(self, enabled: bool = True, cross_family_fallback: bool = False) -> None:
        self.enabled = enabled
        self.cross_family_fallback = cross_family_fallback

    async def process(self, ctx: Context) -> Context:
        if not self.enabled:
            return ctx

        if ctx.chat_type == "compaction":
            return ctx  # CompactionLockStage handles chain matching downstream

        session_id = ctx.session_id
        if not session_id:
            return ctx

        sys_c, tools_c, msgs_c = ctx.sys, ctx.tools, ctx.messages
        all_msgs_c = sys_c + tools_c + msgs_c
        if not all_msgs_c:
            return ctx

        rt = ctx.request.get("runtime")
        store: MemoryStore | None = rt.store if rt else None
        if not store:
            return ctx

        # 按分支族过滤：同 session + 同 sys+tools → 只匹配同族链
        if ctx.family_key:
            chains = store.get_family_chains(session_id, ctx.family_key)
        else:
            chains = store.get_session_chains(session_id)

        # ── CREATE：无已有链 ──
        if not chains:
            if self.cross_family_fallback:
                chains = store.get_session_chains(session_id)
                sticky_id = store.get_sticky_chain(session_id, ctx.chat_type)
                if sticky_id and chains:
                    chains.sort(key=lambda c: 0 if c.chain_id == sticky_id else 1)

        if not chains:
            ctx.meta["pref_fail_reason"] = "init"
            ctx.meta["matched_chain_id"] = ""
            ctx.meta["chain_action_type"] = "create"
            ctx.meta["new_char_count"] = ctx.char_count_of(all_msgs_c)
            create_chain(ctx, all_msgs_c, hit_count=0)
            return ctx

        from deepseek_tap_proxy.shared.match import match_prefix
        from deepseek_tap_proxy.shared.boundary import extract_stable_prefix

        # ── 提取 stable prefix（对 compaction/subagent 不做边界提取） ──
        if ctx.chat_type == "normal":
            msgs_boundary = extract_stable_prefix(msgs_c)
        else:
            msgs_boundary = len(msgs_c)  # compaction/subagent: 全量匹配

        stable_msgs = msgs_c[:msgs_boundary]
        curr_round_msgs = msgs_c[msgs_boundary:]
        stable_hashes = ctx.messages_hashes[:msgs_boundary]

        # ── 三段独立匹配 ──

        def _match_segment(hashes_key: str, curr_hashes: list[str]):
            if not curr_hashes:
                return None
            return match_prefix(
                [chain_hash_view(c, hashes_key, store) for c in chains],
                curr_hashes,
            )

        msg_result = _match_segment("messages_hashes", stable_hashes)  # ← only stable
        sys_result = _match_segment("sys_hashes", ctx.sys_hashes)
        tools_result = _match_segment("tools_hashes", ctx.tools_hashes)

        # 三段命中提前写入 meta（无论 msg 是否匹配，LogView 都能读到）
        # 修复：压缩后 msg_result=None 但 sys/tools 可能跨链命中
        sys_fc_pre = sys_result.fc if sys_result else 0
        tools_fc_pre = tools_result.fc if tools_result else 0
        ctx.meta["pref_sys_hit"] = sys_fc_pre
        ctx.meta["pref_tools_hit"] = tools_fc_pre

        if msg_result is None:
            ctx.meta["pref_msgs_hit"] = 0
            # ── 分类 pref=0 原因 ──
            ctx.meta["pref_fail_reason"] = _classify_pref_fail(
                chains, stable_hashes, ctx.family_key, store
            )
            ctx.meta["matched_chain_id"] = ""
            ctx.meta["chain_action_type"] = "create"
            ctx.meta["new_char_count"] = ctx.char_count_of(all_msgs_c)
            # ── 跨链锚点：fork 后 tools 匹配但 messages 不匹配 ──
            _ref = _find_ref_chain(sys_result, tools_result, chains)
            if _ref is not None and store:
                _pref = sys_fc_pre + tools_fc_pre
                anchor = _find_anchor_across_chains(chains, _pref)
                if anchor is None:
                    anchor = _ensure_frozen_chain(
                        store, _ref, _pref,
                        session_id, ctx.family_key or "",
                    )
                ctx.anchor_moment = anchor
            create_chain(ctx, all_msgs_c, hit_count=0)
            return ctx

        best_chain_id = msg_result.chain_id
        best_chain: Chain = next(c for c in chains if c.chain_id == best_chain_id)
        ctx.persist.prefix_chain_id = best_chain.chain_id
        # 更新粘性链
        if self.cross_family_fallback:
            store.set_sticky_chain(session_id, ctx.chat_type, best_chain_id)
        ctx.meta["prefix_chain_msg_count"] = len(best_chain.sys) + len(best_chain.tools) + len(best_chain.messages)

        sys_best = find_chain_result(sys_result, best_chain_id)
        tools_best = find_chain_result(tools_result, best_chain_id)

        sys_p = best_chain.sys
        tools_p = best_chain.tools
        msg_p = best_chain.messages
        all_msgs_p = sys_p + tools_p + msg_p

        # ── 双向匹配结果 ──
        msg_fc = msg_result.fc
        msg_bc = msg_result.bc
        sys_fc = sys_best.fc if sys_best else 0
        tools_fc = tools_best.fc if tools_best else 0

        # 兼容旧接口：n = fc - 1（用于 sys/tools 锁定）
        sys_n = sys_fc - 1 if sys_fc > 0 else -1
        tools_n = tools_fc - 1 if tools_fc > 0 else -1

        msgs_lock = msg_result.lock_type if msg_result else "none"
        sys_lock = sys_best.lock_type if sys_best else "none"
        tools_lock = tools_best.lock_type if tools_best else "none"

        def _effective_lock(lock_str: str, curr_len: int, chain_len: int) -> str:
            if curr_len == 0 and chain_len == 0:
                return "layer1_perfect"
            return lock_str or "none"

        eff_sys = _effective_lock(sys_lock, len(sys_c), len(sys_p))
        eff_tools = _effective_lock(tools_lock, len(tools_c), len(tools_p))
        eff_msgs = _effective_lock(msgs_lock if msg_result else "none", len(stable_msgs), len(msg_p))

        if eff_sys == "layer1_perfect" and eff_tools == "layer1_perfect" and eff_msgs == "layer1_perfect":
            lock_type = "layer1_perfect"
        elif msg_result:
            lock_type = "layer2_partial"
        else:
            lock_type = "none"

        # ── 三段锁定 ──
        # sys: chain[:sys_fc] + curr[sys_fc:]
        locked_sys = sys_p[:sys_fc] + sys_c[sys_fc:] if sys_fc > 0 else sys_c
        # tools: chain[:tools_fc] + curr[tools_fc:]
        locked_tools = tools_p[:tools_fc] + tools_c[tools_fc:] if tools_fc > 0 else tools_c

        # messages: chain[:fc] + stable[fc:n-bc] + chain[n-bc:] + current_round
        n = min(len(msg_p), len(stable_hashes))
        locked_msgs = list(msg_p[:msg_fc])  # front matched from chain
        if msg_fc < n - msg_bc:
            locked_msgs.extend(stable_msgs[msg_fc:n - msg_bc])  # middle from curr
        if msg_bc > 0:
            locked_msgs.extend(msg_p[n - msg_bc:n])  # back matched from chain
        if n < len(stable_msgs):
            locked_msgs.extend(stable_msgs[n:])  # extra stable beyond chain (from curr)
        locked_msgs.extend(curr_round_msgs)  # current round unconditionally

        # 防御性验证仅对 messages 段（tool_call 边界）
        from deepseek_tap_proxy.shared.prefix import (
            compute_safe_breakpoints, find_safe_truncation, validate_tool_call_chain,
        )
        if msg_bc > 0 and not validate_tool_call_chain(locked_msgs):
            safe_bps = compute_safe_breakpoints(msg_p)
            # 后向匹配可能跨 tool_call 边界，用 safe_breakpoints 修正
            safe_trunc = find_safe_truncation(safe_bps, n - msg_bc)
            if safe_trunc < n - msg_bc:
                # 回退后向匹配到安全边界
                adjusted_bc = n - safe_trunc
                locked_msgs = list(msg_p[:msg_fc])
                if msg_fc < n - adjusted_bc:
                    locked_msgs.extend(stable_msgs[msg_fc:n - adjusted_bc])
                if adjusted_bc > 0:
                    locked_msgs.extend(msg_p[n - adjusted_bc:n])
                if n < len(stable_msgs):
                    locked_msgs.extend(stable_msgs[n:])
                locked_msgs.extend(curr_round_msgs)
                log.warning(
                    f"[{ctx.req_id}] back_match corrected for tool_call boundary: "
                    f"{msg_bc}→{adjusted_bc}"
                )
                msg_bc = adjusted_bc

        if lock_type != "layer1_perfect":
            ctx.persist.is_modified = True
        ctx.meta["log_msgs_after"] = len(locked_sys) + len(locked_tools) + len(locked_msgs)

        # ── Hit count：fc + bc（而非旧 n+1） ──
        total_hit = msg_fc + msg_bc
        # Bug 1 fix: pref_hit 含三段总和（用于 _find_anchor + all_msgs_c 切片）
        # 之前只含 msg_fc+msg_bc，遗漏 sys+tools，导致 _find_anchor 找不到 moment → est=?
        pref_hit = sys_fc + tools_fc + msg_fc + msg_bc
        ctx.meta["pref_hit_count"] = pref_hit
        # 只在 best_chain 有匹配时覆盖 pre-gate 值（防 frozen chain 污染）
        ctx.meta["pref_sys_hit"] = sys_fc if sys_best else ctx.meta["pref_sys_hit"]
        ctx.meta["pref_tools_hit"] = tools_fc if tools_best else ctx.meta["pref_tools_hit"]
        ctx.meta["pref_msgs_hit"] = msg_fc

        # 设置 matched_chain, anchor_moment, new_char_count
        ctx.matched_chain = best_chain
        ctx.anchor_moment = find_anchor_moment(best_chain.moments, pref_hit)
        # Fallback 1: 跨链查找（含 frozen chain 的 moment）
        if ctx.anchor_moment is None:
            ctx.anchor_moment = _find_anchor_across_chains(chains, pref_hit)
        # Fallback 2: 创建公共前缀 frozen chain
        if ctx.anchor_moment is None and store:
            ctx.anchor_moment = _ensure_frozen_chain(
                store, best_chain, pref_hit,
                session_id, ctx.family_key or "",
            )
        new_msgs = all_msgs_c[pref_hit:]
        ctx.new_char_count = ctx.char_count_of(new_msgs)

        # 从 anchor_moment 获取 tokens 用于 chain_action
        chain_tokens = ctx.anchor_moment.total_tokens if ctx.anchor_moment else 0

        # 写回规范形式 — setter 自动刷新 hash chain
        ctx.sys = locked_sys
        ctx.tools = locked_tools
        ctx.messages = locked_msgs

        ctx.persist.lock_type = lock_type
        prefix_note = (
            f"{lock_type}: "
            f"sys_fc={sys_fc}, tools_fc={tools_fc}, "
            f"msg_fc={msg_fc}, msg_bc={msg_bc}, "
            f"prefix=({len(sys_p)},{len(tools_p)},{len(msg_p)}), "
            f"curr=({len(sys_c)},{len(tools_c)},{len(msgs_c)})"
            + (f" boundary={msgs_boundary}" if msgs_boundary < len(msgs_c) else "")
        )
        existing = ctx.persist.modification_note or ""
        ctx.persist.modification_note = (
            f"{existing} | {prefix_note}" if existing else prefix_note
        )

        from deepseek_tap_proxy.shared.prefix import make_chain_id

        cc = ctx.total_char_count  # setter 写回后自动更新

        # All chains are mutable (is_frozen removed); always extend when longer
        if len(all_msgs_c) > len(all_msgs_p):
            ctx.chain_action = ChainAction(
                action="update", chain_id=best_chain.chain_id,
                msg_count=len(locked_sys) + len(locked_tools) + len(locked_msgs),
                sys_hashes=ctx.sys_hash_chain, tools_hashes=ctx.tools_hash_chain,
                messages_hashes=ctx.messages_hash_chain,
                sys=ctx.sys, tools=ctx.tools, messages=ctx.messages,
                hit_count=total_hit, sys_hit=sys_fc, tools_hit=tools_fc,
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
                hit_count=0, chain_tokens=0, char_count=ctx.char_count_of(all_msgs_c),
            )
        ctx.persist.chain_id = ctx.chain_action.chain_id
        ctx.persist.chain_action = ctx.chain_action.action
        ctx.persist.chain_msg_count = ctx.chain_action.msg_count
        ctx.meta["matched_chain_id"] = best_chain.chain_id
        ctx.meta["chain_action_type"] = ctx.chain_action.action
        ctx.meta["new_char_count"] = ctx.new_char_count
        if ctx.chain_action.action == "update":
            ctx.persist.chain_tokens = chain_tokens or 0
        return ctx
