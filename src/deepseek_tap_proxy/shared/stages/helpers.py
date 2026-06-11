"""PrefixLock 内部工具函数：char 计数、JSON 解析、链操作、persist 辅助。"""
from __future__ import annotations

import json

from deepseek_tap_proxy.pipeline.chain_action import ChainAction


def compute_char_count(messages: list[dict]) -> int:
    """Compute total serialized character count for token estimation.

    Uses ``json.dumps`` (full message dict serialization) to capture all
    text — including tool definitions (``function.description``,
    ``parameters``) that lack a ``content`` field — so the
    ``tokens / chars`` ratio is consistent.
    """
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)


def json_list(raw) -> list[dict]:
    """Parse a JSON column into list[dict], returning [] on failure."""
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def chain_hash_view(chain: dict, hashes_key: str) -> dict:
    """Wrap a chain dict as a match_prefix-compatible view for a single segment."""
    return {
        "chain_id": chain["chain_id"],
        "hash_chain": chain.get(hashes_key, ""),
    }


def find_chain_result(match_result, chain_id: str):
    """Find the match result for a specific chain_id.

    match_result may be a MatchResult (single) or MatchResult | None.
    Returns the result if its chain_id matches, else None.
    """
    if match_result is not None and match_result.chain_id == chain_id:
        return match_result
    return None


def create_chain(ctx, all_msgs_c: list[dict], hit_count: int) -> ChainAction:
    """Create a new chain action and write to persist.

    Returns the created ChainAction (also set on ctx.chain_action).
    """
    from deepseek_tap_proxy.shared.prefix import make_chain_id, compute_safe_breakpoints
    from deepseek_tap_proxy.shared.stages.helpers import compute_char_count as _ccc

    _all_hashes = ctx.sys_hashes + ctx.tools_hashes + ctx.messages_hashes
    ctx.chain_action = ChainAction(
        action="create",
        chain_id=make_chain_id(_all_hashes),
        msg_count=len(all_msgs_c),
        sys_hashes=ctx.sys_hash_chain,
        tools_hashes=ctx.tools_hash_chain,
        messages_hashes=ctx.messages_hash_chain,
        sys=ctx.sys,
        tools=ctx.tools,
        messages=ctx.messages,
        safe_breakpoints=compute_safe_breakpoints(all_msgs_c),
        hit_count=hit_count,
        chain_tokens=0,
        char_count=_ccc(all_msgs_c),
    )
    ctx.persist.chain_id = ctx.chain_action.chain_id
    ctx.persist.chain_action = ctx.chain_action.action
    ctx.persist.chain_msg_count = ctx.chain_action.msg_count
    ctx.persist.chain_tokens = 0
    ctx.persist.lock_type = None  # CREATE 路径无锁，显式 None（避免默认 "" 竞态）
    return ctx.chain_action


def persist_chain_nowait(ctx, _db=None) -> None:
    """Pipeline 结束后立即写入 MemoryStore + 异步落 SQLite。

    由 openai / anthropic handler 统一调用。替代之前各 handler 的重复实现。
    """
    ca = getattr(ctx, "chain_action", None)
    session_id = ctx.session_id
    if not (ca and ca.action in ("create", "update") and session_id and ctx.chat_type != "compaction"):
        return

    from deepseek_tap_proxy.shared.db import upsert_prefix_chain, insert_chain_token
    from deepseek_tap_proxy.shared.colors import green
    from deepseek_tap_proxy.shared.store import MemoryStore

    store: MemoryStore | None = ctx.request.app.get("store")
    chain_tokens = ctx.persist.total_input_tokens or ctx.meta.get("prompt_tokens", 0)

    sys_json = json.dumps(ca.sys or ctx.sys)
    tools_json = json.dumps(ca.tools or ctx.tools)
    msgs_json = json.dumps(ca.messages or ctx.messages)
    sys_hashes = ca.sys_hashes or ctx.sys_hash_chain
    tools_hashes = ca.tools_hashes or ctx.tools_hash_chain
    msgs_hashes = ca.messages_hashes or ctx.messages_hash_chain
    msg_cnt = ca.msg_count or len(ctx.sys + ctx.tools + ctx.messages)

    # 1. MemoryStore write (0 I/O)
    if store:
        store.upsert_chain(
            session_id, ca.chain_id,
            sys=sys_json, tools=tools_json, messages=msgs_json,
            sys_hashes=sys_hashes, tools_hashes=tools_hashes,
            messages_hashes=msgs_hashes,
            msg_count=msg_cnt, safe_breakpoints=ca.safe_breakpoints,
            is_frozen=ca.is_frozen,
            family_key=ctx.family_key,
        )
        store.insert_chain_token(ca.chain_id, msg_cnt, chain_tokens, ca.char_count)

    # 2. Fire-and-forget SQLite write (durability)
    if store:
        def _write(conn, sid, cid, s, t, m, sh, th, mh, mc, sb, frz, tcid, tm, ttk, tcc):
            upsert_prefix_chain(conn, sid, cid, sys=s, tools=t, messages=m,
                               sys_hashes=sh, tools_hashes=th, messages_hashes=mh,
                               msg_count=mc, safe_breakpoints=sb, is_frozen=frz)
            insert_chain_token(conn, tcid, msg_count=tm, tokens=ttk, char_count=tcc)

        store.enqueue_sqlite_write(
            _write, session_id, ca.chain_id,
            sys_json, tools_json, msgs_json,
            sys_hashes, tools_hashes, msgs_hashes,
            msg_cnt, ca.safe_breakpoints, ca.is_frozen,
            ca.chain_id, msg_cnt, chain_tokens, ca.char_count,
        )

    tag = "+ new" if ca.action == "create" else "~ update"
    s_len = len(ca.sys) if ca.sys else len(ctx.sys)
    t_len = len(ca.tools) if ca.tools else len(ctx.tools)
    m_len = len(ca.messages) if ca.messages else len(ctx.messages)
    print(green(
        f"  [chain] {tag}: {ca.chain_id[:16]}... "
        f"(s={s_len} t={t_len} m={m_len}, hit={ca.hit_count})"
    ))
