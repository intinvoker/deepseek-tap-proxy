"""PrefixLock 内部工具函数：char 计数、JSON 解析、链操作、persist 辅助。"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from deepseek_tap_proxy.pipeline.chain_action import ChainAction
from deepseek_tap_proxy.shared.chain import Chain, ChainMoment, find_anchor_moment


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


def chain_hash_view(chain, hashes_key: str, store=None) -> dict:
    """Wrap a chain (Chain object or dict) as a match_prefix-compatible view.

    Supports both ``Chain`` dataclass objects (attribute access) and
    plain dicts (bracket/key access) for backward compatibility.

    ``store`` is a MemoryStore (optional — currently unused, provided
    for future signature compatibility).
    """
    if isinstance(chain, dict):
        return {
            "chain_id": chain["chain_id"],
            "hash_chain": chain.get(hashes_key, ""),
        }
    # Chain dataclass — hash data via attributes
    hash_chain = getattr(chain, hashes_key, "")
    return {
        "chain_id": chain.chain_id,
        "hash_chain": hash_chain,
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

    from deepseek_tap_proxy.shared.db import upsert_prefix_chain
    from deepseek_tap_proxy.shared.colors import green
    from deepseek_tap_proxy.shared.store import MemoryStore

    rt = ctx.request.get("runtime")
    store: MemoryStore | None = rt.store if rt else None
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
            is_frozen=False,
            family_key=ctx.family_key,
        )

    # 2. Fire-and-forget SQLite write (durability)
    if store:
        def _write(conn, sid, cid, s, t, m, sh, th, mh, mc, sb, frz):
            upsert_prefix_chain(conn, sid, cid, sys=s, tools=t, messages=m,
                               sys_hashes=sh, tools_hashes=th, messages_hashes=mh,
                               msg_count=mc, safe_breakpoints=sb, is_frozen=frz)

        store.enqueue_sqlite_write(
            _write, session_id, ca.chain_id,
            sys_json, tools_json, msgs_json,
            sys_hashes, tools_hashes, msgs_hashes,
            msg_cnt, ca.safe_breakpoints, False,
            op_name="chain_persist",
        )

    tag = "+ new" if ca.action == "create" else "~ update"
    s_len = len(ca.sys) if ca.sys else len(ctx.sys)
    t_len = len(ca.tools) if ca.tools else len(ctx.tools)
    m_len = len(ca.messages) if ca.messages else len(ctx.messages)
    print(green(
        f"  [chain] {tag}: {ca.chain_id[:16]}... "
        f"(s={s_len} t={t_len} m={m_len}, hit={ca.hit_count})"
    ))


def create_chain_moment(ctx, store) -> None:
    """Create a ChainMoment after pipeline completes and persist it.

    Determines op from chat_type (compaction) or chain_action, then
    appends the moment to the matched chain (if available) or to a new
    chain via ``_ensure_chain_exists``.

    For compaction requests, ``persist_chain_nowait`` is skipped, so
    the chain is created here via ``_ensure_chain_exists``.
    """
    if ctx.body_readonly is None:
        return

    op = "compact" if ctx.chat_type == "compaction" else (
        ctx.chain_action.action if ctx.chain_action else "create"
    )
    moment = ChainMoment(
        messages_count=len(ctx.sys) + len(ctx.tools) + len(ctx.messages),
        op=op,
        total_tokens=(ctx.persist.total_input_tokens or 0) + (ctx.persist.output_tokens or 0),
        cache_hit_tokens=ctx.persist.cache_hit_tokens or 0,
        cache_miss_tokens=ctx.persist.cache_miss_tokens or 0,
        char_count=compute_char_count(ctx.sys + ctx.tools + ctx.messages),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Append to matched_chain's moments list for future anchor lookups
    if isinstance(ctx.matched_chain, Chain) and op in ("update", "compact"):
        ctx.matched_chain.moments.append(moment)

    # Persist moment via store
    chain_id = None
    if isinstance(ctx.matched_chain, Chain):
        chain_id = ctx.matched_chain.chain_id
    elif isinstance(ctx.matched_chain, dict):
        chain_id = ctx.matched_chain.get("chain_id")
    if not chain_id and ctx.chain_action:
        chain_id = ctx.chain_action.chain_id

    if not chain_id:
        return

    # Ensure chain exists in store before adding moment.
    # For normal requests, persist_chain_nowait already created the chain.
    # For compaction requests, persist_chain_nowait is skipped (chat_type != "compaction"
    # guard), so we create the chain here.
    _ensure_chain_exists(ctx, store, chain_id)

    store.add_moment(chain_id, moment)


def _ensure_chain_exists(ctx, store, chain_id: str) -> None:
    """Ensure a chain record exists in the store, creating it if needed.

    Handles both in-memory (0 I/O) and SQLite durability.
    ``store.upsert_chain`` is idempotent — it only overwrites when
    ``msg_count`` is strictly greater.

    For compaction requests, ``persist_chain_nowait`` is skipped
    (chat_type guard), so this function ensures the chain is persisted.
    """
    import json

    chain_record = store.get_chain_record(chain_id)
    if chain_record:
        return  # Already exists in memory

    sys_json = json.dumps(ctx.sys, ensure_ascii=False)
    tools_json = json.dumps(ctx.tools, ensure_ascii=False)
    msgs_json = json.dumps(ctx.messages, ensure_ascii=False)
    sys_hashes = ctx.sys_hash_chain
    tools_hashes = ctx.tools_hash_chain
    msgs_hashes = ctx.messages_hash_chain
    msg_cnt = len(ctx.sys) + len(ctx.tools) + len(ctx.messages)

    # 1. In-memory write (0 I/O)
    store.upsert_chain(
        ctx.session_id or "", chain_id,
        sys=sys_json, tools=tools_json, messages=msgs_json,
        sys_hashes=sys_hashes, tools_hashes=tools_hashes,
        messages_hashes=msgs_hashes,
        msg_count=msg_cnt,
        family_key=ctx.family_key,
    )

    # 2. Fire-and-forget SQLite write (durability)
    from deepseek_tap_proxy.shared.db import upsert_prefix_chain

    def _write(conn, sid, cid, s, t, m, sh, th, mh, mc):
        upsert_prefix_chain(conn, sid, cid, sys=s, tools=t, messages=m,
                           sys_hashes=sh, tools_hashes=th, messages_hashes=mh,
                           msg_count=mc)

    store.enqueue_sqlite_write(
        _write, ctx.session_id or "", chain_id,
        sys_json, tools_json, msgs_json,
        sys_hashes, tools_hashes, msgs_hashes,
        msg_cnt,
        op_name="chain_persist",
    )


# ═══════════════════════════════════════════════════════════════════════
#  公共前缀冻结链（fork 场景 est 锚点）
# ═══════════════════════════════════════════════════════════════════════


def _find_anchor_across_chains(
    chains: list[Chain], pref_hit: int,
) -> ChainMoment | None:
    """在所有同族链的 moments 中查找 ≤ pref_hit 的锚点。

    用于 find_anchor_moment(best_chain) 失败时的跨链 fallback。
    冻结链 (is_frozen) 的 moment 在此处被消费。
    """
    best: ChainMoment | None = None
    for chain in chains:
        anchor = find_anchor_moment(chain.moments, pref_hit)
        if anchor is not None:
            if best is None or anchor.messages_count > best.messages_count:
                best = anchor
    return best


def _ensure_frozen_chain(
    store, ref_chain: Chain, pref_hit: int,
    session_id: str, family_key: str,
) -> ChainMoment | None:
    """基于参考链创建公共前缀 frozen chain 并返回估算 moment。

    Frozen chain 特性：
    - is_frozen=True，永不被后续请求 UPDATE
    - 同 family_key，get_family_chains() 自然返回
    - 仅有 1 个估算 moment（token 按 messages_count 比例分割）
    - 创建前检查 idempotent（已在同 family + 同长度时不重复创建）

    精度：假设消息 token/char 均匀分布，偏差约 1-3%（已知限制）。
    """
    from deepseek_tap_proxy.shared.hash import msg_hash
    from deepseek_tap_proxy.shared.prefix import make_chain_id

    if not ref_chain.moments:
        return None

    smallest = min(ref_chain.moments, key=lambda m: m.messages_count)
    if smallest.messages_count <= pref_hit:
        return None  # _find_anchor 应该已经找到了

    # ── 1. 检查 idempotent：是否已存在同 family + 同长度的 frozen chain ──
    all_chains = store.get_family_chains(session_id, family_key)
    for c in all_chains:
        total = len(c.sys) + len(c.tools) + len(c.messages)
        if total == pref_hit and c.chain_id != ref_chain.chain_id:
            anchor = find_anchor_moment(c.moments, pref_hit)
            if anchor is not None:
                return anchor  # 已存在，直接消费

    # ── 2. 估算前缀 moment ──
    ratio = pref_hit / smallest.messages_count
    estimated = ChainMoment(
        messages_count=pref_hit,
        op="create",
        total_tokens=int(smallest.total_tokens * ratio),
        cache_hit_tokens=int((smallest.cache_hit_tokens or 0) * ratio),
        cache_miss_tokens=int((smallest.cache_miss_tokens or 0) * ratio),
        char_count=int((smallest.char_count or 0) * ratio),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # ── 3. 提取前缀消息（按 sys → tools → messages 顺序截断） ──
    prefix_sys: list[dict] = []
    prefix_tools: list[dict] = []
    prefix_msgs: list[dict] = []
    remaining = pref_hit
    for item in ref_chain.sys:
        if remaining <= 0:
            break
        prefix_sys.append(item)
        remaining -= 1
    for item in ref_chain.tools:
        if remaining <= 0:
            break
        prefix_tools.append(item)
        remaining -= 1
    for item in ref_chain.messages:
        if remaining <= 0:
            break
        prefix_msgs.append(item)
        remaining -= 1

    # ── 4. 计算 hash chain ──
    sys_hashes_str = ",".join(msg_hash(m) for m in prefix_sys)
    tools_hashes_str = ",".join(msg_hash(m) for m in prefix_tools)
    msgs_hashes_str = ",".join(msg_hash(m) for m in prefix_msgs)
    _all_hashes = (
        sys_hashes_str.split(",") if sys_hashes_str else []
    ) + (
        tools_hashes_str.split(",") if tools_hashes_str else []
    ) + (
        msgs_hashes_str.split(",") if msgs_hashes_str else []
    )
    frozen_chain_id = make_chain_id(_all_hashes)

    # ── 5. 持久化 frozen chain + moment ──
    store.upsert_chain(
        session_id, frozen_chain_id,
        sys=json.dumps(prefix_sys),
        tools=json.dumps(prefix_tools),
        messages=json.dumps(prefix_msgs),
        sys_hashes=sys_hashes_str,
        tools_hashes=tools_hashes_str,
        messages_hashes=msgs_hashes_str,
        msg_count=pref_hit,
        is_frozen=True,
        family_key=family_key,
        moments=json.dumps([{
            "messages_count": estimated.messages_count,
            "op": estimated.op,
            "total_tokens": estimated.total_tokens,
            "cache_hit_tokens": estimated.cache_hit_tokens,
            "cache_miss_tokens": estimated.cache_miss_tokens,
            "char_count": estimated.char_count,
            "timestamp": estimated.timestamp,
        }]),
    )

    return estimated
