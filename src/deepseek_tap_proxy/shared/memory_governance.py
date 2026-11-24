"""Memory governance: eviction, estimation, and stats.
Deterministic functions — no I/O, no asyncio.
(truncate_moments mutates rec.moments in place.)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import orjson

if TYPE_CHECKING:
    from deepseek_tap_proxy.shared.store import SessionRecord, ChainRecord


@dataclass
class EvictionStats:
    """maybe_maintain() 的返回值。"""
    stale_evicted: int = 0
    lru_evicted: int = 0


# ── Constants ──

SESSION_OVERHEAD_BYTES = 512
CHAIN_OVERHEAD_BYTES = 256
# 一条 canonical 条目 = 键/值两个 2-元组的对象头 + 引用
CANONICAL_OVERHEAD_BYTES = 120

MAX_MOMENTS_DEFAULT = 100


def _str_cost(s: str) -> int:
    """字符串真实成本：CPython str 对象头 + UTF-8 字节宽（中文 3B/字）。"""
    return 49 + len(s.encode("utf-8"))


def _chain_cost(c: "ChainRecord") -> int:
    return (275  # dataclass 结构（槽+引用+字段头）
            + _str_cost(c.chain_id) + _str_cost(c.session_id) + _str_cost(c.sys)
            + _str_cost(c.tools) + _str_cost(c.messages) + _str_cost(c.moments)
            + _str_cost(c.sys_hashes) + _str_cost(c.tools_hashes)
            + _str_cost(c.messages_hashes) + _str_cost(c.safe_breakpoints or "")
            + _str_cost(c.created_at) + _str_cost(c.last_seen))


def _session_cost(s: "SessionRecord") -> int:
    return (300 + _str_cost(s.session_id) + _str_cost(s.client_type)
            + _str_cost(s.max_hash_chain) + _str_cost(s.start_time)
            + _str_cost(s.last_seen_time) + _str_cost(s.canonical_sys)
            + _str_cost(s.canonical_tools))


def _canonical_cost(key: tuple[str, str], val: tuple[str, str]) -> int:
    """一条 canonical 的字节成本（键/值两个 2-元组 + 4 个字符串）。

    canonical 的 sys/tools 是完整 system prompt / 工具定义 JSON，数十 KB 级 ——
    不计入则 LRU 看不到它，_canonicals 会只涨不降。
    """
    return (CANONICAL_OVERHEAD_BYTES
            + _str_cost(key[0]) + _str_cost(key[1])
            + _str_cost(val[0]) + _str_cost(val[1]))


def drop_session_canonicals(
    canonicals: dict[tuple[str, str], tuple[str, str]],
    session_id: str,
) -> int:
    """移除该 session 名下全部 canonical 条目，返回释放的字节数。

    session 被淘汰时调用（否则 _canonicals 残留漏账、只涨不降）。
    """
    freed = 0
    for key in [k for k in canonicals if k[0] == session_id]:
        freed += _canonical_cost(key, canonicals[key])
        del canonicals[key]
    return freed


def estimate_bytes(
    sessions: dict[str, "SessionRecord"],
    chains: dict[str, dict[str, list["ChainRecord"]]],
    canonicals: dict[tuple[str, str], tuple[str, str]] | None = None,
) -> int:
    """确定性字节估算。给定同样输入永远返回同样值。

    校准口径：UTF-8 字节 + str 对象头 + 容器结构开销（按字符数粗估会显著低估）。
    目标：误差 <30%（校准测试 test_memory_governance_calibration）。

    ``canonicals`` — ``MemoryStore._canonicals``（可选；None 视为空，兼容旧调用点）。
    """
    total = 0
    for rec in sessions.values():
        total += _session_cost(rec)
    for fmap in chains.values():
        for clist in fmap.values():
            for c in clist:
                total += _chain_cost(c)
    if canonicals:
        for key, val in canonicals.items():
            total += _canonical_cost(key, val)
    # dict/list 容器结构开销（每 dict ~96B + 每 entry 8B）
    total += 96 + 8 * len(sessions)
    for fmap in chains.values():
        total += 96 + 8 * len(fmap)
        for clist in fmap.values():
            total += 96 + 8 * len(clist)
    if canonicals:
        total += 96 + 8 * len(canonicals)
    return total


def truncate_moments(rec: "ChainRecord", max_moments: int = MAX_MOMENTS_DEFAULT) -> int:
    """保留最近 ``max_moments`` 条 moments。返回截断条数。

    只修改 rec.moments 的字符串值，不创建新对象。
    """
    try:
        moments = orjson.loads(rec.moments or "[]")
    except (orjson.JSONDecodeError, TypeError):
        return 0

    if len(moments) <= max_moments:
        return 0

    truncated = len(moments) - max_moments
    rec.moments = orjson.dumps(moments[-max_moments:]).decode()
    return truncated


def evict_stale_sessions(
    sessions: dict[str, "SessionRecord"],
    chains: dict[str, dict[str, list["ChainRecord"]]],
    *,
    now: datetime,
    ttl_hours: int = 168,
    canonicals: dict[tuple[str, str], tuple[str, str]] | None = None,
) -> int:
    """移除 last_seen 超过 ttl_hours 的 session 及其 chains（+ canonicals）。

    仅操作 dict（0 I/O），不涉及 SQLite。
    返回本次淘汰的 session 数。
    """
    cutoff = now - timedelta(hours=ttl_hours)
    stale_ids: list[str] = []

    for sid, rec in sessions.items():
        if not rec.last_seen_time:
            continue
        try:
            last_seen = datetime.fromisoformat(rec.last_seen_time)
        except (ValueError, TypeError):
            continue
        if last_seen < cutoff:
            stale_ids.append(sid)

    for sid in stale_ids:
        sessions.pop(sid, None)
        chains.pop(sid, None)
        if canonicals is not None:
            drop_session_canonicals(canonicals, sid)

    return len(stale_ids)


def evict_lru(
    sessions: dict[str, "SessionRecord"],
    chains: dict[str, dict[str, list["ChainRecord"]]],
    *,
    max_memory: int,
    canonicals: dict[tuple[str, str], tuple[str, str]] | None = None,
) -> int:
    """LRU 淘汰：按 last_seen 升序淘汰 chains，直到内存估算 ≤ max_memory。

    仅操作 dict（0 I/O），不涉及 SQLite。
    返回本次淘汰的 chain 数。
    """
    current = estimate_bytes(sessions, chains, canonicals)
    if current <= max_memory:
        return 0

    # 收集 entries，同时记录每个 chain 的字节开销
    all_entries: list[tuple[str, str, "ChainRecord", str]] = []
    for sid, fmap in chains.items():
        for fk, clist in fmap.items():
            for c in clist:
                all_entries.append((sid, fk, c, c.last_seen))

    all_entries.sort(key=lambda e: e[3])

    evicted = 0
    for sid, fk, chain, _ts in all_entries:
        if current <= max_memory:
            break
        # 检查 chain 是否仍在 chains 中（前面的淘汰可能已清除整个 session）
        fmap = chains.get(sid)
        if fmap is None:
            continue
        clist = fmap.get(fk, [])
        if chain not in clist:
            continue

        # 计算该 chain 的字节数（用于增量减法）——与 estimate_bytes 同口径（校准）
        chain_bytes = _chain_cost(chain)

        fmap[fk] = [c for c in clist if c.chain_id != chain.chain_id]
        if not fmap[fk]:
            del fmap[fk]
        if not fmap:
            chains.pop(sid, None)
            # N2：session 已无任何链 → canonical 一并回收（否则残留漏账——R2 同款病）
            if sid in sessions:
                current -= _session_cost(sessions[sid])
                sessions.pop(sid, None)
            if canonicals is not None:
                current -= drop_session_canonicals(canonicals, sid)
        current -= chain_bytes
        evicted += 1

    return evicted
