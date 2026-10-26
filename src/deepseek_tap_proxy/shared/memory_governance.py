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

MAX_MOMENTS_DEFAULT = 100


def estimate_bytes(
    sessions: dict[str, "SessionRecord"],
    chains: dict[str, dict[str, list["ChainRecord"]]],
) -> int:
    """确定性字节估算。给定同样输入永远返回同样值。

    SessionRecord: 固定 512 bytes each
    ChainRecord:   sum(JSON field lengths) + 256 overhead
    """
    total = 0
    total += len(sessions) * SESSION_OVERHEAD_BYTES

    for fmap in chains.values():
        for clist in fmap.values():
            for c in clist:
                total += CHAIN_OVERHEAD_BYTES
                total += len(c.sys)
                total += len(c.tools)
                total += len(c.messages)
                total += len(c.moments)
                total += len(c.sys_hashes)
                total += len(c.tools_hashes)
                total += len(c.messages_hashes)

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
) -> int:
    """移除 last_seen 超过 ttl_hours 的 session 及其 chains。

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

    return len(stale_ids)


def evict_lru(
    sessions: dict[str, "SessionRecord"],
    chains: dict[str, dict[str, list["ChainRecord"]]],
    *,
    max_memory: int,
) -> int:
    """LRU 淘汰：按 last_seen 升序淘汰 chains，直到内存估算 ≤ max_memory。

    仅操作 dict（0 I/O），不涉及 SQLite。
    返回本次淘汰的 chain 数。
    """
    current = estimate_bytes(sessions, chains)
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

        # 计算该 chain 的字节数（用于增量减法）
        chain_bytes = (CHAIN_OVERHEAD_BYTES + len(chain.sys) + len(chain.tools)
                       + len(chain.messages) + len(chain.moments)
                       + len(chain.sys_hashes) + len(chain.tools_hashes)
                       + len(chain.messages_hashes))

        fmap[fk] = [c for c in clist if c.chain_id != chain.chain_id]
        if not fmap[fk]:
            del fmap[fk]
        if not fmap:
            chains.pop(sid, None)
        current -= chain_bytes
        evicted += 1

    return evicted
