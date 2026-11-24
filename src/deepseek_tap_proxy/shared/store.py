"""MemoryStore — 运行时主存储（0 I/O dict），SQLite 降级为异步落盘后备。

Architecture::

    启动:  SQLite.load_all → MemoryStore (dicts)
    运行时: Stage → MemoryStore (0 I/O) → Forward (即刻)
    落盘:  MemoryStore → fire-and-forget → SQLite
    关闭:  flush pending → SQLite

所有 Stage 通过 MemoryStore 读写数据，从不直接接触 SQLite。
"""
from __future__ import annotations

import asyncio
import json
import logging
import orjson
import sqlite3
import time as _time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import xxhash

from deepseek_tap_proxy.shared.db import set_meta, _exec
from deepseek_tap_proxy.shared.memory_governance import (
    estimate_bytes, evict_stale_sessions, evict_lru, truncate_moments,
    EvictionStats,
)

log = logging.getLogger("deepseek-tap")


# ═══════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SessionRecord:
    session_id: str
    client_type: str = ""
    anchor_msg_index: int = 0
    max_hash_chain: str = ""
    hash_chain_length: int = 0
    start_time: str = ""
    last_seen_time: str = ""
    total_requests: int = 1
    canonical_sys: str = ""
    canonical_tools: str = ""


@dataclass
class HashIndex:
    """Lazily-built hash index for a chain's sys/tools/messages."""
    sys_hashes: list[str] = field(default_factory=list)
    tools_hashes: list[str] = field(default_factory=list)
    messages_hashes: list[str] = field(default_factory=list)


@dataclass
class ChainRecord:
    chain_id: str
    session_id: str = ""
    sys: str = "[]"
    tools: str = "[]"
    messages: str = "[]"
    sys_hashes: str = ""
    tools_hashes: str = ""
    messages_hashes: str = ""
    msg_count: int = 0
    safe_breakpoints: str | None = None
    total_input_tokens: int = 0
    is_frozen: bool = False
    created_at: str = ""
    last_seen: str = ""
    moments: str = "[]"

    def as_dict(self) -> dict:
        """Return dict for compatibility with old db-returned Row objects."""
        return {
            "chain_id": self.chain_id,
            "session_id": self.session_id,
            "sys": self.sys,
            "tools": self.tools,
            "messages": self.messages,
            "sys_hashes": self.sys_hashes,
            "tools_hashes": self.tools_hashes,
            "messages_hashes": self.messages_hashes,
            "msg_count": self.msg_count,
            "safe_breakpoints": self.safe_breakpoints,
            "total_input_tokens": self.total_input_tokens,
            "is_frozen": self.is_frozen,
            "created_at": self.created_at,
            "last_seen": self.last_seen,
            "moments": self.moments,
        }


# ═══════════════════════════════════════════════════════════════════════
#  写者线程常量
# ═══════════════════════════════════════════════════════════════════════

# 单次写失败后的重试次数（不包含首次尝试）
_MAX_RETRIES = 2
# 每次重试前的休眠秒数（指数退避）
_RETRY_DELAYS = [0.1, 0.5]  # 100ms → 500ms

# ═══════════════════════════════════════════════════════════════════════
#  MemoryStore
# ═══════════════════════════════════════════════════════════════════════


class MemoryStore:
    """In-memory primary data store with async SQLite backend.

    All reads and writes target in-process dicts (0 I/O).
    SQLite is updated asynchronously for durability.
    """

    def __init__(self, db=None,
                 ttl_hours: int = 168,
                 max_memory_bytes: int = 200 * 1024 * 1024,
                 max_moments: int = 100,
                 enable_governance: bool = True,
                 db_commit_interval: float = 1.0,
                 db_batch_max: int = 500,
                 db_queue_max: int = 1000,
                 events_writer=None,
                 gc_trim_interval: float = 300.0,
                 _gc_collect=None,
                 _malloc_trim=None) -> None:
        self._db = db  # SQLite Database instance for fire-and-forget writes
        self._lock = asyncio.Lock()

        # session_id → SessionRecord
        self._sessions: dict[str, SessionRecord] = {}

        # session_id → family_key → list[ChainRecord]  (sorted by last_seen desc)
        self._chains: dict[str, dict[str, list[ChainRecord]]] = {}

        # chain_id → ChainRecord 快速索引（O(1) 定位）。重放逐事件 upsert /
        # get_chain_record 时避免全链扫描（20k 事件 10.5s → 毫秒级）。淘汰后重建。
        self._chain_index: dict[str, ChainRecord] = {}

        # (session_id, chat_type) → chain_id 粘性链，防止 tools 变化导致的雪崩
        self._sticky: dict[str, dict[str, str]] = {}

        # ── canonical 按 (session_id, chain_id) 隔离 ──
        # chain_id 为空表示「该 session 的默认链」，键归一为 session_id。
        # 为何不能用 SessionRecord 的字段：Claude Code 子代理复用主会话
        # session-id，session 级单份存储会让子代理被主代理整体覆写。
        self._canonicals: dict[tuple[str, str], tuple[str, str]] = {}

        # 事件日志写入器（权威事实源）。写路径状态变更同步 append，供任务 5
        # 重放器重建内存状态。为 None 时写路径事件化为 no-op。
        self._events = events_writer

        # Single-writer queue for SQLite (prevents "database is locked")
        self._write_queue: asyncio.Queue | None = None
        self._writer_task: asyncio.Task | None = None

        # write stats — 写熔断可复盘能力
        self._write_successes: int = 0
        self._write_failures: int = 0
        self._last_write_success: str = ""
        self._last_write_failure: str = ""
        # op_name → 失败次数（按操作类型聚合）
        self._write_failures_by_op: dict[str, int] = {}
        # 异常类型 → 失败次数（按错误类型聚合）
        self._write_failures_by_error: dict[str, int] = {}
        # 批量 commit 次数（批量事务路径）
        self._batches_committed: int = 0

        # 注入点 + 冷却期
        self._now = lambda: datetime.now(timezone.utc)
        self._last_eviction_check: float = 0.0
        self._eviction_check_interval: int = 300  # 5 minutes

        # L4 GC/trim（N3）：周期回收（pymalloc arenas 不归还 OS → 长跑水位只升不降）
        import gc as _gc
        self._gc_trim_enabled: bool = True
        self._gc_trim_interval: float = gc_trim_interval
        self._last_gc_trim: float = 0.0
        self._gc_collect = _gc_collect or _gc.collect

        def _default_malloc_trim() -> None:
            """glibc malloc_trim(0)（容器 Linux）；非 glibc 平台静默跳过。"""
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except (OSError, AttributeError):
                pass

        self._malloc_trim = _malloc_trim or _default_malloc_trim

        # 配置
        self.enable_governance = enable_governance
        self.ttl_hours = ttl_hours
        self.max_memory_bytes = max_memory_bytes
        self.max_moments = max_moments

        # 批量写者配置
        self._db_commit_interval = db_commit_interval
        self._db_batch_max = db_batch_max
        self._db_queue_max = db_queue_max

        # 统计（累积）
        self._stale_evictions_total: int = 0
        self._lru_evictions_total: int = 0
        self._moments_truncated_total: int = 0
        self._lazy_reload_count: int = 0

        if db is not None:
            self._write_queue = asyncio.Queue(maxsize=db_queue_max)
            self._start_writer()

    def maybe_maintain(self) -> EvictionStats:
        """每次请求入口调用。TTL 淘汰 → LRU 淘汰。返回统计信息。

        TTL 淘汰有 5 分钟冷却期。
        LRU 检查每次执行。

        当 enable_governance=False 时立即返回空统计（调试回放用）。
        """
        if not self.enable_governance:
            return EvictionStats()

        now = self._now()
        stale_evicted = 0

        before = set(self._sessions)

        # L2: TTL（冷却期保护）
        if now.timestamp() - self._last_eviction_check >= self._eviction_check_interval:
            stale_evicted = evict_stale_sessions(
                self._sessions, self._chains,
                now=now, ttl_hours=self.ttl_hours,
                canonicals=self._canonicals,
            )
            self._stale_evictions_total += stale_evicted
            self._last_eviction_check = now.timestamp()

        # L3: LRU
        lru_evicted = evict_lru(
            self._sessions, self._chains,
            max_memory=self.max_memory_bytes,
            canonicals=self._canonicals,
        )
        self._lru_evictions_total += lru_evicted

        # L4: GC/malloc_trim 周期（N3：打破长跑 RSS 只升不降——pymalloc arenas 不归还 OS）
        if (self._gc_trim_enabled
                and now.timestamp() - self._last_gc_trim >= self._gc_trim_interval):
            self._last_gc_trim = now.timestamp()
            self._gc_collect()
            self._malloc_trim()

        # 淘汰从 _chains 移除了链 → 重建 chain_id 索引，保持 get_chain_record 一致。
        # 仅在实际发生淘汰时重建（O(n) 偶发，正常路径零开销）。
        if stale_evicted or lru_evicted:
            evicted = before - set(self._sessions)
            for sid in evicted:
                self._sticky.pop(sid, None)
            self._rebuild_chain_index()

        return EvictionStats(
            stale_evicted=stale_evicted,
            lru_evicted=lru_evicted,
        )

    def _rebuild_chain_index(self) -> None:
        """从 _chains 重建 chain_id → rec 索引（淘汰后调用，保持 O(1) 查询一致）。"""
        index: dict[str, ChainRecord] = {}
        for fmap in self._chains.values():
            for clist in fmap.values():
                for c in clist:
                    index[c.chain_id] = c
        self._chain_index = index

    def _find_chain_by_scan(self, chain_id: str) -> "ChainRecord | None":
        """索引 miss 时的全链扫描回退（直接改 _chains 的测试/未来路径兜底）。

        命中即回填索引（自愈不变量）。正常路径（链均经 upsert_chain/load 建索引）
        不会走到这里 —— 纯优化缓存，非正确性前提。
        """
        for fmap in self._chains.values():
            for clist in fmap.values():
                for c in clist:
                    if c.chain_id == chain_id:
                        self._chain_index[chain_id] = c
                        return c
        return None

    def _get_conn(self):
        """Get SQLite connection for async writes."""
        if self._db is None:
            return None
        return self._db.conn if hasattr(self._db, 'conn') else self._db

    # ── Sessions ─────────────────────────────────────────────────

    def get_session(self, session_id: str) -> SessionRecord | None:
        return self._sessions.get(session_id)

    def get_candidate_sessions(self, client_type: str) -> list[dict]:
        """Find active sessions matching client_type, ordered by last_seen desc."""
        candidates = [
            s for s in self._sessions.values()
            if s.client_type == client_type and s.hash_chain_length > 0
        ]
        candidates.sort(key=lambda s: s.last_seen_time, reverse=True)
        return [
            {
                "session_id": s.session_id,
                "client_type": s.client_type,
                "max_hash_chain": s.max_hash_chain,
                "hash_chain_length": s.hash_chain_length,
                "last_seen_time": s.last_seen_time,
            }
            for s in candidates[:50]
        ]

    async def upsert_session(self, session_id: str, client_type: str,
                             anchor_msg_index: int = 0,
                             curr_hashes: list[str] | None = None,
                             now: str = "",
                             _schedule_persist: bool = True) -> SessionRecord:
        """Get existing or create new session. Always updates last_seen/time.

        When ``_schedule_persist`` is True (default), a fire-and-forget SQLite
        write is queued so that the requests table FK constraint is satisfied.
        """
        if not now:
            now = datetime.now(timezone.utc).isoformat()

        existing = self._sessions.get(session_id)
        if existing:
            existing.last_seen_time = now
            existing.total_requests += 1
            if curr_hashes and len(curr_hashes) > existing.hash_chain_length:
                existing.max_hash_chain = ",".join(curr_hashes)
                existing.hash_chain_length = len(curr_hashes)
            return existing
        else:
            hash_str = ",".join(curr_hashes) if curr_hashes else ""
            rec = SessionRecord(
                session_id=session_id, client_type=client_type,
                anchor_msg_index=anchor_msg_index,
                max_hash_chain=hash_str,
                hash_chain_length=len(curr_hashes) if curr_hashes else 0,
                start_time=now, last_seen_time=now, total_requests=1,
            )
            self._sessions[session_id] = rec
            # Fire-and-forget session to SQLite
            if _schedule_persist:
                await self._enqueue_session_persist(session_id, client_type, anchor_msg_index,
                                                    hash_str, len(curr_hashes) if curr_hashes else 0, now)
            return rec

    async def _enqueue_session_persist(self, session_id: str, client_type: str,
                                        anchor_msg_index: int, hash_str: str,
                                        hash_len: int, now: str) -> None:
        """Fire-and-forget: persist session to SQLite."""
        from deepseek_tap_proxy.shared.db import get_or_create_session_by_explicit_id

        def _write(conn, sid, ct, ami, hs, hl, t):
            # Check if session already exists
            row = _exec(conn, "SELECT 1 FROM sessions WHERE session_id = %(sid)s", {"sid": sid}).fetchone()
            if row:
                _exec(
                    conn,
                    "UPDATE sessions SET last_seen_time = %(t)s, total_requests = total_requests + 1 WHERE session_id = %(sid)s",
                    {"t": t, "sid": sid})
            else:
                _exec(
                    conn,
                    "INSERT INTO sessions (session_id, client_type, anchor_msg_index, max_hash_chain, hash_chain_length, start_time, last_seen_time, total_requests) VALUES (%(sid)s, %(ct)s, %(ami)s, %(hs)s, %(hl)s, %(t)s, %(t)s, 1)",
                    {"sid": sid, "ct": ct, "ami": ami, "hs": hs, "hl": hl, "t": t})
            # 不 commit — 由 writer 线程批量统一 commit

        await self.enqueue_sqlite_write(_write, session_id, client_type, anchor_msg_index,
                                        hash_str, hash_len, now,
                                        op_name="session_persist")

    def update_session_hash_chain(self, session_id: str, curr_hashes: list[str]) -> None:
        existing = self._sessions.get(session_id)
        if existing and len(curr_hashes) > existing.hash_chain_length:
            existing.max_hash_chain = ",".join(curr_hashes)
            existing.hash_chain_length = len(curr_hashes)

    # ── Canonical (SysToolsLock) — (session_id, chain_id) 级 ────────

    @staticmethod
    def _canonical_key(session_id: str, chain_id: str) -> tuple[str, str]:
        """空 chain_id 归一为该 session 的默认链。"""
        return (session_id, chain_id or session_id)

    @staticmethod
    def _normalize_loaded_canonicals(
        rows: dict[tuple[str, str], tuple[str, str]],
    ) -> dict[tuple[str, str], tuple[str, str]]:
        """装载归一：把库里 ``chain_id == ""`` 的行映射到内存键 ``(sid, sid)``。

        DB 层契约是「存归一后的值」（见 ``shared/db.py`` chain_canonical 表注释），
        但库中可能残留空 chain_id 的行 —— 漏归一的旧写入方、或迁移前的历史形态。
        空串是 ``(sid, sid)`` 的同义写法：不归一则装载后内存 key 为 ``(sid, '')``，
        而 ``get_canonical(sid, "")`` 经 ``_canonical_key`` 查的是 ``(sid, sid)``
        ⇒ 永远查不回，主代理重启后静默重设基线。

        归一放在此处（而非 ``load_all_chain_canonicals``）的理由：``_canonical_key``
        是归一规则的单一真源，属于内存层；DB 函数保持「存什么读什么」，既让
        调试回放能读到原始行，也避免规则在两层各存一份。

        同链两行（缺陷产物）时**显式 chain_id 行优先** —— checkpoint 投影的就是
        该形态，且收敛必须确定，不得由 SELECT 的行序决定。
        """
        out: dict[tuple[str, str], tuple[str, str]] = {}
        for (sid, cid), val in rows.items():
            if cid:
                out[MemoryStore._canonical_key(sid, cid)] = val
        for (sid, cid), val in rows.items():
            if not cid:
                out.setdefault(MemoryStore._canonical_key(sid, cid), val)
        return out

    def get_canonical(self, session_id: str, chain_id: str = "") -> dict[str, str]:
        """取该链冻结的 sys/tools；无则返回两个空串。"""
        sys_, tools_ = self._canonicals.get(
            self._canonical_key(session_id, chain_id), ("", ""))
        return {"sys": sys_, "tools": tools_}

    def set_canonical(self, session_id: str, chain_id: str = "",
                      sys: str = "", tools: str = "",
                      *, _persist_event: bool = True) -> None:
        """冻结该链的 sys/tools。空值不覆盖已有值（与既有语义一致）。"""
        key = self._canonical_key(session_id, chain_id)
        cur_sys, cur_tools = self._canonicals.get(key, ("", ""))
        new_sys = sys or cur_sys
        new_tools = tools or cur_tools
        self._canonicals[key] = (new_sys, new_tools)
        # 事件记录合并后终态（非入参）：set(sys=A) → set(tools=B) 两条事件，
        # 第二条必须保留 sys=A，重放才不丢字段。sys/tools 均空时不发。
        if _persist_event and self._events is not None and (sys or tools):
            self._events.append("canonical_set", {
                "session_id": session_id,
                "chain_id": chain_id,
                "sys": new_sys,
                "tools": new_tools,
            })

    # ── Prefix chains ────────────────────────────────────────────

    @staticmethod
    def compute_family_key(sys_hashes_str: str, tools_hashes_str: str) -> str:
        """From comma-separated hash strings, produce a 16-char family key.

        Streaming xxh3_64 over each individual hash (no separator ambiguity —
        each hash is fixed 16 hex chars).  Equivalent to
        ``Context._recompute_family_key``.
        """
        h = xxhash.xxh3_64()
        for s in sys_hashes_str.split(","):
            if s:
                h.update(s.encode())
        for t in tools_hashes_str.split(","):
            if t:
                h.update(t.encode())
        return h.hexdigest()[:16]

    def get_session_chains(self, session_id: str) -> list[dict]:
        """Get ALL prefix chains for a session as ``Chain`` objects (flattened)."""
        from deepseek_tap_proxy.shared.chain import Chain, ChainMoment

        family_map = self._chains.get(session_id, {})
        chains: list[Chain] = []
        for clist in family_map.values():
            for c in clist:
                chains.append(self._record_to_chain(c))
        chains.sort(key=lambda c: c.created_at, reverse=True)
        return chains

    def get_family_chains(self, session_id: str, family_key: str) -> list:
        """Get prefix chains for a specific branch family.

        If memory miss, lazy reload from SQLite.
        """
        from deepseek_tap_proxy.shared.chain import Chain, ChainMoment

        clist = self._chains.get(session_id, {}).get(family_key, [])
        if clist:
            return [self._record_to_chain(c) for c in clist]

        # L4: Lazy reload from SQLite
        if self._db is not None:
            records = self._lazy_load_chains(session_id, family_key)
            if records:
                self._lazy_reload_count += 1
                return [self._record_to_chain(r) for r in records]

        return []

    # ── Sticky chain ─────────────────────────────────────────

    def get_sticky_chain(self, session_id: str, chat_type: str) -> str | None:
        """返回 (session_id, chat_type) 的粘性链 ID。无则返回 None。"""
        return self._sticky.get(session_id, {}).get(chat_type)

    def set_sticky_chain(self, session_id: str, chat_type: str, chain_id: str):
        """设置 (session_id, chat_type) 的粘性链 ID。"""
        self._sticky.setdefault(session_id, {})[chat_type] = chain_id

    def get_chain_hashes(self, chain_id: str) -> HashIndex | None:
        """Return lazily-built hash index for a chain.

        Scans all families for the given chain_id and parses the
        comma-separated hash strings into lists.
        """
        for family_map in self._chains.values():
            for clist in family_map.values():
                for rec in clist:
                    if rec.chain_id == chain_id:
                        return HashIndex(
                            sys_hashes=rec.sys_hashes.split(",") if rec.sys_hashes else [],
                            tools_hashes=rec.tools_hashes.split(",") if rec.tools_hashes else [],
                            messages_hashes=rec.messages_hashes.split(",") if rec.messages_hashes else [],
                        )
        return None

    @staticmethod
    def _record_to_chain(rec: 'ChainRecord') -> 'Chain':
        """Convert a ChainRecord to a Chain object with parsed fields."""
        from deepseek_tap_proxy.shared.chain import Chain, ChainMoment
        from deepseek_tap_proxy.shared.stages.helpers import json_list

        # Parse moments JSON -> list[ChainMoment]
        moments = []
        try:
            raw_moments = orjson.loads(rec.moments or "[]")
            for m in raw_moments:
                moments.append(ChainMoment(
                    messages_count=m.get("messages_count", 0),
                    op=m.get("op", ""),
                    total_tokens=m.get("total_tokens", 0),
                    cache_hit_tokens=m.get("cache_hit_tokens", 0),
                    cache_miss_tokens=m.get("cache_miss_tokens", 0),
                    char_count=m.get("char_count", 0),
                    timestamp=m.get("timestamp", ""),
                ))
        except (json.JSONDecodeError, TypeError, AttributeError):
            moments = []

        return Chain(
            chain_id=rec.chain_id,
            session_id=rec.session_id,
            sys=json_list(rec.sys),
            tools=json_list(rec.tools),
            messages=json_list(rec.messages),
            created_at=rec.created_at,
            moments=moments,
            sys_hashes=rec.sys_hashes,
            tools_hashes=rec.tools_hashes,
            messages_hashes=rec.messages_hashes,
        )

    def _lazy_load_chains(self, session_id: str, family_key: str) -> list["ChainRecord"]:
        """从 SQLite 按需加载 chains 并恢复到内存。

        仅当 session 已在内存中时才回载（启动窗口守卫）。
        """
        # 守卫：session 不在内存中 → 不在启动窗口内 → 拒绝回载
        if session_id not in self._sessions:
            return []

        conn = self._get_conn()
        if conn is None:
            return []

        try:
            from deepseek_tap_proxy.shared.db import get_session_chains as db_get_chains
            rows = db_get_chains(conn, session_id)
        except Exception as exc:
            log.warning("MemoryStore lazy reload failed for session %s (fk=%s): %s",
                        session_id, family_key, exc)
            return []

        records: list["ChainRecord"] = []
        for d in rows:
            rec = ChainRecord(
                chain_id=d.get("chain_id", ""),
                session_id=d.get("session_id", ""),
                sys=d.get("sys", "[]"),
                tools=d.get("tools", "[]"),
                messages=d.get("messages", "[]"),
                sys_hashes=d.get("sys_hashes", ""),
                tools_hashes=d.get("tools_hashes", ""),
                messages_hashes=d.get("messages_hashes", ""),
                msg_count=d.get("msg_count", 0),
                safe_breakpoints=d.get("safe_breakpoints"),
                total_input_tokens=d.get("total_input_tokens", 0),
                is_frozen=bool(d.get("is_frozen", 0)),
                created_at=d.get("created_at", ""),
                last_seen=d.get("last_seen", ""),
                moments=d.get("moments", "[]"),
            )
            fk = self.compute_family_key(rec.sys_hashes, rec.tools_hashes)
            if fk == family_key:
                self._chains.setdefault(session_id, {}).setdefault(fk, []).append(rec)
                self._chain_index[rec.chain_id] = rec
                records.append(rec)

        return records

    def upsert_chain(self, session_id: str, chain_id: str, *,
                     sys: str = "[]", tools: str = "[]", messages: str = "[]",
                     sys_hashes: str = "", tools_hashes: str = "",
                     messages_hashes: str = "",
                     msg_count: int = 0,
                     safe_breakpoints: list[int] | None = None,
                     is_frozen: bool = False,
                     family_key: str = "",
                     moments: str = "[]",
                     total_input_tokens: int = 0,
                     created_at: str = "",
                     last_seen: str = "",
                     _persist_event: bool = True) -> ChainRecord:
        """Insert or update a prefix chain.

        When ``family_key`` is empty it is computed from ``sys_hashes``
        and ``tools_hashes`` (backward compat — all callers should
        eventually pass ``ctx.family_key``).

        ``created_at`` / ``last_seen`` — 事件重放透传（空则用 now）。live 路径
        不传保持原语义；重放路径从事件 data 传入，避免所有链塌缩为同一时间戳。

        ``_persist_event`` — 事件重放开关。重放器重建内存状态时传 False，
        避免重放写路径再次 append 事件造成无限循环（事件日志是权威事实源）。
        """
        now = datetime.now(timezone.utc).isoformat()
        created_at = created_at or now
        last_seen = last_seen or now
        sb_json = orjson.dumps(safe_breakpoints).decode() if safe_breakpoints is not None else None

        if not family_key:
            family_key = self.compute_family_key(sys_hashes, tools_hashes)

        # 查找既有链：chain_id 全局唯一 → O(1) 索引（替代三重循环全链扫描）。
        # 注意：此处不做扫描回退 —— 新建链索引 miss 若触发全链扫描会退化为 O(n²)。
        # 所有链均经本方法 / load 建索引（不变量成立），直接改 _chains 的测试
        # 路径不调用 upsert_chain，无需兜底。
        found: ChainRecord | None = self._chain_index.get(chain_id)
        found_family: str = ""

        changed = False
        if found is not None:
            # Update only if new msg_count > old and not frozen ——
            # 守卫失败（未实际更新）不发事件，重放才不会改写冻结链。
            if msg_count > found.msg_count and not found.is_frozen:
                found.sys = sys
                found.tools = tools
                found.messages = messages
                found.sys_hashes = sys_hashes
                found.tools_hashes = tools_hashes
                found.messages_hashes = messages_hashes
                found.msg_count = msg_count
                found.safe_breakpoints = sb_json
                found.total_input_tokens = total_input_tokens
                found.last_seen = last_seen
                # Preserve existing moments (append-only field)
                if moments != "[]" and moments != found.moments:
                    found.moments = moments
                changed = True
            rec = found
        else:
            # Create new
            rec = ChainRecord(
                chain_id=chain_id, session_id=session_id,
                sys=sys, tools=tools, messages=messages,
                sys_hashes=sys_hashes, tools_hashes=tools_hashes,
                messages_hashes=messages_hashes,
                msg_count=msg_count, safe_breakpoints=sb_json,
                is_frozen=is_frozen, created_at=created_at, last_seen=last_seen,
                moments=moments, total_input_tokens=total_input_tokens,
            )
            fmap = self._chains.setdefault(session_id, {})
            fmap.setdefault(family_key, []).append(rec)
            # Keep sorted by last_seen desc
            fmap[family_key].sort(key=lambda c: c.last_seen, reverse=True)
            self._chain_index[rec.chain_id] = rec
            found_family = family_key
            changed = True

        # 事件记录变更后终态（从 rec 取，非入参）；family_key 为链实际所在 family
        # （更新分支 = 找到时的 family，新建分支 = 插入时的 family）。守卫失败不发。
        if changed and _persist_event and self._events is not None:
            # 更新分支链不迁移 family：索引定位到 rec 后，从 _chains 找出其所在 family
            if not found_family:
                for fk, clist in self._chains.get(rec.session_id, {}).items():
                    if any(c is rec for c in clist):
                        found_family = fk
                        break
            self._events.append("chain_upsert", {
                "session_id": rec.session_id, "chain_id": rec.chain_id,
                "family_key": found_family,
                "sys": rec.sys, "tools": rec.tools, "messages": rec.messages,
                "sys_hashes": rec.sys_hashes, "tools_hashes": rec.tools_hashes,
                "messages_hashes": rec.messages_hashes,
                "msg_count": rec.msg_count, "is_frozen": rec.is_frozen,
                "safe_breakpoints": rec.safe_breakpoints,
                "moments": rec.moments,
                "created_at": rec.created_at, "last_seen": rec.last_seen,
                "total_input_tokens": rec.total_input_tokens,
            })
        return rec

    # ── Chain moments ──────────────────────────────────────────────

    def get_chain_record(self, chain_id: str):
        """Find a ChainRecord by chain_id, returning None if not found."""
        rec = self._chain_index.get(chain_id)
        if rec is None:
            rec = self._find_chain_by_scan(chain_id)
        return rec

    async def add_moment(self, chain_id: str, moment: 'ChainMoment', *,
                         _persist_event: bool = True) -> None:
        """Append a ChainMoment to an existing chain.

        Updates in-memory record (0 I/O) and queues a fire-and-forget
        SQLite write targeting the ``moments`` JSON column.

        ``_persist_event`` — 事件重放开关。重放器重建内存状态时传 False，
        避免重放 chain_moment 时再次 append 事件造成无限循环（与
        set_canonical/upsert_chain 对称）。
        """
        from deepseek_tap_proxy.shared.chain import ChainMoment

        # 定位链：chain_id 全局唯一 → O(1) 索引（替代三重循环全链扫描）。
        # 索引 miss 回退扫描（直接改 _chains 的测试路径兜底），命中即回填索引。
        c = self._chain_index.get(chain_id)
        if c is None:
            c = self._find_chain_by_scan(chain_id)
        if c is None:
            return
        # Parse existing moments
        try:
            moments = orjson.loads(c.moments or "[]")
        except (json.JSONDecodeError, TypeError):
            moments = []
        # Append new moment as dict
        moments.append({
            "messages_count": moment.messages_count,
            "op": moment.op,
            "total_tokens": moment.total_tokens,
            "cache_hit_tokens": moment.cache_hit_tokens,
            "cache_miss_tokens": moment.cache_miss_tokens,
            "char_count": moment.char_count,
            "timestamp": moment.timestamp,
        })
        c.moments = orjson.dumps(moments).decode()
        # L5: 截断内存中的 moments（DB 永久留痕，SQLite 写入完整列表）
        n = truncate_moments(c, max_moments=self.max_moments)
        self._moments_truncated_total += n
        # 事件日志：记录 moment（同步入队，无需 await）。受
        # _persist_event 开关控制（防重放循环）。
        if _persist_event and self._events is not None:
            self._events.append("chain_moment", {
                "chain_id": chain_id, "moment": {
                    "messages_count": moment.messages_count,
                    "op": moment.op, "total_tokens": moment.total_tokens,
                    "cache_hit_tokens": moment.cache_hit_tokens,
                    "cache_miss_tokens": moment.cache_miss_tokens,
                    "char_count": moment.char_count, "timestamp": moment.timestamp,
                },
            })
        # 传入完整 moments 列表（截断前），DB 保留完整历史
        await self._enqueue_moment_persist(chain_id, moments)

    async def _enqueue_moment_persist(self, chain_id: str, moments: list[dict]) -> None:
        """Fire-and-forget: update the moments JSON column in SQLite."""
        def _write(conn, cid, m_json):
            _exec(
                conn,
                "UPDATE prefix_chains SET moments = %(m_json)s WHERE chain_id = %(cid)s",
                {"m_json": orjson.dumps(m_json).decode(), "cid": cid},
            )
            # 不 commit — 由 writer 线程批量统一 commit

        await self.enqueue_sqlite_write(_write, chain_id, moments, op_name="chain_moment")

    # ── Persist queue ────────────────────────────────────────────

    def _start_writer(self) -> None:
        """Start the single-writer background task for SQLite."""
        if self._writer_task is not None:
            return
        self._writer_task = asyncio.create_task(self._writer_loop())

    async def _writer_loop(self) -> None:
        """Single background task: drain queue, run writes in a dedicated thread.

        Uses a single long-lived thread with its own SQLite connection,
        avoiding "SQLite objects created in a thread can only be used
        in that same thread" and "database is locked" issues.
        """
        db = self._db
        if db is None:
            return
        import queue as _sync_queue
        _q = _sync_queue.Queue(maxsize=self._db_queue_max)
        _done = asyncio.Event()

        def _thread_worker():
            conn = db.conn  # thread-local
            _consecutive_failures = 0
            _MAX_CONSECUTIVE_FAILURES = 20

            def _unpack(item):
                if len(item) == 3:
                    return item[0], item[1], item[2]
                return item[0], item[1], ""

            def _record_failure(op_name, exc):
                nonlocal _consecutive_failures
                _consecutive_failures += 1
                self._write_failures += 1
                ts = datetime.now(timezone.utc).isoformat()
                self._last_write_failure = ts
                reason = type(exc).__qualname__
                self._write_failures_by_error[reason] = \
                    self._write_failures_by_error.get(reason, 0) + 1
                if op_name:
                    self._write_failures_by_op[op_name] = \
                        self._write_failures_by_op.get(op_name, 0) + 1
                log.error(
                    "MemoryStore: SQLite write failed after %d attempt(s): %s | "
                    "op=%s queue_depth=%d successes=%d failures=%d consecutive=%d",
                    _MAX_RETRIES + 1, exc, op_name or "?",
                    _q.qsize(), self._write_successes, self._write_failures,
                    _consecutive_failures,
                )

            def _replay_one_by_one(batch):
                """整批失败 → 逐条重放（每条独立 commit），坏条目不拖垮整批。"""
                for item in batch:
                    sync_fn, args, op_name = _unpack(item)
                    last_exc = None
                    for attempt in range(_MAX_RETRIES + 1):
                        try:
                            sync_fn(conn, *args)
                            conn.commit()
                            self._write_successes += 1
                            self._last_write_success = datetime.now(timezone.utc).isoformat()
                            last_exc = None
                            _consecutive_failures = 0
                            break
                        except Exception as exc:
                            last_exc = exc
                            try:
                                conn.rollback()
                            except Exception:
                                pass
                            if attempt < _MAX_RETRIES:
                                _time.sleep(_RETRY_DELAYS[attempt])
                    if last_exc:
                        _record_failure(op_name, last_exc)

            def _run_batch(batch):
                """执行整批（各 sync_fn 不 commit）→ 单 commit；失败 rollback + 逐条重放。"""
                try:
                    for item in batch:
                        sync_fn, args, op_name = _unpack(item)
                        sync_fn(conn, *args)
                    conn.commit()
                    self._write_successes += len(batch)
                    self._last_write_success = datetime.now(timezone.utc).isoformat()
                    _consecutive_failures = 0
                    self._batches_committed += 1
                except Exception:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    _replay_one_by_one(batch)

            while True:
                # 收集一批：先等第一条（超时=无任务，空闲继续）；低流量单条立即落盘
                try:
                    item = _q.get(timeout=self._db_commit_interval)
                except _sync_queue.Empty:
                    continue
                if item is None:  # shutdown sentinel
                    break
                batch = [item]
                stopping = False
                while len(batch) < self._db_batch_max:
                    try:
                        nxt = _q.get_nowait()
                    except _sync_queue.Empty:
                        break
                    if nxt is None:
                        stopping = True
                        break
                    batch.append(nxt)
                _run_batch(batch)
                if stopping:
                    break

                # 连接退化回收（保留原逻辑）
                if _consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    log.warning(
                        "MemoryStore: writer connection degraded (%d consecutive failures), recycling...",
                        _consecutive_failures)
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = db.conn
                    _consecutive_failures = 0

            _q.task_done()

        import threading
        t = threading.Thread(target=_thread_worker, daemon=True, name="sqlite-writer")
        t.start()

        try:
            while True:
                item = await self._write_queue.get()
                if item is None:  # sentinel
                    break
                # 有界同步队列满时阻塞（to_thread 中执行，不卡事件循环），
                # 停止消费 async 队列 → 背压传导回 enqueue_sqlite_write 侧。
                await asyncio.to_thread(_q.put, item)
        finally:
            _q.put(None)  # stop thread worker
            t.join(timeout=5)

    async def enqueue_sqlite_write(self, sync_fn, *args, op_name: str = "") -> None:
        """Fire-and-forget: queue a SQLite write for the background writer.

        Bounded queue with async backpressure — blocks (awaits) when the
        queue is full so no write is dropped under extreme load.
        """
        if self._write_queue is None:
            return
        await self._write_queue.put((sync_fn, args, op_name))

    async def flush(self) -> None:
        """Drain pending SQLite writes without stopping the writer.

        Best-effort: waits for the async queue to drain and gives the
        writer thread one commit interval to flush a trailing partial
        batch.  Call before opening a new SQLite connection to read
        committed data in tests (WAL snapshot isolation).
        """
        if self._write_queue is None:
            return
        for _ in range(1000):
            if self._write_queue.empty():
                break
            await asyncio.sleep(0.001)
        # 给 writer 一个 commit_interval 让尾批（不足 batch_max）落盘
        await asyncio.sleep(self._db_commit_interval + 0.05)

    async def flush_pending(self) -> None:
        """Await all pending SQLite writes + stop writer (called during shutdown)."""
        if self._write_queue is None:
            return
        for _ in range(1000):
            if self._write_queue.empty():
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(self._db_commit_interval + 0.05)
        if self._writer_task:
            await self._write_queue.put(None)  # sentinel
            try:
                await asyncio.wait_for(self._writer_task, timeout=10)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._writer_task.cancel()
            self._writer_task = None

    # ══════════════════════════════════════════════════════════════════
    #  Write stats (observability / health)
    # ══════════════════════════════════════════════════════════════════

    def get_write_stats(self) -> dict:
        """Return write failure metrics for health endpoint / post-mortem.

        Includes per-operation-type failure counts so operators can see
        which table path is failing (``req_phase1``, ``req_phase2``,
        ``session_persist``, ``chain_persist``, etc.).
        """
        queue_depth = 0
        if self._write_queue is not None:
            try:
                queue_depth = self._write_queue.qsize()
            except NotImplementedError:
                pass
        return {
            "writes_attempted": self._write_successes + self._write_failures,
            "writes_succeeded": self._write_successes,
            "writes_failed": self._write_failures,
            "failure_rate": round(
                self._write_failures / max(self._write_successes + self._write_failures, 1), 4
            ),
            "last_success": self._last_write_success,
            "last_failure": self._last_write_failure,
            "failures_by_op": dict(self._write_failures_by_op),
            "failures_by_error": dict(self._write_failures_by_error),
            "queue_depth": queue_depth,
            "batches_committed": self._batches_committed,
            "queue_max": self._db_queue_max,
        }

    def get_memory_stats(self) -> dict:
        """返回内存治理统计（供 /health 消费）。"""
        est = estimate_bytes(self._sessions, self._chains, self._canonicals)
        chain_count = sum(len(clist) for fmap in self._chains.values() for clist in fmap.values())
        return {
            "estimated_bytes": est,
            "max_bytes": self.max_memory_bytes,
            "usage_pct": round(est / max(self.max_memory_bytes, 1) * 100, 1),
            "sessions": len(self._sessions),
            "chains": chain_count,
            "stale_evictions_total": self._stale_evictions_total,
            "lru_evictions_total": self._lru_evictions_total,
            "moments_truncated_total": self._moments_truncated_total,
            "lazy_reload_count": self._lazy_reload_count,
            "last_eviction_check": datetime.fromtimestamp(
                self._last_eviction_check, tz=timezone.utc
            ).isoformat() if self._last_eviction_check else "",
        }

    def snapshot(self) -> tuple[list["SessionRecord"], list["ChainRecord"],
                                dict[tuple[str, str], tuple[str, str]]]:
        """事件循环线程内取内存态副本（周期 checkpoint 用，I1）。

        同步构建（无 await）→ 与 handler 在同一线程交错时仍原子，无
        "dictionary changed size during iteration" 竞态；``replace`` 复制记录对象，
        隔离 ``to_thread`` 写 SQLite 期间 handler 对 live 记录的 in-place 变更
        （upsert_chain / upsert_session 都是原地改字段）。返回的副本与 live
        dict 解耦，可安全传入 ``create_checkpoint(snapshot=...)``。

        返回 ``(sessions, chains, canonicals)``；canonicals 为 ``_canonicals`` 的
        浅拷贝（值是不可变 2-元组，浅拷贝即足够隔离迭代）。
        """
        return (
            [replace(s) for s in self._sessions.values()],
            [replace(c)
             for fmap in self._chains.values()
             for clist in fmap.values()
             for c in clist],
            dict(self._canonicals),
        )

    def create_checkpoint(self, conn: sqlite3.Connection, checkpoint_offset: int,
                          snapshot: tuple[list["SessionRecord"], list["ChainRecord"],
                                          dict[tuple[str, str], tuple[str, str]]] | None = None) -> None:
        """把当前内存状态重建为 SQLite 投影（TRUNCATE + INSERT）。

        checkpoint 是加速冷启动的投影缓存，非权威 —— 事件日志才是事实源。
        checkpoint_offset 记录已投影到的事件日志字节偏移（checkpoint 即 AOF 压缩点）。

        canonical 投影到 ``chain_canonical``（真源），不再写 ``sessions.canonical_*``
        —— 那两个列是 session 级单份存储的遗留，子代理复用主会话 session-id 时会被
        主代理整体覆写（列保留不删，仅停写）。

        ``snapshot`` — 事件循环线程内取的 (sessions, chains, canonicals) 副本
        （见 ``snapshot()``）。为 None 时直接迭代 live 内存（单线程场景 / 测试兼容）。
        传快照后，本方法迭代快照而非 live dict —— 可安全放入 ``asyncio.to_thread``
        （I1：高流量下 handler 在事件循环线程改 live dict 不产生跨线程迭代竞态）。

        失败原子性：整个 DELETE + INSERT + set_meta 在单一事务内；任一语句失败
        即 rollback 全部并重抛，不留残留写事务（避免 RESERVED 锁阻塞 writer）。

        并发契约：本方法同步直接写 conn，调用方必须保证无并发写者窗口（建议
        调用前先 flush 写队列 / 关闭时天然满足）；否则会与 sqlite-writer 线程
        竞争锁并可能产生 ghost row。不要改走 writer 队列 —— checkpoint 需同步
        结果，且会复杂化单写者逻辑。

        事件循环：同步全量重建可能阻塞事件循环，调用方需以 ``asyncio.to_thread``
        包裹（周期 checkpoint 任务勿在事件循环线程直接调用）。
        """
        sessions = snapshot[0] if snapshot is not None else list(self._sessions.values())
        chains = snapshot[1] if snapshot is not None else [
            c for fmap in self._chains.values()
            for clist in fmap.values()
            for c in clist
        ]
        canonicals = snapshot[2] if snapshot is not None else dict(self._canonicals)
        try:
            conn.execute("DELETE FROM prefix_chains")
            conn.execute("DELETE FROM sessions")
            conn.execute("DELETE FROM chain_canonical")
            for rec in sessions:
                _exec(
                    conn,
                    """INSERT INTO sessions (session_id, client_type, anchor_msg_index,
                        max_hash_chain, hash_chain_length, start_time, last_seen_time,
                        total_requests)
                       VALUES (%(sid)s, %(ct)s, %(ami)s, %(hs)s, %(hl)s, %(st)s,
                        %(lst)s, %(tr)s)""",
                    {"sid": rec.session_id, "ct": rec.client_type, "ami": rec.anchor_msg_index,
                     "hs": rec.max_hash_chain, "hl": rec.hash_chain_length, "st": rec.start_time,
                     "lst": rec.last_seen_time, "tr": rec.total_requests},
                )
            # canonical 真源投影：键即内存 _canonicals 的 key（chain_id 已归一）
            for (canon_sid, canon_cid), (canon_sys, canon_tools) in canonicals.items():
                _exec(
                    conn,
                    """INSERT INTO chain_canonical (session_id, chain_id, sys, tools)
                       VALUES (%(sid)s, %(cid)s, %(sys)s, %(tools)s)""",
                    {"sid": canon_sid, "cid": canon_cid,
                     "sys": canon_sys, "tools": canon_tools},
                )
            for c in chains:
                _exec(
                    conn,
                    """INSERT INTO prefix_chains (session_id, chain_id, sys, tools,
                        messages, sys_hashes, tools_hashes, messages_hashes,
                        msg_count, safe_breakpoints, total_input_tokens, is_frozen,
                        created_at, last_seen, moments)
                       VALUES (%(sid)s, %(cid)s, %(sys)s, %(tools)s, %(messages)s,
                        %(sys_hashes)s, %(tools_hashes)s, %(messages_hashes)s,
                        %(msg_count)s, %(sb)s, %(tit)s, %(frozen)s,
                        %(created_at)s, %(last_seen)s, %(moments)s)""",
                    {"sid": c.session_id, "cid": c.chain_id, "sys": c.sys, "tools": c.tools,
                     "messages": c.messages, "sys_hashes": c.sys_hashes,
                     "tools_hashes": c.tools_hashes, "messages_hashes": c.messages_hashes,
                     "msg_count": c.msg_count, "sb": c.safe_breakpoints,
                     "tit": c.total_input_tokens, "frozen": 1 if c.is_frozen else 0,
                     "created_at": c.created_at, "last_seen": c.last_seen,
                     "moments": c.moments},
                )
            set_meta(conn, "checkpoint_offset", str(checkpoint_offset), commit=False)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ── Lifecycle ───────────────────────────────────────────────

    @classmethod
    async def load_from(cls, db, _db=None, load_window_hours: int = 168,
                        enable_governance: bool = True,
                        db_commit_interval: float = 1.0,
                        db_batch_max: int = 500,
                        db_queue_max: int = 1000,
                        now: str | None = None,
                        events_path: str | None = None,
                        events_writer=None,
                        max_memory_bytes: int = 200 * 1024 * 1024,
                        ttl_hours: int = 168,
                        max_moments: int = 100) -> MemoryStore:
        """Populate MemoryStore from SQLite checkpoint + event log replay.

        事件日志为权威：先读 SQLite checkpoint 基线（含 checkpoint_offset），
        再从 offset 增量重放 chain-events.jsonl。canonical 从 canonical_set
        事件恢复（确定性）；无事件路径下由 checkpoint 基线（chain_canonical 表）
        批量装载。

        ``load_window_hours`` — DEPRECATED。保留参数以兼容调用方签名，但已无效：
        checkpoint 是完整投影，加载全量（不再按窗口过滤）。
        ``enable_governance`` — enable TTL + LRU eviction (default True, set False for replay debugging)。
        ``events_path`` — 事件日志路径（chain-events.jsonl）。为 None 时跳过重放
        （旧版无事件日志 / 首次升级），canonical 由 checkpoint 基线提供。
        ``events_writer`` — 新 store 的事件写入器（启动后继续 append 后续事件）。

        轮转边界：archive 已在任务 7 实现（checkpoint 时归档旧日志段，base 归 0）；
        replay_events 现跨 base + 全部轮转段重放（from_offset 只对 base 生效）。
        轮转 → checkpoint 之间的崩溃窗口已关闭 —— 崩溃时日志完整，重启重放
        不丢轮转段状态变更。
        """
        store = cls(db=_db or db, enable_governance=enable_governance,
                    db_commit_interval=db_commit_interval,
                    db_batch_max=db_batch_max,
                    db_queue_max=db_queue_max,
                    events_writer=events_writer,
                    max_memory_bytes=max_memory_bytes,
                    ttl_hours=ttl_hours,
                    max_moments=max_moments)
        reference_now = now or datetime.now(timezone.utc).isoformat()

        def _load():
            conn = db.conn if hasattr(db, 'conn') else db

            # 清理上次崩溃遗留的 pending 请求（非启发式，崩溃恢复常驻逻辑）
            try:
                deleted = _exec(
                    conn,
                    "DELETE FROM requests WHERE record_state = %(state)s",
                    {"state": "pending"},
                ).rowcount
                conn.commit()
                if deleted:
                    log.info("MemoryStore: cleaned up %d orphaned pending record(s)", deleted)
            except sqlite3.OperationalError:
                pass

            # checkpoint_offset → 事件增量重放起点（无 meta 表 = 无 checkpoint → 0）
            offset = 0
            from deepseek_tap_proxy.shared.db import get_meta
            try:
                offset = int(get_meta(conn, "checkpoint_offset") or "0")
            except (TypeError, ValueError, sqlite3.OperationalError):
                offset = 0

            # checkpoint 基线：sessions + prefix_chains（完整投影，不按窗口过滤）
            # canonical 不在 sessions 行内（已迁 chain_canonical，见下方装载）。
            try:
                for row in conn.execute("SELECT * FROM sessions").fetchall():
                    d = dict(row)
                    store._sessions[d["session_id"]] = SessionRecord(
                        session_id=d.get("session_id", ""),
                        client_type=d.get("client_type", ""),
                        anchor_msg_index=d.get("anchor_msg_index", 0),
                        max_hash_chain=d.get("max_hash_chain", ""),
                        hash_chain_length=d.get("hash_chain_length", 0),
                        start_time=d.get("start_time", reference_now),
                        last_seen_time=d.get("last_seen_time", reference_now),
                        total_requests=d.get("total_requests", 1),
                    )
            except sqlite3.OperationalError:
                pass
            # canonical 基线（无事件路径）：一次批量装载 chain_canonical → _canonicals。
            # 有事件路径不走这里 —— 事件为权威，缺失的键由 load_from 尾部的
            # _canonical_fallback 从同一张表补（顺序：基线 → 事件 → 补空）。
            # 装载归一把库中残留的空 chain_id 历史行折到内存键（见 helper docstring）。
            if not events_path:
                try:
                    from deepseek_tap_proxy.shared.db import load_all_chain_canonicals
                    store._canonicals.update(store._normalize_loaded_canonicals(
                        load_all_chain_canonicals(conn)))
                except sqlite3.OperationalError:
                    pass  # 旧库未迁移（无 chain_canonical 表）→ canonical 缺失，非致命
            try:
                rows = conn.execute(
                    "SELECT * FROM prefix_chains ORDER BY last_seen DESC",
                ).fetchall()
                for row in rows:
                    d = dict(row)
                    rec = ChainRecord(
                        chain_id=d.get("chain_id", ""),
                        session_id=d.get("session_id", ""),
                        sys=d.get("sys", "[]"),
                        tools=d.get("tools", "[]"),
                        messages=d.get("messages", "[]"),
                        sys_hashes=d.get("sys_hashes", ""),
                        tools_hashes=d.get("tools_hashes", ""),
                        messages_hashes=d.get("messages_hashes", ""),
                        msg_count=d.get("msg_count", 0),
                        safe_breakpoints=d.get("safe_breakpoints"),
                        total_input_tokens=d.get("total_input_tokens", 0),
                        is_frozen=bool(d.get("is_frozen", 0)),
                        created_at=d.get("created_at", reference_now),
                        last_seen=d.get("last_seen", reference_now),
                        moments=d.get("moments", "[]"),
                    )
                    store._chains.setdefault(rec.session_id, {}) \
                        .setdefault(store.compute_family_key(rec.sys_hashes, rec.tools_hashes), []) \
                        .append(rec)
                    # 维护 chain_id → rec 索引（重放 upsert_chain / get_chain_record 走 O(1)）
                    store._chain_index[rec.chain_id] = rec
            except sqlite3.OperationalError:
                pass

            _loaded_chains = sum(
                len(recs)
                for _fams in store._chains.values()
                for recs in _fams.values()
            )
            log.info(
                "MemoryStore loaded from DB: %d sessions, %d chains",
                len(store._sessions), _loaded_chains,
            )
            return offset

        offset = await asyncio.to_thread(_load)

        # 事件日志增量重放（权威）：从 checkpoint_offset 起重放未投影事件。
        # canonical 从 canonical_set 事件恢复（确定性），替代旧启发式猜测。
        # 与 _load 对称走 to_thread，避免大日志阻塞事件循环。
        from deepseek_tap_proxy.shared.events import replay_events
        if events_path:
            try:
                last_seq = await asyncio.to_thread(
                    replay_events, events_path, offset, store,
                )
                if last_seq >= 0:
                    log.info(
                        "MemoryStore replayed events from offset %d: %d sessions, %d chains",
                        offset, len(store._sessions),
                        sum(len(r) for f in store._chains.values() for r in f.values()),
                    )
            except Exception as exc:
                log.warning("MemoryStore event replay failed: %s", exc)

        # canonical 兜底：仅事件路径需要（事件可能未覆盖某些链，如事件日志起点
        # 之后新建但无 canonical_set 的链）。无事件时基线已批量装载
        # chain_canonical，兜底完全冗余 → 跳过。
        # 键控后无需再按 _sessions 裁剪：直接按 (session_id, chain_id) 补空，
        # 已由事件提供的键保持事件值（事件权威）。
        if events_path:
            def _canonical_fallback():
                conn = store._get_conn()
                if conn is None:
                    return
                try:
                    from deepseek_tap_proxy.shared.db import load_all_chain_canonicals
                    db_canon = load_all_chain_canonicals(conn)
                except sqlite3.OperationalError:
                    return
                for key, val in store._normalize_loaded_canonicals(db_canon).items():
                    if key not in store._canonicals:
                        store._canonicals[key] = val

            await asyncio.to_thread(_canonical_fallback)

        if hasattr(db, 'close'):
            db.close()
        return store

    # ── Requests 表持久化（统一走单写者队列）─────────────────────

    # 哨兵：区分「未提供快照」和「快照值为 None」
    _NO_SNAPSHOT = object()

    async def enqueue_request_phase1(self, ctx) -> None:
        """Fire-and-forget: insert request record (Phase 1, state='pending')."""
        from deepseek_tap_proxy.shared.db import insert_request_phase1

        _api_format = ctx.persist.api_format
        _trace_category = ctx.persist.trace_category
        _msg_hash_chain = ctx.msg_hash_chain

        def _write(conn, _ctx, _af, _tc, _mhc):
            if _af:
                _ctx.persist.api_format = _af
            if _tc:
                _ctx.persist.trace_category = _tc
            _ctx.persist.msg_hash_chain = _mhc
            insert_request_phase1(conn, _ctx, commit=False)

        await self.enqueue_sqlite_write(_write, ctx, _api_format, _trace_category, _msg_hash_chain,
                                        op_name="req_phase1")

    async def enqueue_request_phase2(self, ctx, *,
                                     jsonl_path: str = "",
                                     lock_type_snapshot: str | None | object = _NO_SNAPSHOT,
                                     is_modified_snapshot: bool = False,
                                     mod_note_snapshot: str = "") -> None:
        """Fire-and-forget: update request record (Phase 2, record_state='completed').

        When ``lock_type_snapshot`` is ``None`` it means "no lock" — distinct
        from the sentinel which means "use ctx.persist directly".
        Pass explicit snapshot values to avoid async closure races.
        """
        from deepseek_tap_proxy.shared.db import update_request_phase2 as _update, _SENTINEL_NO_SNAPSHOT as _DB_SENTINEL

        # Map our sentinel to db.py's sentinel (different module, different object())
        _lt = _DB_SENTINEL if lock_type_snapshot is self._NO_SNAPSHOT else lock_type_snapshot

        def _write(conn, _ctx, _jp, _lt_inner, _im, _mn):
            _update(conn, _ctx, jsonl_path=_jp,
                    lock_type_snapshot=_lt_inner,
                    is_modified_snapshot=_im,
                    mod_note_snapshot=_mn,
                    commit=False)

        await self.enqueue_sqlite_write(
            _write, ctx, jsonl_path,
            _lt, is_modified_snapshot, mod_note_snapshot,
            op_name="req_phase2",
        )