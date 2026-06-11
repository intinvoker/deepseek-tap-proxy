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
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

import xxhash

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
        }


@dataclass
class ChainTokenRecord:
    chain_id: str
    msg_count: int
    tokens: int
    char_count: int = 0
    created_at: str = ""


# ═══════════════════════════════════════════════════════════════════════
#  MemoryStore
# ═══════════════════════════════════════════════════════════════════════

class MemoryStore:
    """In-memory primary data store with async SQLite backend.

    All reads and writes target in-process dicts (0 I/O).
    SQLite is updated asynchronously for durability.
    """

    def __init__(self, db=None) -> None:
        self._db = db  # SQLite Database instance for fire-and-forget writes
        self._lock = asyncio.Lock()

        # session_id → SessionRecord
        self._sessions: dict[str, SessionRecord] = {}

        # session_id → family_key → list[ChainRecord]  (sorted by last_seen desc)
        self._chains: dict[str, dict[str, list[ChainRecord]]] = {}

        # chain_id → list[ChainTokenRecord]  (sorted by msg_count asc)
        self._token_history: dict[str, list[ChainTokenRecord]] = {}

        # Single-writer queue for SQLite (prevents "database is locked")
        self._write_queue: asyncio.Queue | None = None
        self._writer_task: asyncio.Task | None = None
        if db is not None:
            self._write_queue = asyncio.Queue()
            self._start_writer()

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

    def upsert_session(self, session_id: str, client_type: str,
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
                self._enqueue_session_persist(session_id, client_type, anchor_msg_index,
                                              hash_str, len(curr_hashes) if curr_hashes else 0, now)
            return rec

    def _enqueue_session_persist(self, session_id: str, client_type: str,
                                  anchor_msg_index: int, hash_str: str,
                                  hash_len: int, now: str) -> None:
        """Fire-and-forget: persist session to SQLite."""
        from deepseek_tap_proxy.shared.db import get_or_create_session_by_explicit_id

        def _write(conn, sid, ct, ami, hs, hl, t):
            # Check if session already exists
            row = conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (sid,)).fetchone()
            if row:
                conn.execute(
                    "UPDATE sessions SET last_seen_time = ?, total_requests = total_requests + 1 WHERE session_id = ?",
                    (t, sid))
            else:
                conn.execute(
                    "INSERT INTO sessions (session_id, client_type, anchor_msg_index, max_hash_chain, hash_chain_length, start_time, last_seen_time, total_requests) VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
                    (sid, ct, ami, hs, hl, t, t))
            conn.commit()

        self.enqueue_sqlite_write(_write, session_id, client_type, anchor_msg_index,
                                  hash_str, hash_len, now)

    def update_session_hash_chain(self, session_id: str, curr_hashes: list[str]) -> None:
        existing = self._sessions.get(session_id)
        if existing and len(curr_hashes) > existing.hash_chain_length:
            existing.max_hash_chain = ",".join(curr_hashes)
            existing.hash_chain_length = len(curr_hashes)

    # ── Canonical (SysToolsLock) — session-level ────────────────────

    def get_canonical(self, session_id: str) -> dict[str, str]:
        rec = self._sessions.get(session_id)
        if rec and (rec.canonical_sys or rec.canonical_tools):
            return {"sys": rec.canonical_sys, "tools": rec.canonical_tools}
        return {"sys": "", "tools": ""}

    def set_canonical(self, session_id: str, sys: str = "", tools: str = "") -> None:
        rec = self._sessions.get(session_id)
        if rec:
            if sys:
                rec.canonical_sys = sys
            if tools:
                rec.canonical_tools = tools

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
        """Get ALL prefix chains for a session (flattened, backward compat)."""
        family_map = self._chains.get(session_id, {})
        chains: list[dict] = []
        for clist in family_map.values():
            chains.extend(c.as_dict() for c in clist)
        # 保持按 last_seen desc 排序
        chains.sort(key=lambda c: c.get("last_seen", ""), reverse=True)
        return chains

    def get_family_chains(self, session_id: str, family_key: str) -> list[dict]:
        """Get prefix chains for a specific branch family within a session."""
        clist = self._chains.get(session_id, {}).get(family_key, [])
        return [c.as_dict() for c in clist]

    def upsert_chain(self, session_id: str, chain_id: str, *,
                     sys: str = "[]", tools: str = "[]", messages: str = "[]",
                     sys_hashes: str = "", tools_hashes: str = "",
                     messages_hashes: str = "",
                     msg_count: int = 0,
                     safe_breakpoints: list[int] | None = None,
                     is_frozen: bool = False,
                     family_key: str = "") -> ChainRecord:
        """Insert or update a prefix chain.

        When ``family_key`` is empty it is computed from ``sys_hashes``
        and ``tools_hashes`` (backward compat — all callers should
        eventually pass ``ctx.family_key``).
        """
        now = datetime.now(timezone.utc).isoformat()
        sb_json = json.dumps(safe_breakpoints) if safe_breakpoints is not None else None

        if not family_key:
            family_key = self.compute_family_key(sys_hashes, tools_hashes)

        # Find existing chain (across all sessions since chain_id is unique)
        for fmap in self._chains.values():
            for clist in fmap.values():
                for c in clist:
                    if c.chain_id == chain_id:
                        # Update only if new msg_count > old and not frozen
                        if msg_count > c.msg_count and not c.is_frozen:
                            c.sys = sys
                            c.tools = tools
                            c.messages = messages
                            c.sys_hashes = sys_hashes
                            c.tools_hashes = tools_hashes
                            c.messages_hashes = messages_hashes
                            c.msg_count = msg_count
                            c.safe_breakpoints = sb_json
                            c.last_seen = now
                        return c

        # Create new
        rec = ChainRecord(
            chain_id=chain_id, session_id=session_id,
            sys=sys, tools=tools, messages=messages,
            sys_hashes=sys_hashes, tools_hashes=tools_hashes,
            messages_hashes=messages_hashes,
            msg_count=msg_count, safe_breakpoints=sb_json,
            is_frozen=is_frozen, created_at=now, last_seen=now,
        )
        fmap = self._chains.setdefault(session_id, {})
        fmap.setdefault(family_key, []).append(rec)
        # Keep sorted by last_seen desc
        fmap[family_key].sort(key=lambda c: c.last_seen, reverse=True)
        return rec

    # ── Chain token history ──────────────────────────────────────

    def insert_chain_token(self, chain_id: str, msg_count: int,
                           tokens: int, char_count: int = 0) -> None:
        now = datetime.now(timezone.utc).isoformat()
        rec = ChainTokenRecord(
            chain_id=chain_id, msg_count=msg_count,
            tokens=tokens, char_count=char_count, created_at=now,
        )
        history = self._token_history.setdefault(chain_id, [])
        # Check for duplicate (chain_id, msg_count) — insert or ignore
        for i, h in enumerate(history):
            if h.msg_count == msg_count:
                return  # already exists
        history.append(rec)
        history.sort(key=lambda h: h.msg_count)

    def get_chain_tokens_and_chars(self, chain_id: str,
                                   msg_count: int) -> tuple[int, int] | None:
        """Get (tokens, char_count) for closest version ≤ msg_count."""
        history = self._token_history.get(chain_id, [])
        best = None
        for h in history:
            if h.msg_count <= msg_count:
                best = h
            else:
                break
        return (best.tokens, best.char_count) if best else None

    def get_chain_tokens_ge(self, chain_id: str,
                            msg_count: int) -> tuple[int, int, int] | None:
        """Get (tokens, char_count, msg_count) for closest version ≥ msg_count."""
        history = self._token_history.get(chain_id, [])
        for h in history:
            if h.msg_count >= msg_count:
                return (h.tokens, h.char_count, h.msg_count)
        return None

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
        _q = _sync_queue.Queue()
        _done = asyncio.Event()

        def _thread_worker():
            conn = db.conn  # thread-local: creates connection in THIS thread
            while True:
                item = _q.get()
                if item is None:  # sentinel
                    break
                sync_fn, args = item
                try:
                    sync_fn(conn, *args)
                except Exception:
                    log.exception("MemoryStore: SQLite write failed")
            _q.task_done()

        import threading
        t = threading.Thread(target=_thread_worker, daemon=True)
        t.start()

        try:
            while True:
                item = await self._write_queue.get()
                if item is None:  # sentinel
                    break
                _q.put(item)
        finally:
            _q.put(None)  # stop thread worker
            t.join(timeout=5)

    def enqueue_sqlite_write(self, sync_fn, *args) -> None:
        """Fire-and-forget: queue a SQLite write for the background writer.

        The sync_fn receives the SQLite connection as first argument.
        MemoryStore is already updated (0 I/O) — this is durability only.
        """
        if self._write_queue is None:
            return
        self._write_queue.put_nowait((sync_fn, args))

    async def flush(self) -> None:
        """Drain all pending SQLite writes without stopping the writer.

        Puts a sentinel in the queue and polls until all items enqueued
        before this call have been processed by the writer thread.
        Call before opening a new SQLite connection to read committed
        data in tests (WAL snapshot isolation).
        """
        if self._write_queue is None:
            return
        # Drain the async queue first (short poll)
        for _ in range(100):
            if self._write_queue.empty():
                break
            await asyncio.sleep(0.001)
        # Give the sync queue + writer thread time to commit
        await asyncio.sleep(0.05)

    async def flush_pending(self) -> None:
        """Await all pending SQLite writes + stop writer (called during shutdown)."""
        if self._write_queue is None:
            return
        # Drain queue
        while not self._write_queue.empty():
            await asyncio.sleep(0.01)
        # Stop writer
        if self._writer_task:
            self._write_queue.put_nowait(None)  # sentinel
            try:
                await asyncio.wait_for(self._writer_task, timeout=10)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._writer_task.cancel()
            self._writer_task = None

    # ── Lifecycle ───────────────────────────────────────────────

    @classmethod
    async def load_from(cls, db, _db=None) -> MemoryStore:
        """Populate MemoryStore from existing SQLite data at startup."""
        store = cls(db=_db or db)

        def _load():
            conn = db.conn if hasattr(db, 'conn') else db
            now = datetime.now(timezone.utc).isoformat()

            # Clean up orphaned pending records from previous crash
            try:
                deleted = conn.execute(
                    "DELETE FROM requests WHERE record_state = 'pending'"
                ).rowcount
                if deleted:
                    conn.commit()
                    log.info("MemoryStore: cleaned up %d orphaned pending record(s)", deleted)
            except sqlite3.OperationalError:
                pass

            # Load sessions
            try:
                rows = conn.execute("SELECT * FROM sessions").fetchall()
                for row in rows:
                    d = dict(row)
                    store._sessions[d["session_id"]] = SessionRecord(
                        session_id=d.get("session_id", ""),
                        client_type=d.get("client_type", ""),
                        anchor_msg_index=d.get("anchor_msg_index", 0),
                        max_hash_chain=d.get("max_hash_chain", ""),
                        hash_chain_length=d.get("hash_chain_length", 0),
                        start_time=d.get("start_time", now),
                        last_seen_time=d.get("last_seen_time", now),
                        total_requests=d.get("total_requests", 1),
                        canonical_sys=d.get("canonical_sys", ""),
                        canonical_tools=d.get("canonical_tools", ""),
                    )
            except sqlite3.OperationalError:
                pass

            # Load prefix chains
            try:
                rows = conn.execute("SELECT * FROM prefix_chains ORDER BY last_seen DESC").fetchall()
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
                        created_at=d.get("created_at", now),
                        last_seen=d.get("last_seen", now),
                    )
                    store._chains.setdefault(rec.session_id, {}) \
                        .setdefault(store.compute_family_key(rec.sys_hashes, rec.tools_hashes), []) \
                        .append(rec)
            except sqlite3.OperationalError:
                pass

            # Load chain token history
            try:
                rows = conn.execute(
                    "SELECT * FROM chain_token_history ORDER BY chain_id, msg_count ASC"
                ).fetchall()
                for row in rows:
                    d = dict(row)
                    rec = ChainTokenRecord(
                        chain_id=d.get("chain_id", ""),
                        msg_count=d.get("msg_count", 0),
                        tokens=d.get("tokens", 0),
                        char_count=d.get("char_count", 0),
                        created_at=d.get("created_at", now),
                    )
                    store._token_history.setdefault(rec.chain_id, []).append(rec)
            except sqlite3.OperationalError:
                pass

            log.info(
                "MemoryStore loaded from DB: %d sessions, %d chains, %d token records",
                len(store._sessions), sum(len(c) for c in store._chains.values()),
                sum(len(h) for h in store._token_history.values()),
            )

        await asyncio.to_thread(_load)
        return store

    # ── Requests 表持久化（统一走单写者队列）─────────────────────

    # 哨兵：区分「未提供快照」和「快照值为 None」
    _NO_SNAPSHOT = object()

    def enqueue_request_phase1(self, ctx) -> None:
        """Fire-and-forget: insert request record (Phase 1, state='pending')."""
        from deepseek_tap_proxy.shared.db import insert_request_phase1

        # snapshot ctx values for the write lambda
        _api_format = ctx.persist.api_format
        _trace_category = ctx.persist.trace_category
        _msg_hash_chain = ctx.msg_hash_chain

        def _write(conn, _ctx, _af, _tc, _mhc):
            if _af:
                _ctx.persist.api_format = _af
            if _tc:
                _ctx.persist.trace_category = _tc
            _ctx.persist.msg_hash_chain = _mhc
            insert_request_phase1(conn, _ctx)

        self.enqueue_sqlite_write(_write, ctx, _api_format, _trace_category, _msg_hash_chain)

    def enqueue_request_phase2(self, ctx, *,
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
                    mod_note_snapshot=_mn)

        self.enqueue_sqlite_write(
            _write, ctx, jsonl_path,
            _lt, is_modified_snapshot, mod_note_snapshot,
        )
