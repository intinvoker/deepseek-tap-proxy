"""Cross-format database functions shared across providers.

Pure functions that take ``conn`` (sqlite3.Connection) as first parameter.
These operate on the ``sessions``, ``prefix_chains``, and unified ``requests``
tables which are format-agnostic.

All insert/update functions are ctx-based — they read from ``Context.persist``
and properties, never from raw format-specific structures.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone


# ── Sessions table helpers ──


def get_session_by_id(conn: sqlite3.Connection, session_id: str) -> dict | None:
    """Get session by ID. Returns dict or None."""
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def get_or_create_session_by_explicit_id(
    conn: sqlite3.Connection,
    session_id: str,
    client_type: str,
    anchor_msg_index: int,
    curr_hashes: list[str],
    now: str | None = None,
) -> dict:
    """Get existing session or create new. Always updates last_seen_time and max_hash_chain."""
    if now is None:
        now = datetime.now(timezone.utc).isoformat()

    existing = get_session_by_id(conn, session_id)
    if existing:
        conn.execute("""
            UPDATE sessions SET
                last_seen_time = ?,
                total_requests = total_requests + 1
            WHERE session_id = ?
        """, (now, session_id))
        if len(curr_hashes) > existing["hash_chain_length"]:
            hash_chain_str = ",".join(curr_hashes)
            conn.execute("""
                UPDATE sessions SET max_hash_chain = ?, hash_chain_length = ?
                WHERE session_id = ?
            """, (hash_chain_str, len(curr_hashes), session_id))
        conn.commit()
        return dict(conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone())
    else:
        hash_chain_str = ",".join(curr_hashes)
        conn.execute("""
            INSERT INTO sessions (session_id, client_type, anchor_msg_index,
                max_hash_chain, hash_chain_length, start_time, last_seen_time, total_requests)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1)
        """, (session_id, client_type, anchor_msg_index,
              hash_chain_str, len(curr_hashes), now, now))
        conn.commit()
        return dict(conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone())


def update_session_hash_chain(
    conn: sqlite3.Connection,
    session_id: str,
    curr_hashes: list[str],
) -> None:
    """Update max_hash_chain only if curr_hashes is longer than current."""
    existing = get_session_by_id(conn, session_id)
    if existing and len(curr_hashes) > existing["hash_chain_length"]:
        hash_chain_str = ",".join(curr_hashes)
        conn.execute("""
            UPDATE sessions SET max_hash_chain = ?, hash_chain_length = ?
            WHERE session_id = ?
        """, (hash_chain_str, len(curr_hashes), session_id))
        conn.commit()


def find_candidate_sessions(conn: sqlite3.Connection, client_type: str) -> list[dict]:
    """Find active sessions matching client_type, ordered by last_seen desc."""
    rows = conn.execute("""
        SELECT * FROM sessions
        WHERE client_type = ? AND hash_chain_length > 0
        ORDER BY last_seen_time DESC LIMIT 50
    """, (client_type,)).fetchall()
    return [dict(r) for r in rows]


# ── Prefix chains table helpers ──


def upsert_prefix_chain(conn: sqlite3.Connection, session_id: str, chain_id: str, *,
                        sys: str, tools: str, messages: str,
                        sys_hashes: str, tools_hashes: str, messages_hashes: str,
                        msg_count: int,
                        safe_breakpoints: list[int] | None = None,
                        total_input_tokens: int = 0,
                        is_frozen: bool = False) -> None:
    """Insert or update a prefix chain. Only updates if msg_count is longer.

    ``sys``, ``tools``, ``messages`` are JSON strings — canonical form triples.
    ``sys_hashes``, ``tools_hashes``, ``messages_hashes`` are comma-separated hash strings.
    ``total_input_tokens`` is the upstream's total input token count for this request,
    used for estimating KV cache hit rates on subsequent requests.
    ``is_frozen`` marks a public prefix chain — never updated once created.
    """
    now = datetime.now(timezone.utc).isoformat()
    safe_breakpoints_json = json.dumps(safe_breakpoints) if safe_breakpoints is not None else None
    conn.execute("""
        INSERT INTO prefix_chains
            (session_id, chain_id, sys, tools, messages,
             sys_hashes, tools_hashes, messages_hashes,
             msg_count,
             safe_breakpoints, total_input_tokens, is_frozen, created_at, last_seen)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(chain_id) DO UPDATE SET
            sys = CASE WHEN ? > prefix_chains.msg_count AND prefix_chains.is_frozen = 0
                       THEN ? ELSE prefix_chains.sys END,
            tools = CASE WHEN ? > prefix_chains.msg_count AND prefix_chains.is_frozen = 0
                         THEN ? ELSE prefix_chains.tools END,
            messages = CASE WHEN ? > prefix_chains.msg_count AND prefix_chains.is_frozen = 0
                            THEN ? ELSE prefix_chains.messages END,
            sys_hashes = CASE WHEN ? > prefix_chains.msg_count AND prefix_chains.is_frozen = 0
                              THEN ? ELSE prefix_chains.sys_hashes END,
            tools_hashes = CASE WHEN ? > prefix_chains.msg_count AND prefix_chains.is_frozen = 0
                                THEN ? ELSE prefix_chains.tools_hashes END,
            messages_hashes = CASE WHEN ? > prefix_chains.msg_count AND prefix_chains.is_frozen = 0
                                   THEN ? ELSE prefix_chains.messages_hashes END,
            msg_count = CASE WHEN ? > prefix_chains.msg_count AND prefix_chains.is_frozen = 0
                             THEN ? ELSE prefix_chains.msg_count END,
            safe_breakpoints = CASE WHEN ? > prefix_chains.msg_count AND prefix_chains.is_frozen = 0
                                   THEN ? ELSE prefix_chains.safe_breakpoints END,
            total_input_tokens = CASE WHEN ? > prefix_chains.msg_count THEN ? ELSE prefix_chains.total_input_tokens END,
            last_seen = ?
    """, (session_id, chain_id, sys, tools, messages,
          sys_hashes, tools_hashes, messages_hashes,
          msg_count,
          safe_breakpoints_json, total_input_tokens, is_frozen, now, now,
          msg_count, sys, msg_count, tools, msg_count, messages,
          msg_count, sys_hashes, msg_count, tools_hashes, msg_count, messages_hashes,
          msg_count, msg_count, msg_count, safe_breakpoints_json,
          msg_count, total_input_tokens, now))
    conn.commit()


# ── Sessions table ──


def create_sessions_table(conn: sqlite3.Connection) -> None:
    """Create the sessions table for session-level aggregation."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id         TEXT PRIMARY KEY,
            client_type        TEXT NOT NULL,
            anchor_msg_index   INTEGER DEFAULT 0,
            max_hash_chain     TEXT DEFAULT '',
            hash_chain_length  INTEGER DEFAULT 0,
            start_time         TEXT NOT NULL,
            last_seen_time     TEXT NOT NULL,
            total_requests     INTEGER DEFAULT 1,
            canonical_sys      TEXT DEFAULT '',
            canonical_tools    TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_client_type ON sessions(client_type);
        CREATE INDEX IF NOT EXISTS idx_sessions_last_seen ON sessions(last_seen_time);
    """)
    conn.commit()
    # Migration: add columns that may not exist in older schemas
    for col in ("anchor_msg_index", "max_hash_chain", "hash_chain_length",
                "canonical_sys", "canonical_tools"):
        try:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
    conn.commit()


def get_session_canonical(conn: sqlite3.Connection,
                          session_id: str) -> dict[str, str]:
    """Get canonical sys/tools for a session."""
    row = conn.execute(
        "SELECT canonical_sys, canonical_tools FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row:
        return {"sys": row[0] or "", "tools": row[1] or ""}
    return {"sys": "", "tools": ""}


def set_session_canonical(conn: sqlite3.Connection, session_id: str,
                          sys: str = "", tools: str = "") -> None:
    """Set canonical sys/tools for a session (UPDATE only non-empty values)."""
    sets = []
    params = []
    if sys:
        sets.append("canonical_sys = ?")
        params.append(sys)
    if tools:
        sets.append("canonical_tools = ?")
        params.append(tools)
    if not sets:
        return
    params.append(session_id)
    conn.execute(
        f"UPDATE sessions SET {', '.join(sets)} WHERE session_id = ?",
        params,
    )
    conn.commit()


# ── Prefix chains table ──


def create_prefix_chains_table(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS prefix_chains (
            session_id  TEXT NOT NULL,
            chain_id    TEXT NOT NULL PRIMARY KEY,
            -- Note: `id` column removed from new DBs (was INTEGER PRIMARY KEY AUTOINCREMENT).
            -- Existing DBs retain the column; it's harmless and no migration is needed.
            sys         TEXT NOT NULL DEFAULT '[]',
            tools       TEXT NOT NULL DEFAULT '[]',
            messages    TEXT NOT NULL DEFAULT '[]',
            sys_hashes   TEXT NOT NULL DEFAULT '',
            tools_hashes TEXT NOT NULL DEFAULT '',
            messages_hashes TEXT NOT NULL DEFAULT '',
            msg_count   INTEGER NOT NULL DEFAULT 0,
            safe_breakpoints TEXT,
            total_input_tokens INTEGER DEFAULT 0,
            is_frozen   INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT NOT NULL,
            last_seen   TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
        CREATE INDEX IF NOT EXISTS idx_pc_session ON prefix_chains(session_id);
    """)
    # Migrations: add columns that may not exist in older schemas
    for col in ("safe_breakpoints", "sys", "tools", "is_frozen"):
        col_type = "INTEGER" if col == "is_frozen" else "TEXT"
        try:
            conn.execute(f"ALTER TABLE prefix_chains ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass
    # Migration: three-column hash (2026-06-10)
    for col in ("sys_hashes", "tools_hashes", "messages_hashes"):
        try:
            conn.execute(f"ALTER TABLE prefix_chains ADD COLUMN {col} TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass
    conn.commit()


def get_session_chains(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    """Get all prefix chains for a session, ordered by last_seen desc.

    Returns rows with canonical three-column format:
    ``sys``, ``tools``, ``messages`` — all JSON strings.
    """
    rows = conn.execute("""
        SELECT * FROM prefix_chains WHERE session_id = ? ORDER BY last_seen DESC
    """, (session_id,)).fetchall()
    return [dict(r) for r in rows]


# ── Chain token history table ──


def create_chain_token_history_table(conn: sqlite3.Connection) -> None:
    """Create the chain_token_history table for precise token estimation."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chain_token_history (
            chain_id    TEXT NOT NULL REFERENCES prefix_chains(chain_id),
            msg_count   INTEGER NOT NULL,
            tokens      INTEGER NOT NULL,
            char_count  INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT NOT NULL,
            PRIMARY KEY (chain_id, msg_count)
        );
    """)
    conn.commit()
    # Migration: drop redundant index (PK already covers chain_id)
    try:
        conn.execute("DROP INDEX IF EXISTS idx_cth_chain")
    except sqlite3.OperationalError:
        pass
    # Migration: add char_count column if missing
    try:
        conn.execute("ALTER TABLE chain_token_history ADD COLUMN char_count INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()


def insert_chain_token(conn: sqlite3.Connection, chain_id: str, *,
                       msg_count: int, tokens: int,
                       char_count: int = 0) -> None:
    """Record a token snapshot for a chain version.

    Called after each chain CREATE or UPDATE to record the exact
    ``total_input_tokens`` from the upstream response.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT OR IGNORE INTO chain_token_history (chain_id, msg_count, tokens, char_count, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (chain_id, msg_count, tokens, char_count, now))
    conn.commit()


def get_chain_tokens(conn: sqlite3.Connection, chain_id: str,
                     msg_count: int) -> int | None:
    """Get the exact total_input_tokens for the closest version ≤ msg_count.

    Returns ``None`` if no record exists or the table doesn't exist yet.
    """
    try:
        row = conn.execute("""
            SELECT tokens FROM chain_token_history
            WHERE chain_id = ? AND msg_count <= ?
            ORDER BY msg_count DESC LIMIT 1
        """, (chain_id, msg_count)).fetchone()
        return row["tokens"] if row else None
    except sqlite3.OperationalError:
        return None


def get_chain_tokens_and_chars(conn: sqlite3.Connection, chain_id: str,
                               msg_count: int) -> tuple[int, int] | None:
    """Get (tokens, char_count) for the closest version ≤ msg_count.

    Returns ``None`` if no record exists or the table doesn't exist yet.
    """
    try:
        row = conn.execute("""
            SELECT tokens, char_count FROM chain_token_history
            WHERE chain_id = ? AND msg_count <= ?
            ORDER BY msg_count DESC LIMIT 1
        """, (chain_id, msg_count)).fetchone()
        return (row["tokens"], row["char_count"]) if row else None
    except sqlite3.OperationalError:
        return None


def get_chain_tokens_ge(conn: sqlite3.Connection, chain_id: str,
                        msg_count: int) -> tuple[int, int, int] | None:
    """Get (tokens, char_count, msg_count) for the closest version ≥ msg_count.

    Fallback when ``get_chain_tokens_and_chars`` returns None：
    pref_hit 小于链的首个 token_history 记录 → 向更大的方向查找。
    """
    try:
        row = conn.execute("""
            SELECT tokens, char_count, msg_count FROM chain_token_history
            WHERE chain_id = ? AND msg_count >= ?
            ORDER BY msg_count ASC LIMIT 1
        """, (chain_id, msg_count)).fetchone()
        return (row["tokens"], row["char_count"], row["msg_count"]) if row else None
    except sqlite3.OperationalError:
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Unified requests table
#
#  Single ``requests`` table shared across all API formats.
#  ``api_format`` column distinguishes the source format;
#  format-specific columns are nullable.
# ═══════════════════════════════════════════════════════════════════════


def create_requests_table(conn: sqlite3.Connection) -> None:
    """Create the unified requests table with columns for all API formats.

    Replaces the former three-way split (shared/openai/anthropic).
    ``api_format`` ('openai'|'anthropic') distinguishes the source;
    ``trace_category`` and ``system_role_converted`` are Anthropic-specific
    but stored as nullable columns on the shared table.
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT UNIQUE NOT NULL,
            timestamp TEXT NOT NULL,
            duration_ms INTEGER NOT NULL,
            model TEXT NOT NULL,
            stream_mode INTEGER NOT NULL DEFAULT 1,
            status_code INTEGER NOT NULL,
            total_input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            cache_hit_tokens INTEGER DEFAULT 0,
            cache_miss_tokens INTEGER DEFAULT 0,
            cache_hit_ratio REAL DEFAULT 0.0,
            cache_creation_tokens INTEGER DEFAULT 0,
            has_reasoning INTEGER DEFAULT 0,
            has_tool_calls INTEGER DEFAULT 0,
            prompt_messages_count INTEGER DEFAULT 0,
            prompt_tools_count INTEGER DEFAULT 0,
            session_id            TEXT DEFAULT NULL,
            session_uuid          TEXT DEFAULT NULL,
            session_fingerprint   TEXT DEFAULT NULL,
            session_type          TEXT DEFAULT NULL,
            api_format            TEXT DEFAULT NULL,
            jsonl_path            TEXT NOT NULL DEFAULT '',
            is_modified           INTEGER DEFAULT 0,
            modification_note     TEXT DEFAULT '',
            msg_hash_chain        TEXT DEFAULT NULL,
            record_state          TEXT DEFAULT 'pending',
            lock_type             TEXT DEFAULT NULL,
            jsonl_offset          INTEGER DEFAULT 0,
            prefix_chain_id       TEXT DEFAULT NULL,
            trace_category        TEXT DEFAULT NULL,
            system_role_converted INTEGER DEFAULT 0,
            sys_tools_locked      TEXT DEFAULT NULL,
            chat_type             TEXT DEFAULT 'normal',
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE SET NULL,
            FOREIGN KEY (prefix_chain_id) REFERENCES prefix_chains(chain_id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_requests_timestamp   ON requests(timestamp);
        CREATE INDEX IF NOT EXISTS idx_requests_model       ON requests(model);
        CREATE INDEX IF NOT EXISTS idx_requests_cache_ratio ON requests(cache_hit_ratio);
        CREATE INDEX IF NOT EXISTS idx_requests_session_fp  ON requests(session_fingerprint);
        CREATE INDEX IF NOT EXISTS idx_requests_session_uuid ON requests(session_uuid);
        CREATE INDEX IF NOT EXISTS idx_requests_session_id  ON requests(session_id);
        CREATE INDEX IF NOT EXISTS idx_requests_api_format  ON requests(api_format);
    """)
    conn.commit()
    # Migration: drop columns that may exist in older schemas
    for col in ("turn", "msg0_hash", "msg0_content"):
        try:
            conn.execute(f"ALTER TABLE requests DROP COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    # Migration: add chat_type column (2026-06-09)
    try:
        conn.execute("ALTER TABLE requests ADD COLUMN chat_type TEXT DEFAULT 'normal'")
    except sqlite3.OperationalError:
        pass
    # Migration: add sys_tools_locked column (2026-06-12, ADR #17)
    try:
        conn.execute("ALTER TABLE requests ADD COLUMN sys_tools_locked TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass
    conn.commit()


def insert_request_phase1(conn: sqlite3.Connection, ctx, *,
                          jsonl_offset: int = 0) -> None:
    """Unified Phase 1 INSERT: request-side data, ``record_state='pending'``.

    Reads from ``ctx.persist`` (synced by RequestAnalysisStage and CanonicalStage).
    """
    model = ctx.body_readonly.get("model", "") if ctx.body_readonly else ""
    timestamp = datetime.now(timezone.utc).isoformat()

    conn.execute("""
        INSERT OR IGNORE INTO requests
            (request_id, timestamp, duration_ms, status_code, model, stream_mode,
             prompt_messages_count, prompt_tools_count,
             is_modified, modification_note,
             session_id, session_fingerprint, session_type,
             msg_hash_chain,
             jsonl_path, jsonl_offset, record_state,
             api_format, trace_category, chat_type)
        VALUES (?, ?, 0, 0, ?, ?, ?, ?, 0, '', ?, ?,
                ?, ?, ?, ?, 'pending', ?, ?, ?)
    """, (
        ctx.req_id,
        timestamp,
        model,
        1 if ctx.is_streaming else 0,
        ctx.persist.prompt_messages_count,
        ctx.persist.prompt_tools_count,
        ctx.persist.session_id or None,
        "",                                       # session_fingerprint — deprecated
        ctx.persist.client_type or None,          # session_type → client_type
        ctx.persist.msg_hash_chain,
        ctx.meta.get("jsonl_path", ""),
        jsonl_offset,
        ctx.persist.api_format,
        ctx.persist.trace_category,
        ctx.persist.chat_type or "normal",
    ))
    conn.commit()


# ── 快照标识：区分「未提供快照」和「快照值为 None」──
_SENTINEL_NO_SNAPSHOT = object()


def update_request_phase2(conn: sqlite3.Connection, ctx, *,
                          jsonl_path: str = '',
                          lock_type_snapshot: str | None | object = _SENTINEL_NO_SNAPSHOT,
                          is_modified_snapshot: bool = False,
                          mod_note_snapshot: str = "") -> None:
    """Unified Phase 2 UPDATE: fill response data, ``record_state='completed'``.

    Reads response data from ``ctx.persist`` (populated by
    ``apply_cache_analysis`` and the ForwardStage), and ``ctx.property`` view
    fields.

    ``lock_type_snapshot`` / ``is_modified_snapshot`` / ``mod_note_snapshot``
    override ``ctx.persist`` values — callers should pass the values captured
    immediately after pipeline execution to avoid async closure races.
    """
    prompt_tokens = ctx.persist.total_input_tokens
    completion_tokens = ctx.persist.output_tokens
    if lock_type_snapshot is _SENTINEL_NO_SNAPSHOT:
        _lock_type = ctx.persist.lock_type or None
        _is_modified = int(ctx.persist.is_modified)
        _mod_note = ctx.persist.modification_note
    else:
        _lock_type = lock_type_snapshot  # 可能为 None（无锁）
        _is_modified = int(is_modified_snapshot)
        _mod_note = mod_note_snapshot
    conn.execute("""
        UPDATE requests SET
            duration_ms = ?, status_code = ?,
            total_input_tokens = ?, output_tokens = ?,
            total_tokens = ?,
            cache_hit_tokens = ?, cache_miss_tokens = ?,
            cache_hit_ratio = ?,
            cache_creation_tokens = ?,
            has_reasoning = ?, has_tool_calls = ?,
            jsonl_path = ?,
            jsonl_offset = ?,
            lock_type = ?,
            prefix_chain_id = ?,
            is_modified = ?,
            modification_note = ?,
            system_role_converted = ?,
            sys_tools_locked = ?,
            record_state = 'completed'
        WHERE request_id = ?
    """, (
        ctx.duration_ms or ctx.elapsed_ms,
        ctx.status,
        prompt_tokens,
        completion_tokens,
        prompt_tokens + completion_tokens,
        ctx.persist.cache_hit_tokens,
        ctx.persist.cache_miss_tokens,
        ctx.persist.cache_hit_ratio,
        ctx.persist.cache_creation_tokens,
        int(ctx.persist.has_reasoning),
        int(ctx.persist.has_tool_calls),
        jsonl_path,
        ctx.meta.get("jsonl_offset", 0),
        _lock_type,
        ctx.persist.prefix_chain_id or None,
        _is_modified,
        _mod_note,
        int(ctx.persist.system_role_converted),
        ctx.persist.sys_tools_locked or None,
        ctx.req_id,
    ))
    conn.commit()


# ── Query functions (operate on the unified requests table) ─────────


def get_request_jsonl_info(conn: sqlite3.Connection, request_id: str) -> dict | None:
    """Get (jsonl_path, jsonl_offset) for a request."""
    row = conn.execute("""
        SELECT jsonl_path, jsonl_offset FROM requests
        WHERE request_id = ?
    """, (request_id,)).fetchone()
    return dict(row) if row else None


def get_previous_by_session_id(
    conn, session_id: str, exclude_req_id: str,
) -> tuple[str | None, int] | None:
    """Get (msg_hash_chain, prompt_messages_count) by session_id."""
    row = conn.execute("""
        SELECT msg_hash_chain, prompt_messages_count FROM requests
        WHERE session_id = ? AND request_id != ?
          AND msg_hash_chain IS NOT NULL
        ORDER BY id DESC LIMIT 1
    """, (session_id, exclude_req_id)).fetchone()
    if row and row["msg_hash_chain"]:
        return (row["msg_hash_chain"], row["prompt_messages_count"])
    return None


def get_previous_by_fingerprint(
    conn, fingerprint: str, exclude_req_id: str,
) -> tuple[str | None, int] | None:
    """Get (msg_hash_chain, prompt_messages_count) by session_fingerprint."""
    row = conn.execute("""
        SELECT msg_hash_chain, prompt_messages_count FROM requests
        WHERE session_fingerprint = ? AND request_id != ?
          AND msg_hash_chain IS NOT NULL
        ORDER BY id DESC LIMIT 1
    """, (fingerprint, exclude_req_id)).fetchone()
    if row and row["msg_hash_chain"]:
        return (row["msg_hash_chain"], row["prompt_messages_count"])
    return None


def get_previous_jsonl_by_fingerprint(
    conn, fingerprint: str, exclude_req_id: str,
) -> tuple[str, str, int] | None:
    """Get (jsonl_path, request_id, jsonl_offset) by session_fingerprint."""
    row = conn.execute("""
        SELECT jsonl_path, request_id, jsonl_offset FROM requests
        WHERE session_fingerprint = ? AND request_id != ? AND jsonl_path != ''
        ORDER BY id DESC LIMIT 1
    """, (fingerprint, exclude_req_id)).fetchone()
    return (row["jsonl_path"], row["request_id"], row["jsonl_offset"] or 0) if row else None


def get_previous_jsonl_by_session_id(
    conn, session_id: str, exclude_req_id: str,
) -> tuple[str, str, int] | None:
    """Get (jsonl_path, request_id, jsonl_offset) by session_id."""
    row = conn.execute("""
        SELECT jsonl_path, request_id, jsonl_offset FROM requests
        WHERE session_id = ? AND request_id != ? AND jsonl_path != ''
        ORDER BY id DESC LIMIT 1
    """, (session_id, exclude_req_id)).fetchone()
    return (row["jsonl_path"], row["request_id"], row["jsonl_offset"] or 0) if row else None


# ═══════════════════════════════════════════════════════════════════════
#  Migrations
# ═══════════════════════════════════════════════════════════════════════


def migrate_requests_rename(conn: sqlite3.Connection) -> None:
    """Rename prompt_tokens->total_input_tokens, completion_tokens->output_tokens."""
    cursor = conn.execute("PRAGMA table_info(requests)")
    cols = {row[1] for row in cursor.fetchall()}

    # prompt_tokens -> total_input_tokens
    if "prompt_tokens" in cols and "total_input_tokens" not in cols:
        conn.execute("ALTER TABLE requests ADD COLUMN total_input_tokens INTEGER DEFAULT 0")
        conn.execute("UPDATE requests SET total_input_tokens = prompt_tokens WHERE prompt_tokens IS NOT NULL")
        try:
            conn.execute("ALTER TABLE requests DROP COLUMN prompt_tokens")
        except sqlite3.OperationalError:
            pass  # SQLite version may not support DROP COLUMN

    # completion_tokens -> output_tokens
    if "completion_tokens" in cols and "output_tokens" not in cols:
        conn.execute("ALTER TABLE requests ADD COLUMN output_tokens INTEGER DEFAULT 0")
        conn.execute("UPDATE requests SET output_tokens = completion_tokens WHERE completion_tokens IS NOT NULL")
        try:
            conn.execute("ALTER TABLE requests DROP COLUMN completion_tokens")
        except sqlite3.OperationalError:
            pass

    conn.commit()


def drop_deprecated_tables(conn: sqlite3.Connection) -> None:
    """Drop tables that no longer exist in the schema."""
    conn.executescript("""
        PRAGMA foreign_keys=OFF;
        DROP TABLE IF EXISTS request_audit;
        DROP TABLE IF EXISTS sys_tools_baseline;
        PRAGMA foreign_keys=ON;
    """)
    conn.commit()


def cleanup_prefix_chains_id(conn: sqlite3.Connection) -> None:
    """Drop legacy id column from prefix_chains if it exists."""
    try:
        conn.execute("ALTER TABLE prefix_chains DROP COLUMN id")
    except sqlite3.OperationalError:
        pass
    conn.commit()


