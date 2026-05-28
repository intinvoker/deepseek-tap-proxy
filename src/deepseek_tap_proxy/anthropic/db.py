from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

def create_anthropic_tables(conn: sqlite3.Connection) -> None:
    """Create Anthropic trace tables."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT UNIQUE NOT NULL,
            timestamp TEXT NOT NULL,
            duration_ms INTEGER NOT NULL,
            model TEXT NOT NULL,
            stream_mode INTEGER NOT NULL DEFAULT 1,
            status_code INTEGER NOT NULL,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            cache_hit_tokens INTEGER DEFAULT 0,
            cache_miss_tokens INTEGER DEFAULT 0,
            cache_hit_ratio REAL DEFAULT 0.0,
            cache_creation_tokens INTEGER DEFAULT 0,
            session_uuid TEXT DEFAULT NULL,
            session_type TEXT DEFAULT NULL,
            jsonl_path TEXT NOT NULL DEFAULT '',
            api_format TEXT DEFAULT 'anthropic',
            turn INTEGER DEFAULT 0,
            record_state TEXT DEFAULT 'completed',
            is_modified INTEGER DEFAULT 0,
            lock_type TEXT DEFAULT NULL,
            modification_note TEXT DEFAULT NULL,
            trace_category TEXT DEFAULT NULL,
            jsonl_offset INTEGER DEFAULT 0,
            system_role_converted INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_requests_format ON requests(api_format);
        CREATE INDEX IF NOT EXISTS idx_requests_anthropic_session ON requests(session_uuid);

        CREATE TABLE IF NOT EXISTS sys_tools_baseline (
            session_id           TEXT NOT NULL PRIMARY KEY,
            system_json          TEXT NOT NULL,
            tools_json           TEXT NOT NULL,
            baseline_request_id  TEXT NOT NULL,
            system_hash          TEXT NOT NULL,
            tools_hash           TEXT NOT NULL,
            user_id              TEXT DEFAULT NULL,
            created_at           TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_baseline_hashes
            ON sys_tools_baseline(system_hash, tools_hash);

        CREATE TABLE IF NOT EXISTS request_audit (
            request_id             TEXT NOT NULL PRIMARY KEY,
            session_id             TEXT NOT NULL,
            system_hash            TEXT NOT NULL,
            tools_hash             TEXT NOT NULL,
            is_baseline            INTEGER NOT NULL DEFAULT 0,
            sys_changed            INTEGER NOT NULL DEFAULT 0,
            tools_changed          INTEGER NOT NULL DEFAULT 0,
            was_replaced           INTEGER NOT NULL DEFAULT 0,
            user_id_was_overridden INTEGER NOT NULL DEFAULT 0,
            metadata               TEXT DEFAULT NULL,
            created_at             TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sys_tools_baseline(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_request_audit_session
            ON request_audit(session_id);
    """)
    # Migration: add session_type column if missing
    try:
        conn.execute("ALTER TABLE requests ADD COLUMN session_type TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # Already exists

    # Migration: add trace_category column if missing
    try:
        conn.execute("ALTER TABLE requests ADD COLUMN trace_category TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # Already exists

    # Migration: add jsonl_offset column if missing
    try:
        conn.execute("ALTER TABLE requests ADD COLUMN jsonl_offset INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Already exists


def insert_anthropic_trace(
    conn: sqlite3.Connection,
    req_id: str,
    timestamp: str,
    duration_ms: int,
    model: str,
    status_code: int,
    stream_mode: bool,
    jsonl_path: str,
    session_id: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> None:
    """Insert a minimal Anthropic trace record into DB for fast lookup."""
    conn.execute("""
        INSERT OR IGNORE INTO requests
            (request_id, timestamp, duration_ms, model, stream_mode, status_code,
             prompt_tokens, completion_tokens, cache_hit_tokens,
             cache_creation_tokens,
             session_uuid, jsonl_path, api_format, record_state)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'anthropic', 'completed')
    """, (
        req_id, timestamp, duration_ms, model,
        1 if stream_mode else 0, status_code,
        input_tokens, output_tokens, cache_read_tokens,
        cache_creation_tokens,
        session_id, jsonl_path,
    ))
    conn.commit()





def insert_phase1(conn: sqlite3.Connection, *, req_id: str, timestamp: str,
                  model: str, stream_mode: bool, jsonl_path: str,
                  session_id: str | None = None,
                  trace_category: str | None = None) -> None:
    """Phase 1: insert with 'recording' state."""
    conn.execute("""
        INSERT OR IGNORE INTO requests
            (request_id, timestamp, duration_ms, model, stream_mode, status_code,
             session_uuid, trace_category, jsonl_path, api_format, record_state)
        VALUES (?, ?, 0, ?, ?, 0, ?, ?, ?, 'anthropic', 'recording')
    """, (req_id, timestamp, model, 1 if stream_mode else 0,
          session_id, trace_category, jsonl_path))
    conn.commit()


def update_phase2(conn: sqlite3.Connection, *, req_id: str, duration_ms: int,
                  status_code: int, input_tokens: int = 0, output_tokens: int = 0,
                  cache_hit_tokens: int = 0, cache_creation_tokens: int = 0,
                  jsonl_path: str | None = None,
                  jsonl_offset: int = 0) -> None:
    """Phase 2: fill response data, set 'completed'."""
    total = input_tokens + output_tokens
    # cache_miss_tokens: tokens not served from cache = new input tokens
    # (DeepSeek's input_tokens excludes cached tokens)
    cache_miss = input_tokens
    total_cache = cache_hit_tokens + cache_miss
    ratio = round(cache_hit_tokens / total_cache, 4) if total_cache > 0 else 0.0
    sets = """
        duration_ms = ?, status_code = ?,
        prompt_tokens = ?, completion_tokens = ?, total_tokens = ?,
        cache_hit_tokens = ?, cache_miss_tokens = ?, cache_hit_ratio = ?,
        jsonl_offset = ?,
        record_state = 'completed'
    """
    params = [duration_ms, status_code, input_tokens, output_tokens,
              total, cache_hit_tokens, cache_miss, ratio, jsonl_offset]
    if jsonl_path:
        sets += ", jsonl_path = ?"
        params.append(jsonl_path)
    params.append(req_id)
    conn.execute(f"UPDATE requests SET {sets} WHERE request_id = ?", params)
    conn.commit()


def update_modification_flag(conn: sqlite3.Connection, *, req_id: str,
                              is_modified: bool = False,
                              lock_info: str | None = None,
                              modification_note: str | None = None) -> None:
    """Set modification flags after persist."""
    conn.execute("""\
        UPDATE requests SET is_modified = ?, lock_type = ?, modification_note = ?
        WHERE request_id = ?
    """, (1 if is_modified else 0, lock_info or "", modification_note or "", req_id))
    conn.commit()


def migrate_add_system_role_converted(conn: sqlite3.Connection) -> None:
    """Add system_role_converted column if missing (for existing DBs)."""
    try:
        conn.execute("ALTER TABLE requests ADD COLUMN system_role_converted INTEGER DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists


def update_system_role_converted(conn: sqlite3.Connection, *, req_id: str,
                                  converted: bool = True) -> None:
    """Mark a request as having had role=system converted to role=user."""
    conn.execute(
        "UPDATE requests SET system_role_converted = ? WHERE request_id = ?",
        (1 if converted else 0, req_id)
    )
    conn.commit()


def create_baseline(conn: sqlite3.Connection, *, session_id: str,
                    system_json: str, tools_json: str,
                    request_id: str, system_hash: str, tools_hash: str,
                    user_id: str = "") -> None:
    """Create sys/tools baseline for a session."""
    conn.execute(
        "INSERT OR REPLACE INTO sys_tools_baseline "
        "(session_id, system_json, tools_json, baseline_request_id, "
        " system_hash, tools_hash, created_at, user_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, system_json, tools_json, request_id,
         system_hash, tools_hash,
         datetime.now(timezone.utc).isoformat(), user_id or None))
    conn.commit()


def find_baseline_by_hashes(conn: sqlite3.Connection,
                             system_hash: str, tools_hash: str) -> dict | None:
    """Find existing baseline by sys/tools hashes (cross-session APC)."""
    row = conn.execute(
        "SELECT * FROM sys_tools_baseline WHERE system_hash = ? AND tools_hash = ? "
        "AND user_id IS NOT NULL AND user_id != '' ORDER BY created_at DESC LIMIT 1",
        (system_hash, tools_hash)).fetchone()
    return dict(row) if row else None


def get_baseline(conn: sqlite3.Connection, session_id: str) -> dict | None:
    """Get baseline for a specific session."""
    row = conn.execute(
        "SELECT * FROM sys_tools_baseline WHERE session_id = ?",
        (session_id,)).fetchone()
    return dict(row) if row else None


def insert_audit(conn: sqlite3.Connection, *, request_id: str, session_id: str,
                 system_hash: str, tools_hash: str,
                 is_baseline: bool, sys_changed: bool,
                 tools_changed: bool, was_replaced: bool,
                 user_id_was_overridden: bool = False,
                 metadata: dict | None = None) -> None:
    """Insert audit record for sys/tools change tracking."""
    conn.execute(
        "INSERT OR REPLACE INTO request_audit "
        "(request_id, session_id, system_hash, tools_hash, is_baseline, "
        " sys_changed, tools_changed, was_replaced, "
        " user_id_was_overridden, metadata, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (request_id, session_id, system_hash, tools_hash,
         1 if is_baseline else 0,
         1 if sys_changed else 0, 1 if tools_changed else 0,
         1 if was_replaced else 0,
         1 if user_id_was_overridden else 0,
         json.dumps(metadata) if metadata else None,
         datetime.now(timezone.utc).isoformat()))
    conn.commit()
