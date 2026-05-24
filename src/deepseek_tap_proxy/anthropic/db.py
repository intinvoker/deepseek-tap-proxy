from __future__ import annotations

import sqlite3


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
            cache_hit_tokens INTEGER DEFAULT 0,
            cache_creation_tokens INTEGER DEFAULT 0,
            session_uuid TEXT DEFAULT NULL,
            jsonl_path TEXT NOT NULL DEFAULT '',
            api_format TEXT DEFAULT 'anthropic',
            turn INTEGER DEFAULT 0,
            record_state TEXT DEFAULT 'completed'
        );

        CREATE INDEX IF NOT EXISTS idx_requests_format ON requests(api_format);
        CREATE INDEX IF NOT EXISTS idx_requests_anthropic_session ON requests(session_uuid);
    """)


def insert_anthropic_trace(
    conn: sqlite3.Connection,
    req_id: str,
    timestamp: str,
    duration_ms: int,
    model: str,
    status_code: int,
    stream_mode: bool,
    jsonl_path: str,
    turn: int,
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
             session_uuid, jsonl_path, api_format, turn, record_state)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'anthropic', ?, 'completed')
    """, (
        req_id, timestamp, duration_ms, model,
        1 if stream_mode else 0, status_code,
        input_tokens, output_tokens, cache_read_tokens,
        cache_creation_tokens,
        session_id, jsonl_path, turn,
    ))
    conn.commit()


def get_previous_anthropic_turn(
    conn: sqlite3.Connection,
    session_uuid: str,
    current_turn: int,
) -> dict | None:
    """Get the previous turn's DB record for diff."""
    row = conn.execute("""
        SELECT request_id, jsonl_path FROM requests
        WHERE api_format = 'anthropic' AND session_uuid = ? AND turn = ?
        LIMIT 1
    """, (session_uuid, current_turn - 1)).fetchone()
    return dict(row) if row else None
