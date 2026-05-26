from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

from deepseek_tap_proxy.openai.db import create_openai_tables
from deepseek_tap_proxy.anthropic.db import create_anthropic_tables

log = logging.getLogger("deepseek-tap")


class Database:
    """Thread-safe SQLite connection pool (WAL mode)."""

    def __init__(self, db_path: str) -> None:
        self._path = db_path
        self._local = threading.local()

    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._connect()
        return self._local.conn

    def _connect(self) -> sqlite3.Connection:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        conn = self.conn
        _create_unified_tables(conn)
        create_openai_tables(conn)
        create_anthropic_tables(conn)
        conn.commit()

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


def _create_unified_tables(conn: sqlite3.Connection) -> None:
    """Create the shared requests table with columns for all API formats.

    Uses IF NOT EXISTS so subsequent format-specific CREATE TABLE
    calls are harmless no-ops.
    """
    conn.executescript('''
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
            has_reasoning INTEGER DEFAULT 0,
            has_tool_calls INTEGER DEFAULT 0,
            prompt_messages_count INTEGER DEFAULT 0,
            prompt_tools_count INTEGER DEFAULT 0,
            session_fingerprint TEXT DEFAULT NULL,
            session_type TEXT DEFAULT NULL,
            session_uuid TEXT DEFAULT NULL,
            api_format TEXT DEFAULT NULL,
            jsonl_path TEXT NOT NULL DEFAULT '',
            turn INTEGER DEFAULT 0,
            msg0_hash TEXT DEFAULT NULL,
            msg0_content TEXT DEFAULT NULL,
            is_modified INTEGER DEFAULT 0,
            modification_note TEXT DEFAULT '',
            msg_hash_chain TEXT DEFAULT NULL,
            record_state TEXT DEFAULT 'pending',
            lock_type TEXT DEFAULT NULL,
            jsonl_offset INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);
        CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model);
        CREATE INDEX IF NOT EXISTS idx_requests_cache_ratio ON requests(cache_hit_ratio);
        CREATE INDEX IF NOT EXISTS idx_requests_session_fp ON requests(session_fingerprint);
        CREATE INDEX IF NOT EXISTS idx_requests_session_uuid ON requests(session_uuid);
    ''')

