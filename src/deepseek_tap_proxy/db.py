from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

from deepseek_tap_proxy.shared.db import (
    create_sessions_table,
    create_prefix_chains_table,
    create_chain_token_history_table,
    create_requests_table,
    migrate_requests_rename,
    drop_deprecated_tables,
    cleanup_prefix_chains_id,
)

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
        # Unified requests table (all formats), replaces former 3-way split
        create_requests_table(conn)
        # Shared tables
        create_sessions_table(conn)
        create_prefix_chains_table(conn)
        create_chain_token_history_table(conn)
        # Data migrations
        migrate_requests_rename(conn)
        drop_deprecated_tables(conn)
        cleanup_prefix_chains_id(conn)
        conn.commit()

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

