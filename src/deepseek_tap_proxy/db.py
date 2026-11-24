from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

from deepseek_tap_proxy.shared.db import (
    migrate_requests_rename,
    drop_deprecated_tables,
    cleanup_prefix_chains_id,
)
from deepseek_tap_proxy.shared.dbdialect import (
    create_dialect,
    REQUESTS_COLUMNS,
    REQUESTS_INDEXES,
    SESSIONS_COLUMNS,
    SESSIONS_INDEXES,
    PREFIX_CHAINS_COLUMNS,
    PREFIX_CHAINS_INDEXES,
    CHAIN_CANONICAL_COLUMNS,
    CHAIN_TOKEN_HISTORY_COLUMNS,
    META_COLUMNS,
)

log = logging.getLogger("deepseek-tap")


class _TapConnection(sqlite3.Connection):
    """sqlite3.Connection subclass that allows attaching custom attributes."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tap_db = None


class Database:
    """Thread-safe connection pool with dual-backend support (sqlite/postgres)."""

    def __init__(self, db_type: str = "sqlite", db_path: str = "", db_dsn: str = "") -> None:
        # Backward compatibility: old single-arg form Database(db_path).
        # If db_type is not a known backend, treat it as the db_path.
        if db_type not in ("sqlite", "postgres"):
            db_path, db_type = db_type, "sqlite"
        self._type = db_type
        self._path = db_path
        self._dsn = db_dsn
        self._dialect = create_dialect(db_type)
        self._local = threading.local()

    @property
    def conn(self):
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._connect()
        elif self._type == "postgres" and self._local.conn.closed:
            # 死连接自愈：服务端重启/连接被终止后 thread-local 缓存连接必须重建
            # （psycopg conn.closed = 底层 PGconn 状态 BAD）
            self._local.conn = self._connect()
        return self._local.conn

    def _connect(self):
        if self._type == "postgres":
            import psycopg
            # autocommit：每条语句立即提交、不发 BEGIN——SELECT 读路径不再悬挂事务
            # （避免读路径持有长事务快照 → TOAST 死元组无法回收 → 表膨胀）。
            # 兼容性：autocommit 下显式 commit()/rollback() 为 no-op（psycopg3 _commit_gen 语义），
            # shared/db.py 写函数无需改动；批量原子性由投影语义兜底（JSONL 事件日志=权威源）。
            conn = psycopg.connect(self._dsn, autocommit=True)
            self._dialect.set_row_factory(conn)
            conn._tap_db = self._dialect
            return conn
        # sqlite branch (default)
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        conn = _TapConnection(self._path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        # FK OFF: MemoryStore is source of truth, SQLite is backup.
        # Fire-and-forget writes may interleave session/request creation.
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.row_factory = sqlite3.Row
        setattr(conn, "_tap_db", self._dialect)
        return conn

    def init_db(self) -> None:
        conn = self.conn
        dialect = self._dialect
        # 建表顺序 = FK 依赖拓扑序（PG 严格校验目标表存在，SQLite FK OFF 掩盖了此问题）：
        #   sessions / prefix_chains 先建 → requests（FK 引用二者）→ chain_token_history
        conn.execute(dialect.create_table_sql("sessions", SESSIONS_COLUMNS))
        for idx in SESSIONS_INDEXES:
            conn.execute(idx)
        conn.execute(dialect.create_table_sql("prefix_chains", PREFIX_CHAINS_COLUMNS))
        for idx in PREFIX_CHAINS_INDEXES:
            conn.execute(idx)
        # canonical 落盘真源（按 (session_id, chain_id) 隔离）。无 FK → 建表顺序无关。
        conn.execute(dialect.create_table_sql("chain_canonical", CHAIN_CANONICAL_COLUMNS))
        # ── 升级迁移：老库 canonical 的两列 → chain_canonical 表 ──
        # 必须紧随建表、且在首次 checkpoint 之前 —— 新代码不再读 sessions.canonical_*，
        # 而 checkpoint 是 DELETE FROM sessions + 重 INSERT（列清单里没有那两列），
        # 升级后第一次 checkpoint 就会把老库的 canonical 唯一副本清成 DEFAULT ''，
        # 且 canonical_set 事件通常已被上次 checkpoint 归档删除 → 事后无从恢复。
        # 映射：旧格式是 session 级的 → 归属默认链（chain_id = session_id），
        # 与归一规则 _canonical_key(sid, "") == (sid, sid) 一致。
        # 幂等：ON CONFLICT DO NOTHING ⇒ 已有新值不被老值覆盖，重复启动安全。
        # 只生效一次：停写后首次 checkpoint 清空那两列，后续启动该 SELECT 无行可迁。
        # COALESCE：老列经 ALTER 加列时可能为 NULL，直接 `!= ''` 会漏行，
        # 且把 NULL 写进 NOT NULL 的 sys/tools 会中断启动。
        conn.execute(
            """INSERT INTO chain_canonical (session_id, chain_id, sys, tools)
               SELECT session_id, session_id,
                      COALESCE(canonical_sys, ''), COALESCE(canonical_tools, '')
               FROM sessions
               WHERE COALESCE(canonical_sys, '') != ''
                  OR COALESCE(canonical_tools, '') != ''
               ON CONFLICT (session_id, chain_id) DO NOTHING"""
        )
        # Unified requests table (all formats), replaces former 3-way split
        conn.execute(dialect.create_table_sql("requests", REQUESTS_COLUMNS))
        for idx in REQUESTS_INDEXES:
            conn.execute(idx)
        # 迁移：passthrough-policy 新增列（老库 ALTER TABLE 补齐；PG 用 IF NOT EXISTS）
        for _col in (
            "mode TEXT DEFAULT 'managed'",
            "kind TEXT DEFAULT 'chat'",
            "intervention INTEGER DEFAULT 0",
            "intervention_note TEXT DEFAULT ''",
        ):
            dialect.add_column_if_missing(conn, "requests", _col)
        conn.execute(dialect.create_table_sql("chain_token_history", CHAIN_TOKEN_HISTORY_COLUMNS))
        conn.execute(dialect.create_table_sql("meta", META_COLUMNS))
        # Data migrations (sqlite only)
        if self._type == "sqlite":
            migrate_requests_rename(conn)
            drop_deprecated_tables(conn)
            cleanup_prefix_chains_id(conn)
        conn.commit()
        # L2: 迁移函数会临时 PRAGMA foreign_keys=OFF→ON（shared/db.py 迁移），
        # 此处恢复 _connect 的设计意图（MemoryStore 为真源，FK 应 OFF）
        if self._type == "sqlite":
            conn.execute("PRAGMA foreign_keys=OFF")

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None