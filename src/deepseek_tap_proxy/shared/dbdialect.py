"""存储方言层模块。

业务 SQL 统一采用 %(name)s 命名占位符；方言对象为无状态（stateless），
每次调用按需构造 SQL。参考 gitlab 线 temp-master 的 dbdialect 实现。
MemoryStore 为真源，SQLite/PG 均为后备投影（backup projection）。

本模块不持有任何数据库连接，仅提供方言相关的 SQL 生成与参数转换工具。
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

# ---------------------------------------------------------------------------
# 表结构常量（业务 SQL 统一 %(name)s 命名占位符）
# ---------------------------------------------------------------------------

# 表结构以 shared/db.py 业务 SQL 的消费方列为准（schema 单一来源）。
# 6 组表全部对齐：SQLite 与 Postgres 共用同一列定义，
# 方言差异仅由 create_table_sql 转换（AUTOINCREMENT → BIGSERIAL 等）。

REQUESTS_COLUMNS = [
    "id INTEGER PRIMARY KEY AUTOINCREMENT",
    "request_id TEXT UNIQUE NOT NULL",
    "timestamp TEXT NOT NULL",
    "duration_ms INTEGER NOT NULL",
    "model TEXT NOT NULL",
    "stream_mode INTEGER NOT NULL DEFAULT 1",
    "status_code INTEGER NOT NULL",
    "total_input_tokens INTEGER DEFAULT 0",
    "output_tokens INTEGER DEFAULT 0",
    "total_tokens INTEGER DEFAULT 0",
    "cache_hit_tokens INTEGER DEFAULT 0",
    "cache_miss_tokens INTEGER DEFAULT 0",
    "cache_hit_ratio REAL DEFAULT 0.0",
    "cache_creation_tokens INTEGER DEFAULT 0",
    "has_reasoning INTEGER DEFAULT 0",
    "has_tool_calls INTEGER DEFAULT 0",
    "prompt_messages_count INTEGER DEFAULT 0",
    "prompt_tools_count INTEGER DEFAULT 0",
    "session_id TEXT DEFAULT NULL",
    "session_uuid TEXT DEFAULT NULL",
    "session_fingerprint TEXT DEFAULT NULL",  # deprecated: fingerprint-retirement V1
    "session_type TEXT DEFAULT NULL",
    "api_format TEXT DEFAULT NULL",
    "jsonl_path TEXT NOT NULL DEFAULT ''",
    "is_modified INTEGER DEFAULT 0",
    "modification_note TEXT DEFAULT ''",
    "msg_hash_chain TEXT DEFAULT NULL",
    "record_state TEXT DEFAULT 'pending'",
    "lock_type TEXT DEFAULT NULL",
    "jsonl_offset INTEGER DEFAULT 0",
    "prefix_chain_id TEXT DEFAULT NULL",
    "trace_category TEXT DEFAULT NULL",
    "system_role_converted INTEGER DEFAULT 0",
    "sys_tools_locked TEXT DEFAULT NULL",
    "chat_type TEXT DEFAULT 'normal'",
    # passthrough-policy：模式 / 端点类型 / 请求改写留痕（向后兼容：老行取默认值）
    "mode TEXT DEFAULT 'managed'",
    "kind TEXT DEFAULT 'chat'",
    "intervention INTEGER DEFAULT 0",
    "intervention_note TEXT DEFAULT ''",
    "FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE SET NULL",
    "FOREIGN KEY (prefix_chain_id) REFERENCES prefix_chains(chain_id) ON DELETE SET NULL",
]

REQUESTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests (timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_requests_model ON requests (model)",
    "CREATE INDEX IF NOT EXISTS idx_requests_cache_ratio ON requests (cache_hit_ratio)",
    "CREATE INDEX IF NOT EXISTS idx_requests_session_fp ON requests (session_fingerprint)",
    "CREATE INDEX IF NOT EXISTS idx_requests_session_uuid ON requests (session_uuid)",
    "CREATE INDEX IF NOT EXISTS idx_requests_session_id ON requests (session_id)",
    "CREATE INDEX IF NOT EXISTS idx_requests_api_format ON requests (api_format)",
]

SESSIONS_COLUMNS = [
    "session_id TEXT PRIMARY KEY",
    "client_type TEXT NOT NULL",
    "anchor_msg_index INTEGER DEFAULT 0",
    "max_hash_chain TEXT DEFAULT ''",
    "hash_chain_length INTEGER DEFAULT 0",
    "start_time TEXT NOT NULL",
    "last_seen_time TEXT NOT NULL",
    "total_requests INTEGER DEFAULT 1",
    "canonical_sys TEXT DEFAULT ''",
    "canonical_tools TEXT DEFAULT ''",
]

SESSIONS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_sessions_client_type ON sessions (client_type)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_last_seen ON sessions (last_seen_time)",
]

PREFIX_CHAINS_COLUMNS = [
    "session_id TEXT NOT NULL",
    "chain_id TEXT NOT NULL PRIMARY KEY",
    "sys TEXT NOT NULL DEFAULT '[]'",
    "tools TEXT NOT NULL DEFAULT '[]'",
    "messages TEXT NOT NULL DEFAULT '[]'",
    "sys_hashes TEXT NOT NULL DEFAULT ''",
    "tools_hashes TEXT NOT NULL DEFAULT ''",
    "messages_hashes TEXT NOT NULL DEFAULT ''",
    "msg_count INTEGER NOT NULL DEFAULT 0",
    "safe_breakpoints TEXT",
    "total_input_tokens INTEGER DEFAULT 0",
    "is_frozen INTEGER NOT NULL DEFAULT 0",
    "created_at TEXT NOT NULL",
    "last_seen TEXT NOT NULL",
    "moments TEXT DEFAULT '[]'",
    "FOREIGN KEY (session_id) REFERENCES sessions(session_id)",
]

PREFIX_CHAINS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pc_session ON prefix_chains (session_id)",
]

# canonical 的持久化真源（按 (session_id, chain_id) 隔离）。
# 组合主键与内存 MemoryStore._canonicals 的 key 一一对应；chain_id 存归一后的值
# （主代理 = session_id）。不设 FK：canonical 可能先于 sessions 行建立
# （MemoryStore 为真源，SQLite 是后备投影，FK OFF）。
CHAIN_CANONICAL_COLUMNS = [
    "session_id TEXT NOT NULL",
    "chain_id TEXT NOT NULL",
    "sys TEXT NOT NULL DEFAULT ''",
    "tools TEXT NOT NULL DEFAULT ''",
    "PRIMARY KEY (session_id, chain_id)",
]

CHAIN_TOKEN_HISTORY_COLUMNS = [
    "chain_id TEXT NOT NULL REFERENCES prefix_chains(chain_id)",
    "msg_count INTEGER NOT NULL",
    "tokens INTEGER NOT NULL",
    "char_count INTEGER NOT NULL DEFAULT 0",
    "created_at TEXT NOT NULL",
    "PRIMARY KEY (chain_id, msg_count)",
]

META_COLUMNS = [
    "key TEXT PRIMARY KEY",
    "value TEXT NOT NULL",
]

# 命名参数占位符正则：%(name)s
_NAMED_PARAM = re.compile(r"%\((\w+)\)s")

# 自增主键标记
_AUTOINC_MARKER = "INTEGER PRIMARY KEY AUTOINCREMENT"


class SqliteDialect:
    """SQLite 方言。"""

    param_style = "qmark"

    def bind(self, sql: str) -> tuple[str, list[str]]:
        """把 %(name)s 占位符替换为 ?，并返回 names 列表（按 SQL 出现顺序）。"""
        names: list[str] = []

        def _repl(match: re.Match) -> str:
            names.append(match.group(1))
            return "?"

        bound_sql = _NAMED_PARAM.sub(_repl, sql)
        return bound_sql, names

    def translate_params(self, params: dict[str, Any] | None, names: list[str]) -> tuple:
        """按 names 顺序把字典转成元组；params 为 None 时返回空元组。"""
        if params is None:
            return ()
        return tuple(params.get(name) for name in names)

    def create_table_sql(self, table: str, columns: list[str]) -> str:
        coldefs = []
        for col in columns:
            if col.startswith("id ") and coldefs and "PRIMARY KEY" in col:
                pass
            if "PRIMARY KEY" in col and col == _AUTOINC_MARKER:
                coldefs.append(_AUTOINC_MARKER)
            else:
                coldefs.append(col)
        return f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(coldefs)})"

    def insert_sql(self, table: str, columns: list[str]) -> str:
        col_names = ", ".join(columns)
        placeholders = ", ".join(f"%({c})s" for c in columns)
        return f"INSERT OR IGNORE INTO {table} ({col_names}) VALUES ({placeholders})"

    def add_column_if_missing(self, conn: Any, table: str, column_def: str) -> None:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")
        except sqlite3.OperationalError:
            # 列已存在
            pass

    def set_row_factory(self, conn: Any) -> None:
        conn.row_factory = sqlite3.Row


class PostgresDialect:
    """PostgreSQL 方言。"""

    param_style = "named"

    def bind(self, sql: str) -> tuple[str, list[str]]:
        """命名占位符原样返回，names 按出现顺序提取。"""
        names = [m.group(1) for m in _NAMED_PARAM.finditer(sql)]
        return sql, names

    def translate_params(self, params: dict[str, Any] | None, names: list[str]) -> Any:
        """原样返回 params（命名占位符由驱动处理）。"""
        return params

    def create_table_sql(self, table: str, columns: list[str]) -> str:
        coldefs = []
        for col in columns:
            if col == _AUTOINC_MARKER or "PRIMARY KEY AUTOINCREMENT" in col:
                coldefs.append("id BIGSERIAL PRIMARY KEY")
            else:
                coldefs.append(col)
        return f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(coldefs)})"

    def insert_sql(self, table: str, columns: list[str], conflict_col: str = "request_id") -> str:
        col_names = ", ".join(columns)
        placeholders = ", ".join(f"%({c})s" for c in columns)
        return (
            f"INSERT INTO {table} ({col_names}) VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict_col}) DO NOTHING"
        )

    def add_column_if_missing(self, conn: Any, table: str, column_def: str) -> None:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column_def}")

    def set_row_factory(self, conn: Any) -> None:
        from psycopg.rows import dict_row

        conn.row_factory = dict_row


def create_dialect(db_type: str) -> Any:
    """根据 db_type 创建对应方言实例。"""
    if db_type == "sqlite":
        return SqliteDialect()
    if db_type == "postgres":
        return PostgresDialect()
    raise ValueError(f"未知 db_type: {db_type}")