"""Cross-format database functions shared across providers.

Pure functions that take ``conn`` (sqlite3.Connection) as first parameter.
These operate on the ``sessions``, ``prefix_chains``, and unified ``requests``
tables which are format-agnostic.

All insert/update functions are ctx-based — they read from ``Context.persist``
and properties, never from raw format-specific structures.
"""

from __future__ import annotations

import json
import logging
import orjson
import sqlite3

log = logging.getLogger("deepseek-tap")
from datetime import datetime, timezone

from deepseek_tap_proxy.shared.dbdialect import SqliteDialect


def _dialect_of(conn):
    """Return the dialect attached to ``conn`` (``_tap_db``), else a SqliteDialect."""
    dialect = getattr(conn, "_tap_db", None)
    if dialect is not None:
        return dialect
    return SqliteDialect()


def _exec(conn, sql, params):
    """Execute ``sql`` (with %(name)s named placeholders) via the conn's dialect."""
    dialect = _dialect_of(conn)
    bound_sql, names = dialect.bind(sql)
    return conn.execute(bound_sql, dialect.translate_params(params, names))


# ── Sessions table helpers ──


def get_session_by_id(conn: sqlite3.Connection, session_id: str) -> dict | None:
    """Get session by ID. Returns dict or None."""
    row = _exec(conn, "SELECT * FROM sessions WHERE session_id = %(session_id)s",
                {"session_id": session_id}).fetchone()
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
        _exec(conn, """
            UPDATE sessions SET
                last_seen_time = %(now)s,
                total_requests = total_requests + 1
            WHERE session_id = %(session_id)s
        """, {"now": now, "session_id": session_id})
        if len(curr_hashes) > existing["hash_chain_length"]:
            hash_chain_str = ",".join(curr_hashes)
            _exec(conn, """
                UPDATE sessions SET max_hash_chain = %(hash_chain_str)s, hash_chain_length = %(len)s
                WHERE session_id = %(session_id)s
            """, {"hash_chain_str": hash_chain_str, "len": len(curr_hashes),
                  "session_id": session_id})
        conn.commit()
        return dict(_exec(conn, "SELECT * FROM sessions WHERE session_id = %(session_id)s",
                          {"session_id": session_id}).fetchone())
    else:
        hash_chain_str = ",".join(curr_hashes)
        _exec(conn, """
            INSERT INTO sessions (session_id, client_type, anchor_msg_index,
                max_hash_chain, hash_chain_length, start_time, last_seen_time, total_requests)
            VALUES (%(session_id)s, %(client_type)s, %(anchor_msg_index)s,
                    %(hash_chain_str)s, %(len)s, %(now)s, %(now)s, 1)
        """, {"session_id": session_id, "client_type": client_type,
              "anchor_msg_index": anchor_msg_index, "hash_chain_str": hash_chain_str,
              "len": len(curr_hashes), "now": now})
        conn.commit()
        return dict(_exec(conn, "SELECT * FROM sessions WHERE session_id = %(session_id)s",
                          {"session_id": session_id}).fetchone())


def update_session_hash_chain(
    conn: sqlite3.Connection,
    session_id: str,
    curr_hashes: list[str],
) -> None:
    """Update max_hash_chain only if curr_hashes is longer than current."""
    existing = get_session_by_id(conn, session_id)
    if existing and len(curr_hashes) > existing["hash_chain_length"]:
        hash_chain_str = ",".join(curr_hashes)
        _exec(conn, """
            UPDATE sessions SET max_hash_chain = %(hash_chain_str)s, hash_chain_length = %(len)s
            WHERE session_id = %(session_id)s
        """, {"hash_chain_str": hash_chain_str, "len": len(curr_hashes),
              "session_id": session_id})
        conn.commit()


def find_candidate_sessions(conn: sqlite3.Connection, client_type: str) -> list[dict]:
    """Find active sessions matching client_type, ordered by last_seen desc."""
    rows = _exec(conn, """
        SELECT * FROM sessions
        WHERE client_type = %(client_type)s AND hash_chain_length > 0
        ORDER BY last_seen_time DESC LIMIT 50
    """, {"client_type": client_type}).fetchall()
    return [dict(r) for r in rows]


# ── Prefix chains table helpers ──


def upsert_prefix_chain(conn: sqlite3.Connection, session_id: str, chain_id: str, *,
                        sys: str, tools: str, messages: str,
                        sys_hashes: str, tools_hashes: str, messages_hashes: str,
                        msg_count: int,
                        safe_breakpoints: list[int] | None = None,
                        total_input_tokens: int = 0,
                        is_frozen: bool = False,
                        moments: str = "[]",
                        commit: bool = True) -> None:
    """Insert or update a prefix chain. Only updates if msg_count is longer.

    ``sys``, ``tools``, ``messages`` are JSON strings — canonical form triples.
    ``sys_hashes``, ``tools_hashes``, ``messages_hashes`` are comma-separated hash strings.
    ``total_input_tokens`` is the upstream's total input token count for this request,
    used for estimating KV cache hit rates on subsequent requests.
    ``is_frozen`` marks a public prefix chain — never updated once created.
    ``moments`` is a JSON array of Moment dicts.

    ``commit=False`` 时调用方负责 commit/rollback（用于批量事务，writer 统一提交）。
    """
    now = datetime.now(timezone.utc).isoformat()
    # PG is_frozen 列是 INTEGER；Python bool → psycopg 适配为 boolean → DatatypeMismatch
    # （SQLite 隐式 0/1 从未暴露）。显式归一。
    is_frozen = int(is_frozen)
    safe_breakpoints_json = orjson.dumps(safe_breakpoints).decode() if safe_breakpoints is not None else None
    # ON CONFLICT(...) DO UPDATE：SQLite 3.24+ 与 PostgreSQL 均支持的 UPSERT 语法
    # 命名占位符可重复出现（_exec 按 names 顺序逐次取值，双方言一致）
    _exec(conn, """
        INSERT INTO prefix_chains
            (session_id, chain_id, sys, tools, messages,
             sys_hashes, tools_hashes, messages_hashes,
             msg_count,
             safe_breakpoints, total_input_tokens, is_frozen, created_at, last_seen, moments)
        VALUES (%(session_id)s, %(chain_id)s, %(sys)s, %(tools)s, %(messages)s,
                %(sys_hashes)s, %(tools_hashes)s, %(messages_hashes)s,
                %(msg_count)s,
                %(safe_breakpoints_json)s, %(total_input_tokens)s, %(is_frozen)s, %(now)s, %(now)s, %(moments)s)
        ON CONFLICT(chain_id) DO UPDATE SET
            sys = CASE WHEN %(msg_count)s > prefix_chains.msg_count AND prefix_chains.is_frozen = 0
                       THEN %(sys)s ELSE prefix_chains.sys END,
            tools = CASE WHEN %(msg_count)s > prefix_chains.msg_count AND prefix_chains.is_frozen = 0
                         THEN %(tools)s ELSE prefix_chains.tools END,
            messages = CASE WHEN %(msg_count)s > prefix_chains.msg_count AND prefix_chains.is_frozen = 0
                            THEN %(messages)s ELSE prefix_chains.messages END,
            sys_hashes = CASE WHEN %(msg_count)s > prefix_chains.msg_count AND prefix_chains.is_frozen = 0
                              THEN %(sys_hashes)s ELSE prefix_chains.sys_hashes END,
            tools_hashes = CASE WHEN %(msg_count)s > prefix_chains.msg_count AND prefix_chains.is_frozen = 0
                                THEN %(tools_hashes)s ELSE prefix_chains.tools_hashes END,
            messages_hashes = CASE WHEN %(msg_count)s > prefix_chains.msg_count AND prefix_chains.is_frozen = 0
                                   THEN %(messages_hashes)s ELSE prefix_chains.messages_hashes END,
            msg_count = CASE WHEN %(msg_count)s > prefix_chains.msg_count AND prefix_chains.is_frozen = 0
                             THEN %(msg_count)s ELSE prefix_chains.msg_count END,
            safe_breakpoints = CASE WHEN %(msg_count)s > prefix_chains.msg_count AND prefix_chains.is_frozen = 0
                                    THEN %(safe_breakpoints_json)s ELSE prefix_chains.safe_breakpoints END,
            total_input_tokens = CASE WHEN %(msg_count)s > prefix_chains.msg_count THEN %(total_input_tokens)s ELSE prefix_chains.total_input_tokens END,
            last_seen = %(now)s
    """, {"session_id": session_id, "chain_id": chain_id, "sys": sys, "tools": tools,
          "messages": messages, "sys_hashes": sys_hashes, "tools_hashes": tools_hashes,
          "messages_hashes": messages_hashes, "msg_count": msg_count,
          "safe_breakpoints_json": safe_breakpoints_json,
          "total_input_tokens": total_input_tokens, "is_frozen": is_frozen,
          "now": now, "moments": moments})
    if commit:
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
    row = _exec(conn,
                "SELECT canonical_sys, canonical_tools FROM sessions WHERE session_id = %(session_id)s",
                {"session_id": session_id}).fetchone()
    if row:
        # PG dict_row vs SQLite Row：统一键访问（整数索引仅 SQLite 支持）
        return {"sys": row["canonical_sys"] or "", "tools": row["canonical_tools"] or ""}
    return {"sys": "", "tools": ""}


def set_session_canonical(conn: sqlite3.Connection, session_id: str,
                          sys: str = "", tools: str = "",
                          *, commit: bool = True) -> None:
    """Set canonical sys/tools for a session (UPDATE only non-empty values).

    ``commit=False`` 时调用方负责 commit/rollback（用于批量事务，writer 统一提交）。
    """
    sets = []
    params = {}
    if sys:
        sets.append("canonical_sys = %(sys)s")
        params["sys"] = sys
    if tools:
        sets.append("canonical_tools = %(tools)s")
        params["tools"] = tools
    if not sets:
        return
    params["session_id"] = session_id
    cur = _exec(
        conn,
        f"UPDATE sessions SET {', '.join(sets)} WHERE session_id = %(session_id)s",
        params,
    )
    # 显式化：session 行不存在 → UPDATE 0 行静默失败。sessions 表已废弃
    # （upsert_session 无调用点，6/24 后不落库），canonical 落盘长期失效——
    # 打 warning 暴露，避免再次潜伏数周。
    if cur.rowcount == 0:
        log.warning(
            "set_session_canonical: session %r not in sessions table, "
            "UPDATE affected 0 rows (upsert_session 无调用点 → sessions 表未维护)",
            session_id,
        )
    if commit:
        conn.commit()


# ── Chain canonical table（canonical 持久化真源）──
#
# canonical 按 (session_id, chain_id) 隔离：Claude Code 子代理复用主会话
# session-id，session 级单份存储会让子代理冻结面被主代理整体覆写。
# chain_id 一律存**归一后**的值（主代理 = session_id，见 MemoryStore._canonical_key），
# 读回时直接构成内存 key。本层保持「存什么读什么」——库中若残留空 chain_id 的
# 历史行（漏归一的旧写入方），由装载侧 MemoryStore._normalize_loaded_canonicals
# 容错归一，规则不在两层各存一份。


def create_chain_canonical_table(conn: sqlite3.Connection) -> None:
    """Create the chain_canonical table (canonical 的落盘真源)。"""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chain_canonical (
            session_id  TEXT NOT NULL,
            chain_id    TEXT NOT NULL,
            sys         TEXT NOT NULL DEFAULT '',
            tools       TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (session_id, chain_id)
        );
    """)
    conn.commit()


def get_chain_canonical(conn: sqlite3.Connection, session_id: str,
                        chain_id: str) -> dict[str, str]:
    """Get canonical sys/tools for one (session_id, chain_id). 缺失返回两个空串。"""
    row = _exec(conn,
                "SELECT sys, tools FROM chain_canonical "
                "WHERE session_id = %(session_id)s AND chain_id = %(chain_id)s",
                {"session_id": session_id, "chain_id": chain_id}).fetchone()
    if row:
        return {"sys": row["sys"] or "", "tools": row["tools"] or ""}
    return {"sys": "", "tools": ""}


def set_chain_canonical(conn: sqlite3.Connection, session_id: str, chain_id: str,
                        *, sys: str = "", tools: str = "",
                        commit: bool = True) -> None:
    """UPSERT canonical for one (session_id, chain_id) —— 仅覆盖非空值。

    单条 UPSERT 完成读改写（无 read-modify-write 竞态），语义与
    ``set_session_canonical`` 一致：空入参不覆盖已落库的值。sys/tools 均空
    时不建行（无实际冻结内容）。

    ``commit=False`` 时调用方负责 commit/rollback（用于批量事务，writer 统一提交）。
    """
    if not sys and not tools:
        return
    _exec(conn, """
        INSERT INTO chain_canonical (session_id, chain_id, sys, tools)
        VALUES (%(session_id)s, %(chain_id)s, %(sys)s, %(tools)s)
        ON CONFLICT (session_id, chain_id) DO UPDATE SET
            sys = CASE WHEN %(sys)s != '' THEN %(sys)s ELSE chain_canonical.sys END,
            tools = CASE WHEN %(tools)s != '' THEN %(tools)s
                         ELSE chain_canonical.tools END
    """, {"session_id": session_id, "chain_id": chain_id,
          "sys": sys, "tools": tools})
    if commit:
        conn.commit()


def load_all_chain_canonicals(conn: sqlite3.Connection,
                              ) -> dict[tuple[str, str], tuple[str, str]]:
    """一次读全部 canonical → {(session_id, chain_id): (sys, tools)}。

    返回形态与 ``MemoryStore._canonicals`` 同形，供冷启动批量装载（避免 N+1）。
    表不存在（旧库未迁移）时由调用方捕获 ``sqlite3.OperationalError``。
    """
    rows = _exec(conn,
                 "SELECT session_id, chain_id, sys, tools FROM chain_canonical",
                 None).fetchall()
    return {(r["session_id"], r["chain_id"]): (r["sys"] or "", r["tools"] or "")
            for r in rows}


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
    # Migration: three-column hash
    for col in ("sys_hashes", "tools_hashes", "messages_hashes"):
        try:
            conn.execute(f"ALTER TABLE prefix_chains ADD COLUMN {col} TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass
    # Migration: moments JSON column
    try:
        conn.execute("ALTER TABLE prefix_chains ADD COLUMN moments TEXT DEFAULT '[]'")
    except sqlite3.OperationalError:
        pass
    conn.commit()


def get_session_chains(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    """Get all prefix chains for a session, ordered by last_seen desc.

    Returns rows with canonical three-column format:
    ``sys``, ``tools``, ``messages`` — all JSON strings.
    """
    rows = _exec(conn, """
        SELECT * FROM prefix_chains WHERE session_id = %(session_id)s ORDER BY last_seen DESC
    """, {"session_id": session_id}).fetchall()
    return [dict(r) for r in rows]


# ── Chain token history table (deprecated, kept for backward compat) ──


def create_chain_token_history_table(conn: sqlite3.Connection) -> None:
    """Create the chain_token_history table for backward compatibility.

    The table exists in existing SQLite databases and is kept for
    backward compatibility. No new data is written to it — token
    estimates are now stored in chain moments.
    """
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
            session_fingerprint   TEXT DEFAULT NULL,  -- deprecated: fingerprint-retirement V1（只读历史列，新查询用 session_id/chain_id）
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
            mode                  TEXT DEFAULT 'managed',
            kind                  TEXT DEFAULT 'chat',
            intervention          INTEGER DEFAULT 0,
            intervention_note     TEXT DEFAULT '',
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
    # Migration: add chat_type column
    try:
        conn.execute("ALTER TABLE requests ADD COLUMN chat_type TEXT DEFAULT 'normal'")
    except sqlite3.OperationalError:
        pass
    # Migration: add sys_tools_locked column
    try:
        conn.execute("ALTER TABLE requests ADD COLUMN sys_tools_locked TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass
    # Migration: passthrough-policy columns
    # 老库补列（新库已在上方 CREATE TABLE 中带上）；SQLite 不支持 ADD COLUMN IF NOT EXISTS，
    # 用 OperationalError 兜底（列已存在）。
    for _col in (
        "mode TEXT DEFAULT 'managed'",
        "kind TEXT DEFAULT 'chat'",
        "intervention INTEGER DEFAULT 0",
        "intervention_note TEXT DEFAULT ''",
    ):
        try:
            conn.execute(f"ALTER TABLE requests ADD COLUMN {_col}")
        except sqlite3.OperationalError:
            pass
    conn.commit()


def insert_request_phase1(conn: sqlite3.Connection, ctx, *,
                          jsonl_offset: int = 0, commit: bool = True) -> None:
    """Unified Phase 1 INSERT: request-side data, ``record_state='pending'``.

    Reads from ``ctx.persist`` (synced by RequestAnalysisStage and CanonicalStage).

    ``commit=False`` 时调用方负责 commit/rollback（用于批量事务，writer 统一提交）。
    """
    model = ctx.body_readonly.get("model", "") if ctx.body_readonly else ""
    timestamp = datetime.now(timezone.utc).isoformat()

    # INSERT OR IGNORE 是 SQLite 专有语法 → 统一用 UPSERT 写法（SQLite 3.24+/PG 均支持）
    _exec(conn, """
        INSERT INTO requests
            (request_id, timestamp, duration_ms, status_code, model, stream_mode,
             prompt_messages_count, prompt_tools_count,
             is_modified, modification_note,
             session_id, session_fingerprint, session_type,
             msg_hash_chain,
             jsonl_path, jsonl_offset, record_state,
             api_format, trace_category, chat_type, mode, kind)
        VALUES (%(request_id)s, %(timestamp)s, 0, 0, %(model)s, %(stream_mode)s,
                %(prompt_messages_count)s, %(prompt_tools_count)s,
                0, '', %(session_id)s, %(session_fingerprint)s, %(session_type)s,
                %(msg_hash_chain)s,
                %(jsonl_path)s, %(jsonl_offset)s, 'pending',
                %(api_format)s, %(trace_category)s, %(chat_type)s,
                %(mode)s, %(kind)s)
        ON CONFLICT(request_id) DO NOTHING
    """, {
        "request_id": ctx.req_id,
        "timestamp": timestamp,
        "model": model,
        "stream_mode": 1 if ctx.is_streaming else 0,
        "prompt_messages_count": ctx.persist.prompt_messages_count,
        "prompt_tools_count": ctx.persist.prompt_tools_count,
        "session_id": ctx.persist.session_id or None,
        "session_fingerprint": "",                       # deprecated
        "session_type": ctx.persist.client_type or None,  # session_type → client_type
        "msg_hash_chain": ctx.persist.msg_hash_chain,
        "jsonl_path": ctx.meta.get("jsonl_path", ""),
        "jsonl_offset": jsonl_offset,
        "api_format": ctx.persist.api_format,
        "trace_category": ctx.persist.trace_category,
        "chat_type": ctx.persist.chat_type or "normal",
        "mode": getattr(ctx.persist, "mode", "") or "managed",
        "kind": getattr(ctx.persist, "kind", "") or "",
    })
    if commit:
        conn.commit()


# ── 快照标识：区分「未提供快照」和「快照值为 None」──
_SENTINEL_NO_SNAPSHOT = object()


def update_request_phase2(conn: sqlite3.Connection, ctx, *,
                          jsonl_path: str = '',
                          lock_type_snapshot: str | None | object = _SENTINEL_NO_SNAPSHOT,
                          is_modified_snapshot: bool = False,
                          mod_note_snapshot: str = "",
                          commit: bool = True) -> None:
    """Unified Phase 2 UPDATE: fill response data, ``record_state='completed'``.

    Reads response data from ``ctx.persist`` (populated by
    ``apply_cache_analysis`` and the ForwardStage), and ``ctx.property`` view
    fields.

    ``lock_type_snapshot`` / ``is_modified_snapshot`` / ``mod_note_snapshot``
    override ``ctx.persist`` values — callers should pass the values captured
    immediately after pipeline execution to avoid async closure races.

    ``commit=False`` 时调用方负责 commit/rollback（用于批量事务，writer 统一提交）。
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
    _exec(conn, """
        UPDATE requests SET
            duration_ms = %(duration_ms)s, status_code = %(status_code)s,
            total_input_tokens = %(total_input_tokens)s, output_tokens = %(output_tokens)s,
            total_tokens = %(total_tokens)s,
            cache_hit_tokens = %(cache_hit_tokens)s, cache_miss_tokens = %(cache_miss_tokens)s,
            cache_hit_ratio = %(cache_hit_ratio)s,
            cache_creation_tokens = %(cache_creation_tokens)s,
            has_reasoning = %(has_reasoning)s, has_tool_calls = %(has_tool_calls)s,
            jsonl_path = %(jsonl_path)s,
            jsonl_offset = %(jsonl_offset)s,
            lock_type = %(lock_type)s,
            prefix_chain_id = %(prefix_chain_id)s,
            is_modified = %(is_modified)s,
            modification_note = %(modification_note)s,
            system_role_converted = %(system_role_converted)s,
            sys_tools_locked = %(sys_tools_locked)s,
            intervention = %(intervention)s,
            intervention_note = %(intervention_note)s,
            record_state = 'completed'
        WHERE request_id = %(request_id)s
    """, {
        "duration_ms": ctx.duration_ms or ctx.elapsed_ms,
        "status_code": ctx.status,
        "total_input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cache_hit_tokens": ctx.persist.cache_hit_tokens,
        "cache_miss_tokens": ctx.persist.cache_miss_tokens,
        "cache_hit_ratio": ctx.persist.cache_hit_ratio,
        "cache_creation_tokens": ctx.persist.cache_creation_tokens,
        "has_reasoning": int(ctx.persist.has_reasoning),
        "has_tool_calls": int(ctx.persist.has_tool_calls),
        "jsonl_path": jsonl_path,
        "jsonl_offset": ctx.meta.get("jsonl_offset", 0),
        "lock_type": _lock_type,
        "prefix_chain_id": ctx.persist.prefix_chain_id or None,
        "is_modified": _is_modified,
        "modification_note": _mod_note,
        "system_role_converted": int(ctx.persist.system_role_converted),
        "sys_tools_locked": ctx.persist.sys_tools_locked or None,
        "intervention": int(_is_modified),
        "intervention_note": _mod_note or "",
        "request_id": ctx.req_id,
    })
    if commit:
        conn.commit()


# ── Query functions (operate on the unified requests table) ─────────


def get_request_jsonl_info(conn: sqlite3.Connection, request_id: str) -> dict | None:
    """Get (jsonl_path, jsonl_offset) for a request."""
    row = _exec(conn, """
        SELECT jsonl_path, jsonl_offset FROM requests
        WHERE request_id = %(request_id)s
    """, {"request_id": request_id}).fetchone()
    return dict(row) if row else None


def get_previous_by_session_id(
    conn, session_id: str, exclude_req_id: str,
) -> tuple[str | None, int] | None:
    """Get (msg_hash_chain, prompt_messages_count) by session_id."""
    row = _exec(conn, """
        SELECT msg_hash_chain, prompt_messages_count FROM requests
        WHERE session_id = %(session_id)s AND request_id != %(exclude_req_id)s
          AND msg_hash_chain IS NOT NULL
        ORDER BY id DESC LIMIT 1
    """, {"session_id": session_id, "exclude_req_id": exclude_req_id}).fetchone()
    if row and row["msg_hash_chain"]:
        return (row["msg_hash_chain"], row["prompt_messages_count"])
    return None


def get_previous_jsonl_by_session_id(
    conn, session_id: str, exclude_req_id: str,
) -> tuple[str, str, int] | None:
    """Get (jsonl_path, request_id, jsonl_offset) by session_id."""
    row = _exec(conn, """
        SELECT jsonl_path, request_id, jsonl_offset FROM requests
        WHERE session_id = %(session_id)s AND request_id != %(exclude_req_id)s AND jsonl_path != ''
        ORDER BY id DESC LIMIT 1
    """, {"session_id": session_id, "exclude_req_id": exclude_req_id}).fetchone()
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


# ── Meta table (checkpoint_offset) ──


def create_meta_table(conn: sqlite3.Connection) -> None:
    """Key-value meta table，存 checkpoint_offset 等。"""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = _exec(conn, "SELECT value FROM meta WHERE key = %(key)s",
                {"key": key}).fetchone()
    if row is None:
        return None
    # PG dict_row vs SQLite Row：统一键访问
    return row["value"] if isinstance(row, dict) else row[0]


def set_meta(conn: sqlite3.Connection, key: str, value: str, *, commit: bool = True) -> None:
    _exec(conn,
        "INSERT INTO meta (key, value) VALUES (%(key)s, %(value)s) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        {"key": key, "value": value},
    )
    if commit:
        conn.commit()