"""OpenAI-specific database functions.

Pure functions that take ``conn`` (sqlite3.Connection) as first parameter.
Format-specific fields are read from ``ctx.meta``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _is_copilot_msg0(msgs: list) -> bool:
    """Identify Copilot session by message structure.

    Copilot signature:
      msg[0] = System Prompt (starts with 'You are an expert', 100KB+)
      msg[1] = Environment Info (<environment_info>, injected once per session)
      msg[2+] = Conversation layer (attachments, user messages, assistant replies)
    """
    if len(msgs) < 2:
        return False
    c0 = msgs[0].get("content", "")
    c1 = msgs[1].get("content", "")
    return (
        isinstance(c0, str)
        and c0.startswith("You are an expert")
        and isinstance(c1, str)
        and "<environment_info>" in c1
    )


def _is_hermes_msg0(msgs: list) -> bool:
    """Identify Hermes session by message structure.

    Hermes signature:
      msg[0] role = "system" (system persona like '# Hermes Agent...')
      msg[1]+ = user/assistant conversation
    """
    if len(msgs) < 1:
        return False
    return msgs[0].get("role", "") == "system"


def _compute_copilot_fingerprint(msgs: list) -> str | None:
    """Copilot session fingerprint from available messages."""
    if len(msgs) < 3:
        if len(msgs) >= 2:
            c1 = msgs[1].get("content", "")
            if isinstance(c1, str):
                return hashlib.md5(c1.encode("utf-8")).hexdigest()[:16]
        if len(msgs) >= 1:
            c0 = msgs[0].get("content", "")
            if isinstance(c0, str):
                return hashlib.md5(c0.encode("utf-8")).hexdigest()[:16]
        return None
    c1 = msgs[1].get("content", "")
    c2 = msgs[2].get("content", "")
    if isinstance(c1, str) and isinstance(c2, str):
        return hashlib.md5((c1 + "<<<SEP>>>" + c2[:500]).encode("utf-8")).hexdigest()[:16]
    return None


def _compute_hermes_fingerprint(msgs: list) -> str | None:
    """Hermes session fingerprint: msg[1] onward (skip system persona)."""
    if len(msgs) < 2:
        c0 = msgs[0].get("content", "")
        if isinstance(c0, str):
            return hashlib.md5(c0.encode("utf-8")).hexdigest()[:16]
        return None
    parts = []
    for i in range(1, min(len(msgs), 5)):
        content = msgs[i].get("content", "")
        if isinstance(content, str):
            parts.append(content)
        else:
            return None
    if not parts:
        return None
    return hashlib.md5(("<<<MSG>>>".join(parts)).encode("utf-8")).hexdigest()[:16]


def _compute_generic_fingerprint(msgs: list) -> str | None:
    """Generic session fingerprint from available messages (up to first 4)."""
    if len(msgs) < 1:
        return None
    parts = []
    for i in range(min(len(msgs), 4)):
        content = msgs[i].get("content", "")
        if isinstance(content, str):
            parts.append(content)
        else:
            return None
    if not parts:
        return None
    return hashlib.md5(("<<<MSG>>>".join(parts)).encode("utf-8")).hexdigest()[:16]


def _resolve_session_fingerprint(record: dict) -> tuple[str, str] | tuple[None, None]:
    """Resolve session fingerprint and type.

    Type identification:
      Copilot: msg[0] starts with 'You are an expert' AND msg[1] has <environment_info>
      Generic: everything else

    Fingerprint uses as many available messages as possible (1-4).
    Returns None only if record has no messages at all.
    """
    msgs = record.get("request", {}).get("body", {}).get("messages", [])
    if not msgs or not isinstance(msgs, list) or len(msgs) < 1:
        return None, None

    if _is_copilot_msg0(msgs):
        fp: str | None = _compute_copilot_fingerprint(msgs)
        return (fp, "copilot") if fp else (None, None)

    if _is_hermes_msg0(msgs):
        fp: str | None = _compute_hermes_fingerprint(msgs)
        return (fp, "hermes") if fp else (None, None)

    fp: str | None = _compute_generic_fingerprint(msgs)
    return (fp, "generic") if fp else (None, None)


def create_openai_tables(conn: sqlite3.Connection) -> None:
    """Create and migrate OpenAI requests table and related schema."""
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
            has_reasoning INTEGER DEFAULT 0,
            has_tool_calls INTEGER DEFAULT 0,
            prompt_messages_count INTEGER DEFAULT 0,
            prompt_tools_count INTEGER DEFAULT 0,
            session_fingerprint TEXT DEFAULT NULL,
            session_type TEXT DEFAULT NULL,
            jsonl_path TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);
        CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model);
        CREATE INDEX IF NOT EXISTS idx_requests_cache_ratio ON requests(cache_hit_ratio);

    """)

    # Migrations for columns added after initial creation
    for col in ("msg0_hash", "msg0_content", "modification_note"):
        try:
            conn.execute(f"ALTER TABLE requests ADD COLUMN {col} TEXT DEFAULT NULL")
        except sqlite3.OperationalError:
            pass
    try:
        conn.execute("ALTER TABLE requests ADD COLUMN is_modified INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE requests ADD COLUMN record_state TEXT DEFAULT 'completed'")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE requests ADD COLUMN api_format TEXT DEFAULT 'openai'")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE requests ADD COLUMN turn INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_format ON requests(api_format)")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE requests ADD COLUMN msg_hash_chain TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE requests ADD COLUMN session_uuid TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_session ON requests(session_fingerprint)")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE requests ADD COLUMN session_type TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass
    for col_name, col_def in [
        ("lock_type", "TEXT DEFAULT NULL"),
        ("jsonl_offset", "INTEGER DEFAULT 0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE requests ADD COLUMN {col_name} {col_def}")
        except sqlite3.OperationalError:
            pass
    conn.commit()


def insert_request_phase1(conn: sqlite3.Connection, ctx, jsonl_offset: int = 0) -> None:
    """Phase 1 INSERT: request-side data only. is_modified always 0."""
    model: str = ""
    if ctx.req_body:
        model = ctx.req_body.get("model", "unknown")
    timestamp: str = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT OR IGNORE INTO requests
            (request_id, timestamp, duration_ms, status_code, model, stream_mode,
             prompt_messages_count, prompt_tools_count,
             msg0_hash, msg0_content,
             is_modified, modification_note,
             session_fingerprint, session_type,
             msg_hash_chain, session_uuid,
             jsonl_path, jsonl_offset, record_state)
        VALUES (?, ?, 0, 0, ?, ?, ?, ?, ?, ?, 0, '', ?, ?,
                ?, ?, '', ?, 'pending')
    """, (
        ctx.req_id,
        timestamp,
        model,
        1 if ctx.is_streaming else 0,
        ctx.meta.get("prompt_messages_count", 0),
        ctx.meta.get("prompt_tools_count", 0),
        ctx.meta.get("msg0_hash"),
        ctx.meta.get("msg0_content"),
        ctx.meta.get("session_fingerprint"),
        ctx.meta.get("session_type"),
        ctx.meta.get("msg_hash_chain"),
        ctx.meta.get("session_uuid"),
        jsonl_offset,
    ))
    conn.commit()


def update_request_phase2(conn: sqlite3.Connection, ctx, jsonl_path: str = '') -> None:
    """Phase 2 UPDATE: response data. state='completed'."""
    prompt_tokens = ctx.meta.get("prompt_tokens", 0)
    completion_tokens = ctx.meta.get("completion_tokens", 0)
    conn.execute("""
        UPDATE requests SET
            duration_ms = ?, status_code = ?,
            prompt_tokens = ?, completion_tokens = ?,
            total_tokens = ?,
            cache_hit_tokens = ?, cache_miss_tokens = ?,
            cache_hit_ratio = ?,
            has_reasoning = ?, has_tool_calls = ?,
            jsonl_path = ?,
                jsonl_offset = ?,
                lock_type = ?,
            record_state = 'completed'
        WHERE request_id = ?
    """, (
        ctx.duration_ms or ctx.elapsed_ms,
        ctx.status,
        prompt_tokens,
        completion_tokens,
        prompt_tokens + completion_tokens,
        ctx.meta.get("cache_hit_tokens", 0),
        ctx.meta.get("cache_miss_tokens", 0),
        ctx.meta.get("cache_hit_ratio", 0.0),
        1 if ctx.meta.get("has_reasoning", False) else 0,
        1 if ctx.meta.get("has_tool_calls", False) else 0,
        jsonl_path,
        ctx.meta.get("jsonl_offset", 0),
        ctx.meta.get("lock_type"),
        ctx.req_id,
    ))
    conn.commit()


def update_modification_flag(conn: sqlite3.Connection, ctx) -> None:
    """Update is_modified flag after Phase 1 INSERT completes."""
    conn.execute("""
        UPDATE requests SET
            is_modified = 1,
            modification_note = ?
        WHERE request_id = ?
    """, (
        ctx.meta.get("modification_note"),
        ctx.req_id,
    ))
    conn.commit()


def get_previous_msg0(conn: sqlite3.Connection, session_fingerprint: str) -> tuple[str, str] | None:
    """Get (msg0_hash, msg0_content) of the latest request in a session by fingerprint."""
    row = conn.execute("""
        SELECT msg0_hash, msg0_content FROM requests
        WHERE session_fingerprint = ? AND msg0_hash IS NOT NULL
        ORDER BY timestamp DESC LIMIT 1
    """, (session_fingerprint,)).fetchone()
    if row and row["msg0_hash"]:
        return (row["msg0_hash"], row["msg0_content"])
    return None


def get_previous_msg0_by_uuid(conn: sqlite3.Connection, session_uuid: str) -> tuple[str, str] | None:
    """Get (msg0_hash, msg0_content) of the latest request in a session by UUID."""
    row = conn.execute("""
        SELECT msg0_hash, msg0_content FROM requests
        WHERE session_uuid = ? AND msg0_hash IS NOT NULL
        ORDER BY timestamp DESC LIMIT 1
    """, (session_uuid,)).fetchone()
    if row and row["msg0_hash"]:
        return (row["msg0_hash"], row["msg0_content"])
    return None


def get_previous_msg_chain_by_uuid(
    conn: sqlite3.Connection, session_uuid: str, exclude_req_id: str,
) -> tuple[str | None, int] | None:
    """Get (msg_hash_chain, prompt_messages_count) of latest request with same session_uuid."""
    row = conn.execute("""
        SELECT msg_hash_chain, prompt_messages_count FROM requests
        WHERE session_uuid = ? AND request_id != ?
          AND msg_hash_chain IS NOT NULL
        ORDER BY id DESC LIMIT 1
    """, (session_uuid, exclude_req_id)).fetchone()
    if row and row["msg_hash_chain"]:
        return (row["msg_hash_chain"], row["prompt_messages_count"])
    return None


def get_previous_msg_chain_by_fingerprint(
    conn: sqlite3.Connection, fingerprint: str, exclude_req_id: str,
) -> tuple[str | None, int] | None:
    """Get (msg_hash_chain, prompt_messages_count) of latest request with same fingerprint."""
    row = conn.execute("""
        SELECT msg_hash_chain, prompt_messages_count FROM requests
        WHERE session_fingerprint = ? AND request_id != ?
          AND msg_hash_chain IS NOT NULL
        ORDER BY id DESC LIMIT 1
    """, (fingerprint, exclude_req_id)).fetchone()
    if row and row["msg_hash_chain"]:
        return (row["msg_hash_chain"], row["prompt_messages_count"])
    return None


def get_previous_request_jsonl_path(
    conn: sqlite3.Connection, session_key: str, by_uuid: bool, exclude_req_id: str,
) -> tuple[str, str, int] | None:
    """Get (jsonl_path, request_id, jsonl_offset) of the latest request with matching session."""
    if by_uuid:
        row = conn.execute("""
            SELECT jsonl_path, request_id, jsonl_offset FROM requests
            WHERE session_uuid = ? AND request_id != ? AND jsonl_path != ''
            ORDER BY id DESC LIMIT 1
        """, (session_key, exclude_req_id)).fetchone()
    else:
        row = conn.execute("""
            SELECT jsonl_path, request_id, jsonl_offset FROM requests
            WHERE session_fingerprint = ? AND request_id != ? AND jsonl_path != ''
            ORDER BY id DESC LIMIT 1
        """, (session_key, exclude_req_id)).fetchone()
    return (row["jsonl_path"], row["request_id"], row["jsonl_offset"] or 0) if row else None
