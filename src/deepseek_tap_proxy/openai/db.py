"""OpenAI-specific database functions.

Pure functions that take ``conn`` (sqlite3.Connection) as first parameter.
Format-specific fields are read from ``ctx.meta``.
"""

from __future__ import annotations

import sqlite3


# ── Query functions (operate on the unified requests table) ─────────


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


# Re-export from shared (canonical three-column format).
# Kept for backward compat — new code should import directly from shared.db.
from deepseek_tap_proxy.shared.db import get_session_chains  # noqa: F401
