"""Tests for db.py write functions' ``commit`` parameter.

The 4 write functions gain a ``commit: bool = True`` parameter so a batch
writer can accumulate changes in one transaction and commit once. Default
``commit=True`` keeps existing per-call semantics unchanged.
"""

import sqlite3

from deepseek_tap_proxy.shared.db import (
    create_requests_table, create_sessions_table, create_prefix_chains_table,
    insert_request_phase1, update_request_phase2,
    upsert_prefix_chain, set_session_canonical,
)


def _conn():
    conn = sqlite3.connect(":memory:")
    create_requests_table(conn)
    create_sessions_table(conn)
    create_prefix_chains_table(conn)
    return conn


class FakePersist:
    request_id = "req_1"
    api_format = "openai"
    trace_category = "chat"
    chat_type = "normal"
    prompt_messages_count = 1
    prompt_tools_count = 2
    session_id = "sess_1"
    client_type = "generic"
    msg_hash_chain = "abc"
    total_input_tokens = 10
    output_tokens = 5
    cache_hit_tokens = 8
    cache_miss_tokens = 2
    cache_creation_tokens = 0
    cache_hit_ratio = 0.8
    has_reasoning = False
    has_tool_calls = True
    lock_type = "prefix"
    prefix_chain_id = "chain_1"
    is_modified = False
    modification_note = ""
    system_role_converted = False
    sys_tools_locked = "lock"


class FakeCtx:
    req_id = "req_1"
    is_streaming = False
    body_readonly = {"model": "m"}
    persist = FakePersist()
    meta = {"jsonl_path": "/tmp/t.jsonl"}
    duration_ms = 100
    elapsed_ms = 100
    status = 200


def test_insert_no_commit_when_disabled():
    """commit=False 时 insert 不应提交（隐式事务累积）。"""
    conn = _conn()
    ctx = FakeCtx()
    insert_request_phase1(conn, ctx, commit=False)
    conn.rollback()
    row = conn.execute("SELECT 1 FROM requests WHERE request_id='req_1'").fetchone()
    assert row is None  # 已回滚


def test_upsert_chain_commit_param():
    conn = _conn()
    upsert_prefix_chain(conn, "s1", "c1", sys="[]", tools="[]", messages="[]",
                        sys_hashes="a", tools_hashes="b", messages_hashes="c",
                        msg_count=1, commit=False)
    conn.rollback()
    row = conn.execute("SELECT 1 FROM prefix_chains WHERE chain_id='c1'").fetchone()
    assert row is None


def test_update_no_commit_when_disabled():
    """commit=False 时 update 不提交；已提交的 insert 行不受回滚影响。"""
    conn = _conn()
    ctx = FakeCtx()
    insert_request_phase1(conn, ctx)  # 默认 commit=True，行已持久化
    update_request_phase2(conn, ctx, commit=False)  # 未提交
    conn.rollback()
    row = conn.execute(
        "SELECT duration_ms, status_code FROM requests WHERE request_id='req_1'"
    ).fetchone()
    assert row == (0, 0)  # update 被回滚，duration_ms/status_code 仍为初始 0


def test_set_canonical_no_commit_when_disabled():
    """commit=False 时 set_session_canonical 不提交（回滚后 canonical 未更新）。"""
    conn = _conn()
    conn.execute(
        "INSERT INTO sessions (session_id, client_type, start_time, last_seen_time) "
        "VALUES ('s1', 'generic', '2026-01-01T00:00:00', '2026-01-01T00:00:00')"
    )
    conn.commit()
    set_session_canonical(conn, "s1", sys="x", commit=False)
    conn.rollback()
    row = conn.execute(
        "SELECT canonical_sys FROM sessions WHERE session_id='s1'"
    ).fetchone()
    assert row is not None  # 已提交的 insert 行仍在
    assert row[0] == ""  # canonical_sys 未更新（回滚）
