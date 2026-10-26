"""Tests for MemoryStore bounded queue + async backpressure + batch commit + failure replay."""

import asyncio
import sqlite3
import time

import pytest

from deepseek_tap_proxy.shared.db import (
    create_requests_table, create_sessions_table, create_prefix_chains_table,
)
from deepseek_tap_proxy.shared.store import MemoryStore


def _in_memory_db():
    # check_same_thread=False: 单连接需被 writer 线程（独立线程）共享使用。
    # 生产环境 Database.conn 是 per-thread 连接（file 型 DB 经 WAL 跨连接可见），
    # 而 :memory: 每个连接是独立库，必须共享同一连接才能看到 writer 提交的数据。
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    create_requests_table(conn)
    create_sessions_table(conn)
    create_prefix_chains_table(conn)
    return conn


class _MemDB:
    """Minimal db-like object exposing conn (mirrors Database)."""

    def __init__(self, conn):
        self._conn = conn

    @property
    def conn(self):
        return self._conn


@pytest.mark.asyncio
async def test_batch_commit_merges_multiple_writes():
    """多条入队在 writer 线程合并为单 commit（隐式事务累积）。"""
    db = _MemDB(_in_memory_db())
    store = MemoryStore(db=db, db_batch_max=10, db_commit_interval=0.05)
    calls = []

    def w(conn, v):
        calls.append(v)
        conn.execute(
            "INSERT INTO sessions (session_id, client_type, start_time, last_seen_time) VALUES (?, 't', 'x', 'x')",
            (v,),
        )

    for i in range(5):
        await store.enqueue_sqlite_write(w, f"s{i}", op_name="test")
    await store.flush()
    rows = db.conn.execute("SELECT session_id FROM sessions ORDER BY session_id").fetchall()
    assert [r[0] for r in rows] == ["s0", "s1", "s2", "s3", "s4"]
    assert len(calls) == 5
    await store.flush_pending()


@pytest.mark.asyncio
async def test_timeout_flush_single_record():
    """低流量：单条入队后由 writer 立即处理（get 到即落盘）。"""
    db = _MemDB(_in_memory_db())
    store = MemoryStore(db=db, db_batch_max=500, db_commit_interval=0.05)
    await store.enqueue_sqlite_write(
        lambda conn, v: conn.execute(
            "INSERT INTO sessions (session_id, client_type, start_time, last_seen_time) VALUES (?, 't', 'x', 'x')",
            (v,),
        ),
        "only", op_name="test")
    await asyncio.sleep(0.2)
    await store.flush()
    rows = db.conn.execute("SELECT session_id FROM sessions").fetchall()
    assert len(rows) == 1
    await store.flush_pending()


@pytest.mark.asyncio
async def test_backpressure_blocks_when_queue_full():
    """队列满时 enqueue 阻塞直到消费后有空间（不丢不抛）。"""
    db = _MemDB(_in_memory_db())
    store = MemoryStore(db=db, db_batch_max=1, db_commit_interval=0.01,
                        db_queue_max=2)
    for i in range(3):
        await store.enqueue_sqlite_write(
            lambda conn, v: conn.execute(
                "INSERT INTO sessions (session_id, client_type, start_time, last_seen_time) VALUES (?, 't', 'x', 'x')",
                (v,),
            ),
            f"b{i}", op_name="test")
    await asyncio.wait_for(
        store.enqueue_sqlite_write(
            lambda conn, v: conn.execute(
                "INSERT INTO sessions (session_id, client_type, start_time, last_seen_time) VALUES (?, 't', 'x', 'x')",
                (v,),
            ),
            "b3", op_name="test"),
        timeout=5.0)
    await store.flush()
    rows = db.conn.execute("SELECT session_id FROM sessions ORDER BY session_id").fetchall()
    assert len(rows) == 4
    await store.flush_pending()


@pytest.mark.asyncio
async def test_batch_failure_replays_one_by_one():
    """整批 commit 失败 → rollback → 逐条重放；坏条目计数，好条目落盘。"""
    db = _MemDB(_in_memory_db())
    store = MemoryStore(db=db, db_batch_max=10, db_commit_interval=0.05)
    good = []

    def bad_write(conn, v):
        conn.execute(
            "INSERT INTO sessions (session_id, client_type, start_time, last_seen_time) VALUES ('dup', 't', 'x', 'x')")
        raise sqlite3.IntegrityError("boom")

    def good_write(conn, v):
        good.append(v)

    await store.enqueue_sqlite_write(bad_write, "x", op_name="bad")
    await store.enqueue_sqlite_write(good_write, "g1", op_name="good")
    await store.flush()
    # 整批失败 → 逐条重放含重试 sleep（0.1s+0.5s），flush 的固定 sleep 可能不足；
    # 轮询等待重放完成（坏条目计数 + 好条目落盘），避免时序 flaky。
    deadline = time.monotonic() + 5.0
    stats = store.get_write_stats()
    while stats["writes_failed"] < 1 and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        stats = store.get_write_stats()
    assert stats["writes_failed"] == 1
    assert stats["writes_succeeded"] >= 1
    assert "g1" in good
    await store.flush_pending()
