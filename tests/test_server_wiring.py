"""Tests for server.py I/O governance wiring.

任务 5：create_proxy_app 需把 ProxyConfig 的 I/O 治理参数真正装配到
TraceWriter / MemoryStore，并启动/停止 flusher。
"""

import pytest
from aiohttp.test_utils import TestClient, TestServer

from deepseek_tap_proxy.config import ProxyConfig
from deepseek_tap_proxy.server import create_proxy_app


@pytest.mark.asyncio
async def test_app_wires_io_governance_params(tmp_path):
    cfg = ProxyConfig(
        trace_dir=str(tmp_path / "runs"),
        db_path=str(tmp_path / "proxy.db"),
        otlp_log_dir=str(tmp_path / "otlp"),
        trace_flush_interval=1.0, trace_batch_max=10,
        trace_file_max_bytes=1024 * 1024,
        db_commit_interval=1.0, db_batch_max=10, db_queue_max=100,
    )
    app = await create_proxy_app(cfg)
    writer = app["proxy_ctx"]["writer"]
    assert writer._flush_interval == 1.0
    assert writer._batch_max == 10
    assert writer._file_max_bytes == 1024 * 1024
    assert writer._flusher_task is not None  # create 时已启动

    client = TestClient(TestServer(app))
    await client.start_server()
    resp = await client.get("/health")
    assert resp.status == 200
    data = await resp.json()
    assert "write_stats" in data
    assert "batches_committed" in data["write_stats"]
    assert data["write_stats"]["queue_max"] == 100  # MemoryStore 有界队列上限已装配
    await client.close()
    await app.shutdown()
