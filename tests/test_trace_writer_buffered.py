import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from deepseek_tap_proxy.shared.log import TraceWriter


def make_record(i):
    return {
        "request_id": f"req_{i}",
        "timestamp": "2026-07-31T00:00:00Z",
        "api_format": "openai",
        "usage": {"total_input_tokens": 1, "output_tokens": 1},
        "request": {"body": {"model": "m"}},
    }


@pytest.mark.asyncio
async def test_no_flush_until_batch_threshold(tmp_path):
    w = TraceWriter(str(tmp_path), "run1", batch_max=3, flush_interval=10.0)
    await w.write(make_record(0))
    await w.write(make_record(1))
    assert w._pending_count == 2
    assert w._pending_count < w._batch_max
    await w.write(make_record(2))
    assert w._pending_count == 0  # 达阈值已 flush
    await w.close()


@pytest.mark.asyncio
async def test_flush_on_interval(tmp_path):
    w = TraceWriter(str(tmp_path), "run1", batch_max=500, flush_interval=0.1)
    await w.start_flusher()
    await w.write(make_record(0))
    assert w._pending_count == 1
    await asyncio.sleep(0.25)  # 超过间隔
    assert w._pending_count == 0  # flusher 已 flush
    await w.close()


@pytest.mark.asyncio
async def test_size_rotation(tmp_path):
    w = TraceWriter(str(tmp_path), "run1", batch_max=500, flush_interval=60.0,
                    file_max_bytes=50)
    for i in range(3):
        await w.write(make_record(i))
    files = sorted(p.name for p in Path(w.run_dir).glob("**/traces*.jsonl"))
    assert len(files) >= 2  # traces.jsonl + traces-2.jsonl
    await w.close()


@pytest.mark.asyncio
async def test_close_flushes_pending(tmp_path):
    w = TraceWriter(str(tmp_path), "run1", batch_max=500, flush_interval=60.0)
    await w.write(make_record(0))
    assert w._pending_count == 1
    await w.close()
    assert w._pending_count == 0


@pytest.mark.asyncio
async def test_flush_writes_real_content(tmp_path):
    """flush 后真实落盘：读回 JSONL，行数与内容正确（核心契约）。"""
    w = TraceWriter(str(tmp_path), "run1", batch_max=3, flush_interval=60.0)
    await w.write(make_record(1))
    await w.write(make_record(2))
    # 未达阈值也未到间隔 → 文件内此时应无内容（缓冲中）
    # 达阈值触发 flush
    await w.write(make_record(3))
    path = w.path
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines, start=1):
        rec = json.loads(line)
        assert rec["request_id"] == f"req_{i}"
    await w.close()
