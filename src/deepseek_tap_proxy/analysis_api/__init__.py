"""Inspector 读取 proxy 产物的统一入口。纯文件读取，不碰 DB。

使用:
    from deepseek_tap_proxy.analysis_api import get_trace, list_requests, find_run, read_otlp

    run_dir = find_run("/path/to/data")
    traces = list_requests(run_dir)
    trace = get_trace(run_dir, "req_abc123")
    otlp = read_otlp(run_dir)
"""
from __future__ import annotations

import json
from pathlib import Path


def find_run(data_dir: str, run_id: str | None = None) -> str | None:
    """定位 run 目录。run_id 为 None 时返回最新的。

    Args:
        data_dir: proxy data 根目录 (含 runs/ 子目录)
        run_id: 指定 run ID，None 则返回最新

    Returns:
        run 目录的绝对路径，或 None
    """
    runs_dir = Path(data_dir) / "runs"
    if not runs_dir.exists():
        return None
    if run_id:
        candidate = runs_dir / run_id
        return str(candidate) if candidate.exists() else None
    dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return str(dirs[0]) if dirs else None


def get_trace(run_dir: str, req_id: str) -> dict | None:
    """从 traces.jsonl 读取单条 trace（按 request_id 匹配）。

    Args:
        run_dir: run 目录路径
        req_id: 请求 ID (如 "req_abc123")

    Returns:
        trace dict 或 None
    """
    for trace_file in _find_trace_files(run_dir):
        with open(trace_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("request_id") == req_id:
                    return record
    return None


def list_requests(run_dir: str) -> list[dict]:
    """按 timestamp 排序返回所有请求的 trace 记录（跨日期聚合）。

    Args:
        run_dir: run 目录路径

    Returns:
        trace dict 列表，按 timestamp 升序
    """
    records = []
    for trace_file in _find_trace_files(run_dir):
        with open(trace_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    records.sort(key=lambda r: r.get("timestamp", ""))
    return records


def read_otlp(run_dir: str) -> list[dict]:
    """读取 OTLP JSONL 文件（跨日期聚合）。

    Args:
        run_dir: run 目录路径

    Returns:
        OTLP 记录列表
    """
    records = []
    for otlp_file in _find_otlp_files(run_dir):
        with open(otlp_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def _find_trace_files(run_dir: str) -> list[Path]:
    """在 run_dir 下查找所有 traces.jsonl（支持跨日期聚合）。"""
    run = Path(run_dir)
    files = []
    if (run / "traces.jsonl").exists():
        files.append(run / "traces.jsonl")
    files.extend(sorted(run.glob("*/traces.jsonl")))
    return files


def _find_otlp_files(run_dir: str) -> list[Path]:
    """在 run_dir 下查找所有 otlp.jsonl（支持跨日期聚合）。"""
    run = Path(run_dir)
    files = []
    if (run / "otlp.jsonl").exists():
        files.append(run / "otlp.jsonl")
    files.extend(sorted(run.glob("*/otlp.jsonl")))
    return files
