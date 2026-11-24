"""只读分析入口 — 从 proxy 产物读取请求元数据（纯文件读取，不碰 DB）。

使用:
    from deepseek_tap_proxy.analysis_api import get_trace, list_requests, find_run, read_otlp

    run_dir = find_run("/path/to/data")
    traces = list_requests(run_dir)
    trace = get_trace(run_dir, "req_abc123")
    otlp = read_otlp(run_dir)
"""
from __future__ import annotations

import re
from pathlib import Path

try:
    import orjson as _fast_json
except ImportError:
    import json as _fast_json  # noqa: F811


# 轻量扫描：只提取 timestamp + request_id，不解析完整 JSON
_TS_RE = re.compile(rb'"timestamp"\s*:\s*"([^"]+)"')
_RID_RE = re.compile(rb'"request_id"\s*:\s*"([^"]+)"')


def _loads(data: str | bytes) -> dict:
    """JSON 解析：优先 orjson（2x faster），fallback stdlib json。"""
    if _fast_json.__name__ == "orjson":
        if isinstance(data, str):
            data = data.encode("utf-8")
    return _fast_json.loads(data)


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
    """从 traces.jsonl 读取单条 trace（按 request_id 匹配；含旋转分片）。"""
    for trace_file in _iter_trace_files_for_lookup(run_dir):
        with open(trace_file, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                record = _loads(line)
                if record.get("request_id") == req_id:
                    return record
    return None


def list_requests(run_dir: str) -> list[dict]:
    """按 timestamp 排序返回所有请求的 trace 记录（跨日期聚合）。

    注意：此函数会解析完整 JSON，大文件（1GB+）耗时数秒。
    若只需 request_id 列表，用 list_requests_light() 替代。
    """
    records = []
    for trace_file in _find_trace_files(run_dir):
        with open(trace_file, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(_loads(line))
    records.sort(key=lambda r: r.get("timestamp", ""))
    return records


def list_requests_light(run_dir: str) -> list[tuple[str, int]]:
    """返回 [(request_id, byte_offset), ...]，按 timestamp 排序。

    只正则提取 timestamp + request_id，不解析完整 JSON body。
    比 list_requests() 快 ~20x（大文件 6s → 0.3s），内存 ~7500x less。
    """
    entries: list[tuple[str, str, int]] = []  # (ts, rid, offset)
    for trace_file in _find_trace_files(run_dir):
        offset = 0
        with open(trace_file, "rb") as f:
            for line in f:
                if not line.strip():
                    offset = f.tell()
                    continue
                ts_m = _TS_RE.search(line)
                rid_m = _RID_RE.search(line)
                if ts_m and rid_m:
                    ts = ts_m.group(1).decode("ascii")
                    rid = rid_m.group(1).decode("ascii")
                    entries.append((ts, rid, offset))
                offset = f.tell()
    entries.sort(key=lambda x: x[0])  # sort by timestamp
    return [(rid, offset) for _, rid, offset in entries]


def read_trace_at(run_dir: str, byte_offset: int) -> dict | None:
    """在 traces.jsonl 的指定 byte_offset 处读取单条完整 trace。

    配合 list_requests_light() 使用：先收集 offset，再按需 seek+read。
    """
    for trace_file in _find_trace_files(run_dir):
        with open(trace_file, "rb") as f:
            f.seek(byte_offset)
            line = f.readline()
            if not line.strip():
                return None
            return _loads(line)
    return None


def read_otlp(run_dir: str) -> list[dict]:
    """读取 OTLP JSONL 文件（跨日期聚合）。"""
    records = []
    for otlp_file in _find_otlp_files(run_dir):
        with open(otlp_file, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(_loads(line))
    return records


def _find_trace_files(run_dir: str) -> list[Path]:
    """在 run_dir 下查找所有 trace 文件（支持新旧两种命名）。

    ⚠️ 共享扫描路径（``list_requests`` / ``list_requests_light`` /
    ``read_trace_at``）只认标准 + 旧版命名，**不含** TraceWriter 的旋转分片
    ``traces-<seq>.jsonl``：``list_requests_light`` 返回的 ``(rid, offset)``
    不携带文件身份，``read_trace_at`` 固定从首个文件 seek——把旋转分片塞进
    这里会让 replay 错位/静默漏数据。按 id 精确查找用
    ``_iter_trace_files_for_lookup``。
    """
    run = Path(run_dir)
    files = []
    # 标准命名: traces.jsonl
    if (run / "traces.jsonl").exists():
        files.append(run / "traces.jsonl")
    files.extend(sorted(run.glob("*/traces.jsonl")))
    # 旧版命名: trace_HHMMSS.jsonl
    files.extend(sorted(run.rglob("trace_*.jsonl")))
    return files


def _iter_trace_files_for_lookup(run_dir: str) -> list[Path]:
    """``get_trace`` 专用扫描顺序：共享路径 + TraceWriter 旋转分片（去重保序）。

    旋转分片（``traces-<seq>.jsonl``，按日目录内；根目录兼容）只在按 request_id
    精确查找时需要。去重保序：标准/旧命名在前（与共享路径一致），旋转分片追加，
    不与 ``traces.jsonl`` 重复（``traces-`` 前缀 glob 与 ``traces.jsonl`` 互斥）。
    """
    run = Path(run_dir)
    files = list(_find_trace_files(run_dir))
    seen = set(files)
    extras = [
        *sorted(run.glob("*/traces-*.jsonl")),
        *sorted(run.glob("traces-*.jsonl")),
    ]
    for path in extras:
        if path not in seen:
            files.append(path)
            seen.add(path)
    return files


def _find_otlp_files(run_dir: str) -> list[Path]:
    """在 run_dir 下查找所有 otlp.jsonl（支持跨日期聚合）。"""
    run = Path(run_dir)
    files = []
    if (run / "otlp.jsonl").exists():
        files.append(run / "otlp.jsonl")
    files.extend(sorted(run.glob("*/otlp.jsonl")))
    return files
