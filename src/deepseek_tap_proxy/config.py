from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProxyConfig:
    port: int = 11808
    host: str = "0.0.0.0"
    target: str = "https://api.deepseek.com"
    timeout: int = 600
    trace_dir: str = "./data/runs"
    db_path: str = "./data/proxy.db"
    lock_mode: str = "prefix"
    strip_user_id: bool = True
    lock_sys_tools: bool = True
    log_level: str = "DEBUG"
    log_file: str = ""  # 日志文件路径，留空则自动设为 data/proxy.log
    upstreams_config: str = ""  # path to upstreams.yaml, empty = router disabled
    otlp_log_dir: str = ""  # 废弃，由 server.py 从 trace_dir 派生
    enable_governance: bool = True  # 内存治理（TTL + LRU 淘汰），Inspector 回放时关闭
    # ── I/O 治理（回灌 ai-server 经验）──
    trace_flush_interval: float = 2.0      # TraceWriter 缓冲 flush 间隔（秒）
    trace_batch_max: int = 500             # TraceWriter 批量 flush 阈值（条）
    trace_file_max_bytes: int = 500 * 1024 * 1024  # TraceWriter 单文件上限（500MB）
    db_commit_interval: float = 1.0        # MemoryStore 批量 commit 间隔（秒）
    db_batch_max: int = 500                # MemoryStore 批量 commit 阈值（条）
    db_queue_max: int = 1000               # MemoryStore 有界队列上限（条）

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> "ProxyConfig":
        parser = argparse.ArgumentParser(description="DeepSeek Tap Proxy")
        parser.add_argument("--port", type=int, default=11808)
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument(
            "--target",
            default=os.getenv("PROXY_TARGET", "https://api.deepseek.com"),
        )
        parser.add_argument("--timeout", type=int, default=600)
        parser.add_argument("--trace-dir", default="./data/runs")
        parser.add_argument("--db-path", default="./data/proxy.db")
        parser.add_argument(
            "--lock-mode", choices=["prefix", "none"], default="prefix",
            help="Lock mode: prefix (prefix chain lock) or none (disable lock)",
        )
        parser.add_argument(
            "--strip-user-id", action="store_true", default=True,
            help="Strip metadata.user_id from Anthropic requests (default: true)",
        )
        parser.add_argument(
            "--no-strip-user-id", action="store_false", dest="strip_user_id",
            help="Preserve metadata.user_id in Anthropic requests",
        )
        parser.add_argument(
            "--lock-sys-tools", action="store_true", default=True,
            help="Lock sys+tools to canonical values within session (default: true)",
        )
        parser.add_argument(
            "--no-lock-sys-tools", action="store_false", dest="lock_sys_tools",
            help="Disable sys+tools canonical locking",
        )
        parser.add_argument("--log-level", default="INFO")
        parser.add_argument(
            "--log-file", default="",
            help="Log file path (default: auto under trace-dir parent)",
        )
        parser.add_argument(
            "--upstreams-config", default="",
            help="Path to upstreams.yaml (multi-upstream routing config)",
        )
        parser.add_argument(
            "--otlp-log-dir", default="",
            help="OTLP JSONL log directory (AI-consumable structured logs, disabled if empty)",
        )
        parser.add_argument(
            "--no-governance", action="store_false", dest="enable_governance",
            help="Disable memory governance (TTL + LRU eviction). "
                 "Use for Inspector replay or debugging.",
        )
        # ── I/O 治理参数 ──
        parser.add_argument("--trace-flush-interval", type=float, default=2.0,
                            help="TraceWriter buffer flush interval in seconds")
        parser.add_argument("--trace-batch-max", type=int, default=500,
                            help="TraceWriter batch flush threshold (records)")
        parser.add_argument("--trace-file-max-bytes", type=int, default=500 * 1024 * 1024,
                            help="TraceWriter single-file size limit before rotation")
        parser.add_argument("--db-commit-interval", type=float, default=1.0,
                            help="MemoryStore batch commit interval in seconds")
        parser.add_argument("--db-batch-max", type=int, default=500,
                            help="MemoryStore batch commit threshold (records)")
        parser.add_argument("--db-queue-max", type=int, default=1000,
                            help="MemoryStore bounded write queue size")
        args = parser.parse_args(argv)
        cfg = cls(
            port=args.port, host=args.host, target=args.target,
            timeout=args.timeout, trace_dir=args.trace_dir,
            db_path=args.db_path, lock_mode=args.lock_mode,
            strip_user_id=args.strip_user_id,
            lock_sys_tools=args.lock_sys_tools,
            log_level=args.log_level, log_file=args.log_file,
            upstreams_config=args.upstreams_config,
            otlp_log_dir=args.otlp_log_dir,
            enable_governance=args.enable_governance,
            trace_flush_interval=args.trace_flush_interval,
            trace_batch_max=args.trace_batch_max,
            trace_file_max_bytes=args.trace_file_max_bytes,
            db_commit_interval=args.db_commit_interval,
            db_batch_max=args.db_batch_max,
            db_queue_max=args.db_queue_max,
        )
        cfg.trace_dir = str(Path(cfg.trace_dir).resolve())
        cfg.db_path = str(Path(cfg.db_path).resolve())
        if not cfg.log_file:
            cfg.log_file = str(Path(cfg.db_path).parent / "log" / "proxy.log")
        return cfg
