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
    db_type: str = "sqlite"  # sqlite | postgres，PG 仅 docker 部署消费
    db_dsn: str = ""  # postgres 连接 DSN
    lock_mode: str = "prefix"
    strip_user_id: bool = True
    lock_sys_tools: bool = True
    log_level: str = "INFO"
    log_file: str = ""  # 日志文件路径，留空则自动设为 data/proxy.log
    upstreams_config: str = ""  # path to upstreams.yaml, empty = router disabled
    policies_config: str = ""   # path to policies.yaml (mode policy), empty = 内置默认
    otlp_log_dir: str = ""  # 废弃，由 server.py 从 trace_dir 派生
    enable_governance: bool = True  # 内存治理（TTL + LRU 淘汰），调试回放时关闭
    # ── I/O 治理（批提交 / 背压 / 有界队列）──
    trace_flush_interval: float = 2.0      # TraceWriter 缓冲 flush 间隔（秒）
    trace_batch_max: int = 500             # TraceWriter 批量 flush 阈值（条）
    trace_record_max_bytes: int = 10 * 1024 * 1024  # resp_body 写入侧截断阈值（10MB，单请求内存峰值上限）
    trace_file_max_bytes: int = 500 * 1024 * 1024  # TraceWriter 单文件上限（500MB）
    db_commit_interval: float = 1.0        # MemoryStore 批量 commit 间隔（秒）
    db_batch_max: int = 500                # MemoryStore 批量 commit 阈值（条）
    db_queue_max: int = 1000               # MemoryStore 有界队列上限（条）
    # ── 事件日志权威化（Garnet AOF 理念）──
    events_path: str = "./data/chain-events.jsonl"  # 事件日志路径（跨重启稳定）
    checkpoint_interval: float = 300.0              # SQLite 投影重建周期（秒，checkpoint 即 AOF 压缩点）
    event_file_max_bytes: int = 50 * 1024 * 1024    # 事件日志单文件轮转阈值
    # ── 内存治理三参数（R3）──
    max_memory_bytes: int = 200 * 1024 * 1024       # 内存治理上限字节（默认 200MB）
    ttl_hours: int = 168                            # session TTL 小时（默认 168）
    max_moments: int = 100                          # chain moments 上限（默认 100）

    def __post_init__(self) -> None:
        """fail-fast：内存治理三参数非法值直接抛错（对齐 db_type 校验模式）。"""
        if self.max_memory_bytes <= 0:
            raise ValueError("max_memory_bytes 必须 > 0")
        if self.ttl_hours <= 0:
            raise ValueError("ttl_hours 必须 > 0")
        if self.max_moments <= 0:
            raise ValueError("max_moments 必须 > 0")

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
            "--db-type", choices=["sqlite", "postgres"],
            default=os.getenv("DEEPSEEK_TAP_DB_TYPE", "sqlite"),
        )
        parser.add_argument(
            "--db-dsn", default=os.getenv("DEEPSEEK_TAP_DB_DSN", ""),
        )
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
        parser.add_argument(
            "--enforce-client-detection", action="store_true", default=False,
            help="Reject unrecognized clients with 422. Default: off — "
                 "requests without a known client identity (e.g. probe pings "
                 "without session id) pass through as generic",
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
            "--policies-config",
            default=os.getenv("DEEPSEEK_TAP_POLICIES_CONFIG", ""),
            help="Path to policies.yaml (passthrough/adapter/managed mode policy). "
                 "Empty = built-in defaults (unknown clients → passthrough).",
        )
        parser.add_argument(
            "--otlp-log-dir", default="",
            help="OTLP JSONL log directory (AI-consumable structured logs, disabled if empty)",
        )
        parser.add_argument(
            "--no-governance", action="store_false", dest="enable_governance",
            help="Disable memory governance (TTL + LRU eviction). "
                 "Use for replay debugging.",
        )
        # ── I/O 治理参数 ──
        parser.add_argument("--trace-flush-interval", type=float, default=2.0,
                            help="TraceWriter buffer flush interval in seconds")
        parser.add_argument("--trace-batch-max", type=int, default=500,
                            help="TraceWriter batch flush threshold (records)")
        parser.add_argument("--trace-file-max-bytes", type=int, default=500 * 1024 * 1024,
                            help="TraceWriter single-file size limit before rotation")
        parser.add_argument("--trace-record-max-bytes", type=int, default=10 * 1024 * 1024,
                            help="resp_body 写入侧截断阈值（单请求内存峰值上限，0=不截断）")
        parser.add_argument("--db-commit-interval", type=float, default=1.0,
                            help="MemoryStore batch commit interval in seconds")
        parser.add_argument("--db-batch-max", type=int, default=500,
                            help="MemoryStore batch commit threshold (records)")
        parser.add_argument("--db-queue-max", type=int, default=1000,
                            help="MemoryStore bounded write queue size")
        parser.add_argument("--events-path", default="./data/chain-events.jsonl",
                            help="Chain state event log path (authoritative source)")
        parser.add_argument("--checkpoint-interval", type=float, default=300.0,
                            help="SQLite projection rebuild interval in seconds (checkpoint = AOF compaction point)")
        parser.add_argument("--event-file-max-bytes", type=int, default=50 * 1024 * 1024,
                            help="Event log single-file size before rotation")
        parser.add_argument("--max-memory-bytes", type=int,
                            default=os.getenv("DEEPSEEK_TAP_MAX_MEMORY_BYTES", 200 * 1024 * 1024),
                            help="内存治理上限字节（默认 200MB）")
        parser.add_argument("--ttl-hours", type=int,
                            default=os.getenv("DEEPSEEK_TAP_TTL_HOURS", 168),
                            help="session TTL 小时（默认 168）")
        parser.add_argument("--max-moments", type=int,
                            default=os.getenv("DEEPSEEK_TAP_MAX_MOMENTS", 100),
                            help="chain moments 上限（默认 100）")
        args = parser.parse_args(argv)
        cfg = cls(
            port=args.port, host=args.host, target=args.target,
            timeout=args.timeout, trace_dir=args.trace_dir,
            db_path=args.db_path, db_type=args.db_type, db_dsn=args.db_dsn,
            lock_mode=args.lock_mode,
            strip_user_id=args.strip_user_id,
            lock_sys_tools=args.lock_sys_tools,
            log_level=args.log_level, log_file=args.log_file,
            upstreams_config=args.upstreams_config,
            policies_config=args.policies_config,
            otlp_log_dir=args.otlp_log_dir,
            enable_governance=args.enable_governance,
            trace_flush_interval=args.trace_flush_interval,
            trace_batch_max=args.trace_batch_max,
            trace_record_max_bytes=args.trace_record_max_bytes,
            trace_file_max_bytes=args.trace_file_max_bytes,
            db_commit_interval=args.db_commit_interval,
            db_batch_max=args.db_batch_max,
            db_queue_max=args.db_queue_max,
            events_path=args.events_path,
            checkpoint_interval=args.checkpoint_interval,
            event_file_max_bytes=args.event_file_max_bytes,
            max_memory_bytes=args.max_memory_bytes,
            ttl_hours=args.ttl_hours,
            max_moments=args.max_moments,
        )
        cfg.trace_dir = str(Path(cfg.trace_dir).resolve())
        cfg.db_path = str(Path(cfg.db_path).resolve())
        cfg.events_path = str(Path(cfg.events_path).resolve())
        if cfg.db_type == "postgres" and not cfg.db_dsn:
            raise ValueError("--db-type postgres 需要提供 --db-dsn（或 env DEEPSEEK_TAP_DB_DSN）")
        if cfg.max_memory_bytes <= 0:
            raise ValueError("max_memory_bytes 必须 > 0")
        if cfg.ttl_hours <= 0:
            raise ValueError("ttl_hours 必须 > 0")
        if cfg.max_moments <= 0:
            raise ValueError("max_moments 必须 > 0")
        if not cfg.log_file:
            cfg.log_file = str(Path(cfg.db_path).parent / "log" / "proxy.log")
        return cfg