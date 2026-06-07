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
    trace_dir: str = "./data/traces"
    db_path: str = "./data/proxy.db"
    lock_mode: str = "prefix"
    strip_user_id: bool = True
    lock_sys_tools: bool = True
    log_level: str = "INFO"
    log_file: str = ""  # 日志文件路径，留空则自动设为 data/proxy.log
    upstreams_config: str = ""  # path to upstreams.yaml, empty = router disabled

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
        parser.add_argument("--trace-dir", default="./data/traces")
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
        args = parser.parse_args(argv)
        cfg = cls(
            port=args.port, host=args.host, target=args.target,
            timeout=args.timeout, trace_dir=args.trace_dir,
            db_path=args.db_path, lock_mode=args.lock_mode,
            strip_user_id=args.strip_user_id,
            lock_sys_tools=args.lock_sys_tools,
            log_level=args.log_level, log_file=args.log_file,
            upstreams_config=args.upstreams_config,
        )
        cfg.trace_dir = str(Path(cfg.trace_dir).resolve())
        cfg.db_path = str(Path(cfg.db_path).resolve())
        if not cfg.log_file:
            cfg.log_file = str(Path(cfg.db_path).parent / "log" / "proxy.log")
        return cfg
