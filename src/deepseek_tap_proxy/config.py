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
    lock_mode: str = "msg0"
    log_level: str = "INFO"

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
            "--lock-mode", choices=["msg0", "prefix", "none"], default="msg0",
        )
        parser.add_argument("--log-level", default="INFO")
        args = parser.parse_args(argv)
        cfg = cls(
            port=args.port, host=args.host, target=args.target,
            timeout=args.timeout, trace_dir=args.trace_dir,
            db_path=args.db_path, lock_mode=args.lock_mode,
            log_level=args.log_level,
        )
        cfg.trace_dir = str(Path(cfg.trace_dir).resolve())
        cfg.db_path = str(Path(cfg.db_path).resolve())
        return cfg
