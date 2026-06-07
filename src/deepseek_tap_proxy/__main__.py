from __future__ import annotations

import asyncio
import gzip
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from aiohttp import web

from deepseek_tap_proxy.config import ProxyConfig
from deepseek_tap_proxy.server import create_proxy_app


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


class StripColorFormatter(logging.Formatter):
    """Formatter that strips ANSI color codes (for file output)."""

    def format(self, record: logging.LogRecord) -> str:
        return _ANSI_RE.sub("", super().format(record))


class GzipRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler that gzips rolled-over files instead of deleting."""

    def doRollover(self) -> None:
        super().doRollover()
        # 压缩所有轮转出的未压缩文件
        for i in range(self.backupCount, 0, -1):
            sfn = self.namer(self.baseFilename, i)
            if os.path.exists(sfn) and not sfn.endswith(".gz"):
                gz_path = sfn + ".gz"
                try:
                    with open(sfn, "rb") as f_in:
                        with gzip.open(gz_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(sfn)
                except OSError:
                    pass  # 其他进程可能正在读，跳过

    @staticmethod
    def namer(base: str, suffix: int) -> str:
        return f"{base}.{suffix}"


def setup_logging(config: ProxyConfig) -> None:
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)

    # 清除已有 handler（防止重复注册）
    for h in root.handlers[:]:
        root.removeHandler(h)

    # stdout — 颜色由 log 消息内联 ANSI 码实现
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(log_fmt, datefmt=date_fmt))
    root.addHandler(stdout_handler)

    # 文件 — gzip 滚动，剥离 ANSI 颜色码
    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = GzipRotatingFileHandler(
            str(log_path), maxBytes=10 * 1024 * 1024, backupCount=0,
            encoding="utf-8",
        )
        fh.setFormatter(StripColorFormatter(log_fmt, datefmt=date_fmt))
        root.addHandler(fh)

    # 抑制 aiohttp.access 日志（已由合并行替代）
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)


async def async_main() -> None:
    config = ProxyConfig.from_args()
    setup_logging(config)

    app = await create_proxy_app(config)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, config.host, config.port)
    await site.start()

    log = logging.getLogger("deepseek-tap")
    log.info("Proxy session: %s", app.get("proxy_ctx", {}).get("session_id", ""))
    if config.log_file:
        log.info("Log file: %s", config.log_file)

    access_host = "localhost" if config.host in ("0.0.0.0", "127.0.0.1") else config.host
    print(f"Proxy: http://{access_host}:{config.port}")
    print("(Press CTRL+C to quit)", file=sys.stderr)

    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
