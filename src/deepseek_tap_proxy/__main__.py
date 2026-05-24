from __future__ import annotations

import asyncio
import logging
import sys

from aiohttp import web

from deepseek_tap_proxy.config import ProxyConfig
from deepseek_tap_proxy.server import create_proxy_app


def setup_logging(config: ProxyConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


async def async_main() -> None:
    config = ProxyConfig.from_args()
    setup_logging(config)

    app = await create_proxy_app(config)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, config.host, config.port)
    await site.start()

    access_host = "localhost" if config.host in ("0.0.0.0", "127.0.0.1") else config.host
    print(f"Proxy: http://{access_host}:{config.port}")
    print("(Press CTRL+C to quit)")

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
