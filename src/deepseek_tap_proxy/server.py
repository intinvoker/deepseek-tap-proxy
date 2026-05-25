from __future__ import annotations

import asyncio
import logging
import uuid

import aiohttp
from aiohttp import web

from deepseek_tap_proxy.config import ProxyConfig
from deepseek_tap_proxy.db import Database
from deepseek_tap_proxy.shared.trace import TraceWriter
from deepseek_tap_proxy.shared.headers import HOP_BY_HOP
from deepseek_tap_proxy.openai.routes import setup_openai_routes
from deepseek_tap_proxy.anthropic.routes import setup_anthropic_routes

SESSION_ID_LENGTH = 12

log = logging.getLogger("deepseek-tap")


async def create_proxy_app(config: ProxyConfig, *, add_health: bool = True) -> web.Application:
    app = web.Application(client_max_size=10 * 1024**2)

    session_id = f"run_{uuid.uuid4().hex[:SESSION_ID_LENGTH]}"
    writer = TraceWriter(config.trace_dir, session_id)
    db = Database(config.db_path)
    db.init_db()
    client_session = aiohttp.ClientSession(
        headers={"User-Agent": "DeepSeekTap/0.1"},
    )

    app["proxy_ctx"] = {
        "writer": writer,
        "session": client_session,
        "target_url": config.target,
        "timeout": config.timeout,
        "session_id": session_id,
    }
    app["db"] = db
    app["db_conn"] = db.conn
    app["config"] = config
    app["writer"] = writer

    setup_openai_routes(app)
    setup_anthropic_routes(app)

    app.router.add_get("/v1/models", _proxy_get_handler)
    app.router.add_get("/models", _proxy_get_handler)
    app.router.add_get("/v1", _proxy_get_handler)
    if add_health:
        app.router.add_get("/health", _health_handler)

    app.on_shutdown.append(lambda _: asyncio.gather(
        _close_writer(writer), _close_session(client_session),
        _wait_persists(app),
        return_exceptions=True,
    ))

    log.info(f"Proxy session: {session_id}")
    log.info(f"Trace file: {writer.path}")
    log.info(f"Database: {config.db_path}")
    log.info(f"Listening on {config.host}:{config.port}")
    log.info(f"Upstream: {config.target}")

    return app


async def _proxy_get_handler(request: web.Request) -> web.Response:
    proxy_ctx = request.app["proxy_ctx"]
    session: aiohttp.ClientSession = proxy_ctx["session"]
    target: str = proxy_ctx["target_url"]

    fwd_headers = dict(request.headers)
    fwd_headers.pop("Host", None)
    upstream_url = target.rstrip("/") + request.path

    try:
        upstream_resp = await session.request(
            method="GET", url=upstream_url, headers=fwd_headers,
            timeout=aiohttp.ClientTimeout(total=30),
        )
    except Exception as exc:
        log.error(f"[proxy-get] upstream error for {request.path}: {exc}")
        return web.Response(status=502, text=str(exc))

    body = await upstream_resp.read()
    return web.Response(
        status=upstream_resp.status,
        headers={
            k: v for k, v in upstream_resp.headers.items()
            if k.lower() not in HOP_BY_HOP
        },
        body=body,
    )


async def _health_handler(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "service": "proxy"})


async def _close_writer(writer: TraceWriter) -> None:
    writer.close()


async def _close_session(session: aiohttp.ClientSession) -> None:
    await session.close()


async def _wait_persists(app: web.Application) -> None:
    """Await any pending persist tasks before shutdown."""
    proxy_ctx = app.get("proxy_ctx", {})
    persists: set[asyncio.Task] = proxy_ctx.get("_pending_persists", set())
    if persists:
        await asyncio.gather(*persists, return_exceptions=True)
