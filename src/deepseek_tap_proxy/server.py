from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone

import aiohttp
from aiohttp import web

from deepseek_tap_proxy.config import ProxyConfig
from deepseek_tap_proxy.db import Database
from deepseek_tap_proxy.shared.headers import HOP_BY_HOP, filter_hop_by_hop
from deepseek_tap_proxy.shared.log import OtlpLogWriter, TraceWriter
from deepseek_tap_proxy.shared.store import MemoryStore
from deepseek_tap_proxy.router import UpstreamRouter
from deepseek_tap_proxy.openai.routes import setup_openai_routes
from deepseek_tap_proxy.anthropic.routes import setup_anthropic_routes
from deepseek_tap_proxy.shared.runtime import Runtime
from deepseek_tap_proxy.shared.transport import AiohttpTransport

log = logging.getLogger("deepseek-tap")


def _create_run_dir(trace_dir: str) -> tuple[str, str]:
    """创建 run 目录，返回 (run_id, run_dir_path).

    结构: {trace_dir}/{run_id}/
    run_id = HHMMSS_random8

    注：内层日期目录在写入第一条记录时动态创建，不在启动时预创建。
    """
    now = datetime.now(timezone.utc)
    time_str = now.strftime("%H%M%S")
    random_str = uuid.uuid4().hex[:8]
    run_id = f"{time_str}_{random_str}"
    run_dir = os.path.join(trace_dir, run_id)
    os.makedirs(run_dir, exist_ok=False)
    return run_id, run_dir


async def create_proxy_app(config: ProxyConfig, *, add_health: bool = True) -> web.Application:
    app = web.Application(client_max_size=64 * 1024**2)

    run_id, run_dir = _create_run_dir(config.trace_dir)
    writer = TraceWriter(run_dir, run_id)
    db = Database(config.db_path)
    db.init_db()
    # Close the main-thread connection so the fire-and-forget writer thread
    # (started by MemoryStore.__init__) does not hit "database is locked".
    db.close()

    # MemoryStore: load existing data from SQLite into in-memory store (0 I/O runtime)
    store = await MemoryStore.load_from(db, _db=db)

    client_session = aiohttp.ClientSession(
        headers={"User-Agent": "DeepSeekTap/0.1"},
    )

    # Load upstream router (config file driven, disabled if no config)
    router = UpstreamRouter(config.upstreams_config or None)

    # OTLP log writer — 默认写入 run_dir/{date}/otlp.jsonl
    # 如果 config.otlp_log_dir 已设置（测试用），使用该路径
    _otlp_base = config.otlp_log_dir or run_dir
    otlp_writer = OtlpLogWriter(_otlp_base)

    app["proxy_ctx"] = {
        "writer": writer,
        "session": client_session,
        "run_id": run_id,
        "run_dir": run_dir,
        "otlp_log_writer": otlp_writer,
    }
    app["db"] = db
    app["writer"] = writer

    # 🆕 构造 Runtime（强类型依赖注入）
    runtime = Runtime(
        store=store,
        config=config,
        transport=AiohttpTransport(client_session, config.timeout),
        session=client_session,
        router=router,
        run_id=run_id,
        otlp_writer=otlp_writer,
    )

    # 🆕 Middleware: 对每个请求注入 Runtime + upstream
    @web.middleware
    async def runtime_middleware(request: web.Request, handler):
        request["runtime"] = runtime
        # Resolve upstream for this request
        resolved = await _resolve_upstream_for_request(request)
        request["upstream"] = resolved
        return await handler(request)

    app.middlewares.append(runtime_middleware)

    setup_openai_routes(app)
    from deepseek_tap_proxy.anthropic.handler import setup_anthropic_pipelines
    setup_anthropic_pipelines(app)
    setup_anthropic_routes(app)

    # Wrap GET handlers with upstream resolution
    async def _proxy_get_with_upstream(request: web.Request) -> web.Response:
        upstream = await _resolve_upstream_for_request(request)
        request["upstream"] = upstream
        return await _proxy_get_handler(request)

    app.router.add_get("/v1/models", _proxy_get_with_upstream)
    app.router.add_get("/models", _proxy_get_with_upstream)
    app.router.add_get("/v1", _proxy_get_with_upstream)
    if add_health:
        app.router.add_get("/health", _health_handler)

    # Internal endpoint: drain MemoryStore write queue (used by tests)
    async def _flush_handler(request: web.Request) -> web.Response:
        if store:
            await store.flush()
        return web.Response(text="ok")

    app.router.add_post("/__flush", _flush_handler)

    async def _shutdown(app: web.Application) -> None:
        """Ordered shutdown: flush MemoryStore → SQLite, then close resources."""
        # Flush MemoryStore pending writes → SQLite (covers all handlers)
        # No more _pending_persists; all Phase 1/2 writes go through single-writer queue
        if store:
            await store.flush_pending()
        await _close_writer(writer)
        await _close_otlp_log(otlp_writer)
        await _close_session(client_session)
        await _close_db(db)

    app.on_shutdown.append(_shutdown)

    log.info(f"Trace file: {writer.path}")
    log.info(f"Database: {config.db_path}")
    log.info(f"Listening on {config.host}:{config.port}")
    log.info(f"Upstream: {config.target}")
    if router.is_available:
        log.info(f"Upstream router: {config.upstreams_config}")
    else:
        log.info("Upstream router: disabled (no config)")

    return app


async def _resolve_upstream_for_request(request: web.Request):
    """Resolve upstream for GET passthrough requests (no pipeline)."""
    from deepseek_tap_proxy.router import ResolvedTarget
    rt = request.get("runtime")
    router = rt.router if rt else None
    if router is not None and router.is_available:
        resolved = router.resolve(request.headers.get("Authorization"))
        if resolved is not None:
            return resolved
    return ResolvedTarget(
        target_url=getattr(rt.config, "target", "") if rt and rt.config else "",
        upstream_name="default",
        matched_key_prefix=None,
    )


async def _proxy_get_handler(request: web.Request) -> web.Response:
    rt = request.get("runtime")
    session: aiohttp.ClientSession = rt.session if rt else request.app.get("proxy_ctx", {}).get("session")

    # Read resolved upstream from request (set by wrapper)
    upstream = request.get("upstream")
    if upstream is not None:
        target = upstream.target_url
    else:
        target = getattr(rt.config, "target", "") if rt and rt.config else ""

    log.debug(f"[upstream] GET {request.path} → {upstream.upstream_name if upstream else 'default'}")

    fwd_headers = dict(request.headers)
    fwd_headers.pop("Host", None)
    fwd_headers = filter_hop_by_hop(fwd_headers)
    path_prefix = upstream.path_prefix if upstream is not None and upstream.path_prefix else ""
    if path_prefix and not request.path.startswith(path_prefix):
        upstream_url = target.rstrip("/") + path_prefix + request.path
    else:
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
    rt = request.get("runtime")
    store = rt.store if rt else request.app.get("store")
    result = {"status": "ok", "service": "proxy"}
    if store:
        stats = store.get_write_stats()
        result["write_stats"] = stats
        if stats["writes_failed"] > 10:
            result["status"] = "degraded"
        if stats["writes_attempted"] > 100 and stats["failure_rate"] > 0.05:
            result["status"] = "degraded"
    return web.json_response(result)


async def _close_writer(writer: TraceWriter) -> None:
    await writer.close()


async def _close_otlp_log(otlp_writer: OtlpLogWriter | None) -> None:
    if otlp_writer is not None:
        otlp_writer.close()


async def _close_session(session: aiohttp.ClientSession) -> None:
    await session.close()


async def _close_db(db) -> None:
    db.close()


