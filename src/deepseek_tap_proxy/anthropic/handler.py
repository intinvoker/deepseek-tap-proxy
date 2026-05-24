from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from aiohttp import web

from deepseek_tap_proxy.pipeline import Context, Pipeline
from deepseek_tap_proxy.shared.headers import redact
from deepseek_tap_proxy.anthropic.stages import (
    AnthropicParseStage, AnthropicForwardStage, ReturnStage,
)

log = logging.getLogger("deepseek-tap")


def _insert_anthropic_trace_safe(db, **kwargs) -> None:
    """Thread-safe wrapper: gets conn inside the target thread."""
    from deepseek_tap_proxy.anthropic.db import insert_anthropic_trace
    conn = db.conn if hasattr(db, 'conn') else db
    insert_anthropic_trace(conn, **kwargs)


async def anthropic_proxy_handler(request: web.Request) -> web.StreamResponse:
    ctx = Context(request=request, meta={})
    pipeline = Pipeline([AnthropicParseStage(), AnthropicForwardStage(), ReturnStage()])
    ctx = await pipeline.execute(ctx)

    db = request.app.get("db")
    writer_ctx = request.app.get("proxy_ctx", {})
    writer = writer_ctx.get("writer")
    if writer:
        asyncio.create_task(_persist_anthropic(ctx, request, db, writer))

    return ctx.response or web.Response(status=502, text="No response")


async def _persist_anthropic(ctx: Context, request: web.Request, db, writer) -> None:
    try:
        usage = ctx.meta.get("usage", {})
        proxy_ctx = request.app.get("proxy_ctx", {})
        target = proxy_ctx.get("target_url", "")

        path_suffix = request.path
        if path_suffix.startswith("/anthropic"):
            path_suffix = path_suffix[len("/anthropic"):]

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": ctx.req_id,
            "duration_ms": ctx.duration_ms,
            "upstream_base_url": target + "/anthropic",
            "api_format": "anthropic",
            "request": {
                "method": request.method, "path": path_suffix or "/",
                "headers": redact(dict(request.headers)), "body": ctx.req_body,
            },
            "response": {
                "status": ctx.status, "headers": redact(ctx.resp_headers),
                "body": ctx.resp_body,
            },
        }
        if usage:
            record["usage"] = usage
        if "sse_events" in ctx.meta:
            record["response"]["sse_events"] = ctx.meta["sse_events"]

        await writer.write(record)

        if db:
            from deepseek_tap_proxy.anthropic.db import insert_anthropic_trace
            sid = request.headers.get("X-Claude-Code-Session-Id", "") or \
                  request.headers.get("x-claude-code-session-id", "")
            model = (ctx.req_body or {}).get("model", "")
            await asyncio.to_thread(
                _insert_anthropic_trace_safe, db,
                req_id=ctx.req_id, timestamp=record["timestamp"],
                duration_ms=ctx.duration_ms, model=model,
                status_code=ctx.status, stream_mode=ctx.is_streaming,
                jsonl_path=str(writer.path), turn=0,
                session_id=sid or None,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
            )
    except Exception:
        log.exception(f"[{ctx.req_id}] anthropic persist failed")
