"""Anthropic proxy handler -- pipeline orchestration only."""
from __future__ import annotations

import asyncio
import logging
from aiohttp import web

from deepseek_tap_proxy.pipeline import Context, Pipeline
from deepseek_tap_proxy.router import ResolvedTarget
from deepseek_tap_proxy.anthropic.stages import (
    AnthropicParseStage, AnthropicNormalizeStage, AnthropicBlock0ClearStage,
    AnthropicCanonicalUserStage, AnthropicBaselineStage,
    AnthropicSystemRoleConvertStage,
    AnthropicUpstreamPathStage, AnthropicForwardStage, ReturnStage,
)

log = logging.getLogger("deepseek-tap")


async def anthropic_proxy_handler(request: web.Request) -> web.StreamResponse:
    # Resolve upstream target via router (config-driven) or fallback to --target
    proxy_ctx = request.app.get("proxy_ctx", {})
    router = proxy_ctx.get("router")
    resolved = None
    if router is not None and router.is_available:
        resolved = router.resolve(
            request.headers.get("Authorization") or request.headers.get("x-api-key")
        )
    if resolved is None:
        resolved = ResolvedTarget(
            target_url=proxy_ctx.get("target_url", ""),
            upstream_name="default",
            matched_key_prefix=None,
        )
    request["upstream"] = resolved
    log.debug(
        f"[upstream] POST /v1/messages → "
        f"{resolved.upstream_name} "
        f"({'prefix ' + resolved.matched_key_prefix if resolved.matched_key_prefix else 'default'})"
    )

    ctx = Context(request=request, meta={})
    ctx.target_url = resolved.target_url
    ctx.meta["upstream_base_url"] = resolved.target_url
    pipeline = Pipeline([
        AnthropicParseStage(),
        AnthropicNormalizeStage(),
        AnthropicBlock0ClearStage(),
        AnthropicCanonicalUserStage(),
        AnthropicBaselineStage(),
        AnthropicSystemRoleConvertStage(),
        AnthropicUpstreamPathStage(),
        AnthropicForwardStage(),
        ReturnStage(),
    ])
    ctx = await pipeline.execute(ctx)

    # Fire-and-forget persist (tracked for completion during shutdown)
    db = request.app.get("db")
    proxy_ctx = request.app.get("proxy_ctx", {})
    writer = proxy_ctx.get("writer")
    persists: set[asyncio.Task] = proxy_ctx.setdefault("_pending_persists", set())
    if ctx.req_body is not None:
        from deepseek_tap_proxy.anthropic.persist import persist_anthropic
        t = asyncio.create_task(persist_anthropic(ctx, db, writer))
        persists.add(t)
        t.add_done_callback(persists.discard)

    return ctx.response or web.Response(status=502, text="No response")
