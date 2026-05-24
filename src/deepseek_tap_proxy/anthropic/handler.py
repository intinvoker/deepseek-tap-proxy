"""Anthropic proxy handler -- pipeline orchestration only."""
from __future__ import annotations

import asyncio
import logging
from aiohttp import web

from deepseek_tap_proxy.pipeline import Context, Pipeline
from deepseek_tap_proxy.anthropic.stages import (
    AnthropicParseStage, AnthropicNormalizeStage, AnthropicBlock0ClearStage,
    AnthropicCanonicalUserStage, AnthropicBaselineStage,
    AnthropicForwardStage, ReturnStage,
)

log = logging.getLogger("deepseek-tap")


async def anthropic_proxy_handler(request: web.Request) -> web.StreamResponse:
    ctx = Context(request=request, meta={})
    pipeline = Pipeline([
        AnthropicParseStage(),
        AnthropicNormalizeStage(),
        AnthropicBlock0ClearStage(),
        AnthropicCanonicalUserStage(),
        AnthropicBaselineStage(),
        AnthropicForwardStage(),
        ReturnStage(),
    ])
    ctx = await pipeline.execute(ctx)

    # Fire-and-forget persist
    db = request.app.get("db")
    writer_ctx = request.app.get("proxy_ctx", {})
    writer = writer_ctx.get("writer")
    if ctx.req_body is not None:
        from deepseek_tap_proxy.anthropic.persist import persist_anthropic
        asyncio.create_task(persist_anthropic(ctx, db, writer))

    return ctx.response or web.Response(status=502, text="No response")
