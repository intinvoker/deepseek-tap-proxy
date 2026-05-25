from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from aiohttp import web

from deepseek_tap_proxy.pipeline import Context, Pipeline
from deepseek_tap_proxy.openai.stages import (
    ParseStage, Msg0LockStage, PrefixLockStage, ForwardStage, ReturnStage,
)
from deepseek_tap_proxy.openai.db import (
    insert_request_phase1, update_request_phase2, update_modification_flag,
)
from deepseek_tap_proxy.shared.headers import redact
from deepseek_tap_proxy.shared.trace import TraceWriter

log: logging.Logger = logging.getLogger("deepseek-tap")


async def chat_completions_handler(request: web.Request) -> web.StreamResponse:
    """Handle /chat/completions via the request pipeline."""
    ctx = Context(request=request)

    config = request.app.get("config", {})
    lock_mode = getattr(config, "lock_mode", "msg0")
    msg0_enabled: bool = lock_mode in ("msg0", "prefix")
    prefix_enabled: bool = lock_mode == "prefix"

    pipeline: Pipeline = _build_pipeline(msg0_enabled, prefix_enabled)
    ctx = await pipeline.execute(ctx)

    # Background persist (tracked for completion during shutdown)
    db = request.app.get("db")
    proxy_ctx = request.app.get("proxy_ctx", {})
    writer: TraceWriter | None = proxy_ctx.get("writer")
    persists: set[asyncio.Task] = proxy_ctx.setdefault("_pending_persists", set())
    if db:
        t1 = asyncio.create_task(_persist_phase1(ctx, db))
        persists.add(t1)
        t1.add_done_callback(persists.discard)
    if db and writer:
        t2 = asyncio.create_task(_persist_phase2(ctx, db, writer))
        persists.add(t2)
        t2.add_done_callback(persists.discard)

    if ctx.response is not None:
        return ctx.response
    return web.Response(status=502, text="No response from pipeline")


def _build_pipeline(msg0_enabled: bool = True, prefix_enabled: bool = False) -> Pipeline:
    """Build the default pipeline with optional lock stages.

    Pipeline: ParseStage -> Msg0LockStage(opt) -> PrefixLockStage(opt) -> ForwardStage -> ReturnStage
    """
    stages = [
        ParseStage(),
        ForwardStage(),
        ReturnStage(),
    ]
    if msg0_enabled:
        stages.insert(1, Msg0LockStage())
    if prefix_enabled:
        insert_pos = 1
        if msg0_enabled:
            insert_pos = 2
        stages.insert(insert_pos, PrefixLockStage())
    return Pipeline(stages)


def _build_trace_record(ctx: Context) -> dict:
    """Build a full trace record from pipeline context.

    When a lock (msg0/prefix) modified the request, the trace stores what was
    actually sent upstream (parsed from raw_body), so downstream readers like
    PrefixLock see the locked version, not the tampered original.
    """
    trace_body = ctx.req_body
    if ctx.meta.get("modified"):
        try:
            trace_body = json.loads(ctx.raw_body)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass  # fall back to req_body

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": ctx.req_id,
        "duration_ms": ctx.duration_ms or ctx.elapsed_ms,
        "upstream_base_url": ctx.target_url,
        "request": {
            "method": "POST",
            "path": "/chat/completions",
            "headers": redact(dict(ctx.request.headers)),
            "body": trace_body,
        },
        "response": {
            "status": ctx.status,
            "headers": redact(ctx.resp_headers),
            "body": ctx.resp_body,
        },
        "cache_analysis": {
            "cache_hit_tokens": ctx.meta.get("cache_hit_tokens", 0),
            "cache_miss_tokens": ctx.meta.get("cache_miss_tokens", 0),
            "cache_hit_ratio": round(ctx.meta.get("cache_hit_ratio", 0.0), 4),
            "prompt_total_tokens": ctx.meta.get("prompt_tokens", 0),
            "output_tokens": ctx.meta.get("completion_tokens", 0),
            "has_reasoning_content": ctx.meta.get("has_reasoning", False),
            "has_tool_calls": ctx.meta.get("has_tool_calls", False),
            "prompt_messages_count": ctx.meta.get("prompt_messages_count", 0),
            "prompt_tools_count": ctx.meta.get("prompt_tools_count", 0),
        },
    }
    if ctx.meta.get("modified"):
        record["_modification"] = {
            "enabled": True,
            "note": ctx.meta.get("modification_note", "modified"),
        }
    return record


def _get_conn(db) -> sqlite3.Connection:
    import sqlite3
    return db.conn if hasattr(db, 'conn') else db


async def _persist_phase1(ctx: Context, db) -> None:
    """Phase 1: persist request data (state='pending')."""
    try:
        await asyncio.to_thread(
            lambda: insert_request_phase1(_get_conn(db), ctx)
        )
    except Exception:
        log.exception(f"[{ctx.req_id}] phase1 persist failed")


async def _persist_phase2(ctx: Context, db, writer: TraceWriter) -> None:
    """Phase 2: update response data, modification flag, and write JSONL."""
    try:
        record = _build_trace_record(ctx)
        jsonl_offset = await writer.write(record)
        ctx.meta["jsonl_offset"] = jsonl_offset
        await asyncio.to_thread(
            lambda: update_request_phase2(_get_conn(db), ctx, str(writer.path))
        )
        if ctx.meta.get("modified"):
            await asyncio.to_thread(
                lambda: update_modification_flag(_get_conn(db), ctx)
            )
    except Exception:
        log.exception(f"[{ctx.req_id}] phase2 persist failed")
