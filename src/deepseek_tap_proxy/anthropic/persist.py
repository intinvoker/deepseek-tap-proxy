"""Anthropic-specific persist helpers (fire-and-forget background tasks)."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from deepseek_tap_proxy.pipeline import Context
from deepseek_tap_proxy.shared.headers import redact

log = logging.getLogger("deepseek-tap")


def _db_insert_phase1(db, **kwargs) -> None:
    """Thread-safe phase1 insert: resolves db.conn inside target thread."""
    from deepseek_tap_proxy.anthropic.db import insert_phase1
    conn = db.conn
    insert_phase1(conn, **kwargs)


def _db_update_phase2(db, **kwargs) -> None:
    """Thread-safe phase2 update: resolves db.conn inside target thread."""
    from deepseek_tap_proxy.anthropic.db import update_phase2
    conn = db.conn
    update_phase2(conn, **kwargs)


def _db_update_modification_flag(db, **kwargs) -> None:
    """Thread-safe modification flag update: resolves db.conn inside target thread."""
    from deepseek_tap_proxy.anthropic.db import update_modification_flag
    conn = db.conn
    update_modification_flag(conn, **kwargs)


async def persist_anthropic(ctx: Context, db, writer) -> None:
    """Fire-and-forget: phase1 -> JSONL -> phase2 -> modification flag."""
    if writer is None or ctx.req_body is None:
        return

    usage = ctx.meta.get("usage", {})
    sid = ctx.meta.get("anthropic_sid", "")
    model = (ctx.req_body or {}).get("model", "")

    # Phase 1: recording
    if db is not None:
        try:
            await asyncio.to_thread(
                _db_insert_phase1, db,
                req_id=ctx.req_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                model=model,
                stream_mode=ctx.is_streaming,
                jsonl_path=str(writer.path),
                turn=0,
                session_id=sid or None,
            )
        except Exception:
            log.exception("[%s] phase1 insert failed", ctx.req_id)

    # JSONL: use fwd_body when modified (stages like Normalize/Block0Clear/Baseline
    # write to fwd_body; req_body is the original). When unmodified, req_body = fwd_body.
    trace_body = ctx.fwd_body if ctx.fwd_body is not None else ctx.req_body
    try:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": ctx.req_id,
            "duration_ms": ctx.duration_ms or 0,
            "upstream_base_url": ctx.meta.get("upstream_base_url", ""),
            "api_format": "anthropic",
            "request": {
                "method": "POST",
                "path": "/v1/messages",
                "headers": redact(dict(ctx.request.headers)),
                "body": trace_body,
            },
            "response": {
                "status": ctx.status,
                "headers": redact(ctx.resp_headers),
                "body": ctx.resp_body,
            },
        }
        if usage:
            record["usage"] = usage
        if "sse_events" in ctx.meta:
            record["response"]["sse_events"] = ctx.meta["sse_events"]
        await writer.write(record)
    except Exception:
        log.exception("[%s] JSONL write failed", ctx.req_id)

    # Phase 2: completed
    if db is not None:
        try:
            await asyncio.to_thread(
                _db_update_phase2, db,
                req_id=ctx.req_id,
                duration_ms=ctx.duration_ms or 0,
                status_code=ctx.status,
                input_tokens=usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0),
                output_tokens=usage.get("output_tokens", 0) or usage.get("completion_tokens", 0),
                cache_hit_tokens=usage.get("cache_read_input_tokens", 0),
                cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
                jsonl_path=str(writer.path),
            )
        except Exception:
            log.exception("[%s] phase2 update failed", ctx.req_id)

        # Modification flag
        if ctx.meta.get("modified"):
            try:
                await asyncio.to_thread(
                    _db_update_modification_flag, db,
                    req_id=ctx.req_id,
                    is_modified=True,
                    modification_note="body modified",
                )
            except Exception:
                log.exception("[%s] modification flag update failed", ctx.req_id)
