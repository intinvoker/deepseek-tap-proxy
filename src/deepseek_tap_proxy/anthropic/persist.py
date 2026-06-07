"""Anthropic-specific persist helpers (fire-and-forget background tasks)."""
from __future__ import annotations

import asyncio
import json
import logging

from deepseek_tap_proxy.pipeline import Context
from deepseek_tap_proxy.shared.trace import build_trace_record

log = logging.getLogger("deepseek-tap")

# ── Title generation detection ─────────────────────────────────────────

_TITLE_GEN_KEYWORDS = (
    "sentence-case title",
    "Generate a concise",
    "3-7 words",
    "session in a list",
)


def _detect_title_gen(req_body: dict | None) -> str | None:
    """Detect if this request is an auto-generated title request.

    Returns "auto_title" if the system prompt contains title-generation
    keywords AND the request carries no tools. Returns None otherwise.
    This is a pure function — no IO, runs in async fire-and-forget.
    """
    if not isinstance(req_body, dict):
        return None

    system = req_body.get("system")
    if not system:
        return None

    # Extract text from system prompt (string or block array)
    texts = []
    if isinstance(system, str):
        texts = [system]
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                for key in ("text", "thinking"):
                    txt = block.get(key)
                    if isinstance(txt, str):
                        texts.append(txt)

    combined = " ".join(texts)
    if not any(kw in combined for kw in _TITLE_GEN_KEYWORDS):
        return None

    # Title generation requests don't carry tools
    if req_body.get("tools"):
        return None

    return "auto_title"


def _db_insert_phase1(db, ctx, session=None) -> None:
    """Thread-safe phase1 insert via shared unified function."""
    from deepseek_tap_proxy.shared.db import insert_request_phase1
    conn = db.conn
    insert_request_phase1(conn, ctx, session=session)


def _db_update_phase2(db, ctx, jsonl_path: str = '') -> None:
    """Thread-safe phase2 update via shared unified function."""
    from deepseek_tap_proxy.shared.db import update_request_phase2
    conn = db.conn
    update_request_phase2(conn, ctx, jsonl_path=jsonl_path)


async def persist_anthropic(ctx: Context, db, writer) -> None:
    """Fire-and-forget: phase1 -> JSONL -> phase2 -> modification flag."""
    if writer is None or ctx.body_readonly is None:
        return

    usage = ctx.meta.get("usage", {})
    sid = ctx.meta.get("anthropic_sid", "")
    model = (ctx.body_readonly or {}).get("model", "")
    trace_category = _detect_title_gen(ctx.body_readonly)
    session = ctx.meta.get("session")

    # Phase 1: recording (use shared unified function)
    if db is not None:
        try:
            ctx.persist.trace_category = trace_category or ""
            ctx.persist.api_format = "anthropic"
            await asyncio.to_thread(
                _db_insert_phase1, db, ctx, session=session,
            )
        except Exception:
            log.exception("[%s] phase1 insert failed", ctx.req_id)

    # JSONL: unified builder
    try:
        record = build_trace_record(ctx)
        if "sse_events" in ctx.meta:
            record["response"]["sse_events"] = ctx.meta["sse_events"]
        offset = await writer.write(record)
    except Exception:
        log.exception("[%s] JSONL write failed", ctx.req_id)
        offset = 0

    # Phase 2: completed (use shared unified function)
    if db is not None:
        try:
            ctx.meta["jsonl_offset"] = offset
            await asyncio.to_thread(
                _db_update_phase2, db, ctx, jsonl_path=str(writer.path),
            )
        except Exception:
            log.exception("[%s] phase2 update failed", ctx.req_id)

