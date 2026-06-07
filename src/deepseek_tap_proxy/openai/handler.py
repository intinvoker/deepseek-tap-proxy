from __future__ import annotations

import asyncio
import json
import logging
import sqlite3

from aiohttp import web

from deepseek_tap_proxy.pipeline import Context, Pipeline
from deepseek_tap_proxy.router import ResolvedTarget
from deepseek_tap_proxy.openai.stages import (
    ParseStage, ForwardStage, ReturnStage,
)
from deepseek_tap_proxy.shared.stages import (
    CanonicalStage, SysToolsLockStage, PrefixLockStage,
    SessionResolverStage, RequestClassifierStage,
)
from deepseek_tap_proxy.shared.db import (
    insert_request_phase1, update_request_phase2,
)
from deepseek_tap_proxy.shared.trace import TraceWriter, build_trace_record
from deepseek_tap_proxy.openai.copilot.compression import CompactionRewrite

log: logging.Logger = logging.getLogger("deepseek-tap")


async def chat_completions_handler(request: web.Request) -> web.StreamResponse:
    """Handle /chat/completions via the request pipeline."""
    # Resolve upstream target via router (config-driven) or fallback to --target
    proxy_ctx = request.app.get("proxy_ctx", {})
    router = proxy_ctx.get("router")
    resolved = None
    if router is not None and router.is_available:
        resolved = router.resolve(request.headers.get("Authorization"))
    if resolved is None:
        resolved = ResolvedTarget(
            target_url=proxy_ctx.get("target_url", ""),
            upstream_name="default",
            matched_key_prefix=None,
        )
    request["upstream"] = resolved
    log.debug(
        f"[upstream] POST /v1/chat/completions → "
        f"{resolved.upstream_name} "
        f"({'prefix ' + resolved.matched_key_prefix if resolved.matched_key_prefix else 'default'})"
    )

    ctx = Context(request=request)
    ctx.target_url = resolved.target_url

    config = request.app.get("config", {})
    lock_mode = getattr(config, "lock_mode", "prefix")
    lock_sys_tools = getattr(config, "lock_sys_tools", True)
    prefix_enabled: bool = lock_mode == "prefix"

    pipeline: Pipeline = _build_pipeline(
        prefix_enabled=prefix_enabled,
        lock_sys_tools=lock_sys_tools,
    )
    ctx = await pipeline.execute(ctx)

    # ── 前缀链优先持久化：管道结束后立即写入，确保下一条请求能命中 ──
    db = request.app.get("db")
    proxy_ctx = request.app.get("proxy_ctx", {})
    writer: TraceWriter | None = proxy_ctx.get("writer")
    persists: set[asyncio.Task] = proxy_ctx.setdefault("_pending_persists", set())
    if db:
        _persist_chain_nowait(ctx, db)  # sync via asyncio.to_thread

    # Background persist (tracked for completion during shutdown)
    if db:
        session = ctx.meta.get("session")
        t1 = asyncio.create_task(_persist_phase1(ctx, db, session=session))
        persists.add(t1)
        t1.add_done_callback(persists.discard)
    if db and writer:
        t2 = asyncio.create_task(_persist_phase2(ctx, db, writer, phase1_task=t1 if db else None))
        persists.add(t2)
        t2.add_done_callback(persists.discard)

    if ctx.response is not None:
        return ctx.response
    return web.Response(status=502, text="No response from pipeline")


def _build_pipeline(prefix_enabled: bool = False, lock_sys_tools: bool = True) -> Pipeline:
    """Build the pipeline with Copilot compaction optimization.

    Pipeline: Parse -> SessionResolver -> Canonical -> SysToolsLock(opt) -> PrefixLock(opt) -> CompactionRewrite -> Forward -> Return
    """
    stages = [
        ParseStage(),
        SessionResolverStage(),
        CanonicalStage(),
        RequestClassifierStage(),
    ]
    if lock_sys_tools:
        stages.append(SysToolsLockStage())
    if prefix_enabled:
        stages.append(PrefixLockStage())
    stages.append(CompactionRewrite())
    stages.append(ForwardStage())
    stages.append(ReturnStage())
    return Pipeline(stages)



def _persist_get(ctx, key: str, default=None):
    """Safe getter for ctx.persist — handles mock ctx objects without persist attribute."""
    persist = getattr(ctx, "persist", None)
    return getattr(persist, key, default) if persist is not None else default


def _get_conn(db) -> sqlite3.Connection:
    return db.conn if hasattr(db, 'conn') else db


def _persist_chain_nowait(ctx: Context, db) -> None:
    """Write prefix chain synchronously (via to_thread) right after pipeline.

    Ensures the chain is in DB before the next request arrives — no deferring.
    """
    ca = getattr(ctx, "chain_action", None)
    session = ctx.meta.get("session")
    if ca and ca.action in ("create", "update") and session and ctx.meta.get("intent") != "compaction":
        from deepseek_tap_proxy.shared.db import upsert_prefix_chain, insert_chain_token
        from deepseek_tap_proxy.shared.colors import green
        import json as _json

        chain_tokens = ctx.persist.total_input_tokens or ctx.meta.get("prompt_tokens", 0)
        conn = _get_conn(db)
        upsert_prefix_chain(
            conn, session.key, ca.chain_id,
            sys=_json.dumps(ca.sys),
            tools=_json.dumps(ca.tools),
            messages=_json.dumps(ca.messages),
            hash_chain=ca.hash_chain,
            msg_count=ca.msg_count,
            safe_breakpoints=ca.safe_breakpoints,
            is_frozen=ca.is_frozen,
        )
        insert_chain_token(
            conn, ca.chain_id,
            msg_count=ca.msg_count,
            tokens=chain_tokens,
            char_count=ca.char_count,
        )
        tag = "+ new" if ca.action == "create" else "~ update"
        print(green(f"  [chain] {tag}: {ca.chain_id[:16]}... ({ca.msg_count} msgs)"))


async def _persist_phase1(ctx: Context, db, session=None) -> None:
    """Phase 1: persist request data (state='pending')."""
    try:
        await asyncio.to_thread(
            lambda: insert_request_phase1(_get_conn(db), ctx, session=session)
        )
    except Exception:
        log.exception(f"[{ctx.req_id}] phase1 persist failed")


async def _persist_phase2(ctx: Context, db, writer: TraceWriter, phase1_task: asyncio.Task | None = None) -> None:
    """Phase 2: update response data, modification flag, and write JSONL.

    Waits for phase1_task to complete first to avoid a race where the
    UPDATE runs before the INSERT (silently affecting 0 rows), leaving
    the record stuck in 'pending' state.
    """
    try:
        if phase1_task is not None:
            await phase1_task
        record = build_trace_record(ctx)
        jsonl_offset = await writer.write(record)
        ctx.meta["jsonl_offset"] = jsonl_offset
        await asyncio.to_thread(
            lambda: update_request_phase2(_get_conn(db), ctx, jsonl_path=str(writer.path))
        )
        # ── 前缀链已在 _persist_chain_nowait 中优先写入 ──
        pass
    except Exception:
        log.exception(f"[{ctx.req_id}] phase2 persist failed")


def _is_normal_conversation(ctx: Context) -> bool:
    """Check if this is a normal user conversation request (not noise)."""
    if ctx.meta.get("intent") == "compaction":
        return False
    if ctx.meta.get("trace_category") == "auto_title":
        return False
    if not ctx.body_readonly:
        return False
    return True
