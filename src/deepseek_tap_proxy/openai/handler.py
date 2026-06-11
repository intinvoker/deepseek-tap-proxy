"""OpenAI API request handler — 使用共享 pipeline 处理 /chat/completions。

Pipeline: ParseStage → CanonicalStage → RequestAnalysisStage
          → SysToolsLockStage(opt) → PrefixLockStage(opt)
          → CompactionRewrite → ForwardStage → ReturnStage

Persist 统一走 MemoryStore 单写者队列（v7 架构），
与 Anthropic handler 对称。
"""
from __future__ import annotations

import logging

from aiohttp import web

from deepseek_tap_proxy.pipeline import Context, Pipeline
from deepseek_tap_proxy.router import ResolvedTarget
from deepseek_tap_proxy.openai.copilot.compression import CompactionRewrite
from deepseek_tap_proxy.openai.stages import (
    ParseStage, ForwardStage, ReturnStage,
)
from deepseek_tap_proxy.shared.stages import (
    CanonicalStage, RequestAnalysisStage,
    SysToolsLockStage, PrefixLockStage,
    emit_request_log,
)
from deepseek_tap_proxy.shared.log import TraceView
from deepseek_tap_proxy.shared.stages.helpers import persist_chain_nowait

log: logging.Logger = logging.getLogger("deepseek-tap")


async def chat_completions_handler(request: web.Request) -> web.StreamResponse:
    """Handle /chat/completions via the shared pipeline."""
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

    # ── 前缀链优先持久化 ──
    db = request.app.get("db")
    proxy_ctx = request.app.get("proxy_ctx", {})
    writer = proxy_ctx.get("writer")

    _snapshot_lock_type = ctx.persist.lock_type
    _snapshot_is_modified = ctx.persist.is_modified
    _snapshot_mod_note = ctx.persist.modification_note

    if db:
        persist_chain_nowait(ctx, db)

    # ── 请求元数据持久化：统一走 MemoryStore 单写者队列 ──
    store = request.app.get("store")
    if store and ctx.body_readonly is not None:
        ctx.persist.api_format = "openai"
        ctx.persist.msg_hash_chain = ctx.msg_hash_chain

        # Phase 1: fire-and-forget insert (state='pending')
        store.enqueue_request_phase1(ctx)

        if writer:
            # JSONL: await in event loop
            try:
                record = TraceView(ctx).as_dict()
                jsonl_offset = await writer.write(record)
                ctx.meta["jsonl_offset"] = jsonl_offset
            except Exception:
                log.exception(f"[{ctx.req_id}] JSONL write failed")

            # Phase 2: fire-and-forget update (state='completed')
            store.enqueue_request_phase2(
                ctx,
                jsonl_path=str(writer.path),
                lock_type_snapshot=_snapshot_lock_type,
                is_modified_snapshot=_snapshot_is_modified,
                mod_note_snapshot=_snapshot_mod_note,
            )

    if ctx.response is not None:
        return ctx.response
    return web.Response(status=502, text="No response from pipeline")


def _build_pipeline(prefix_enabled: bool = False, lock_sys_tools: bool = True) -> Pipeline:
    """Build the OpenAI pipeline using shared stages."""
    stages = [
        ParseStage(),
        CanonicalStage(),
        RequestAnalysisStage(),
    ]
    if lock_sys_tools:
        stages.append(SysToolsLockStage())
    if prefix_enabled:
        stages.append(PrefixLockStage())
    # CompactionRewrite 在锁阶段之后执行，确保不会被 PrefixLock 覆盖
    stages.append(CompactionRewrite())
    stages.append(ForwardStage())
    stages.append(ReturnStage())
    return Pipeline(stages)
