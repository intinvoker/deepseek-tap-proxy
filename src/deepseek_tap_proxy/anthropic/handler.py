"""Anthropic proxy handler -- pipeline orchestration only."""
from __future__ import annotations

import logging
from aiohttp import web

from deepseek_tap_proxy.pipeline import Context, Pipeline, Stage
from deepseek_tap_proxy.router import ResolvedTarget
from deepseek_tap_proxy.shared.stages import (
    CanonicalStage, SysToolsLockStage, RequestAnalysisStage,
)
from deepseek_tap_proxy.shared.stages.helpers import persist_chain_nowait
from deepseek_tap_proxy.anthropic.persist import _detect_title_gen
from deepseek_tap_proxy.anthropic.stages import (
    AnthropicParseStage, AnthropicBlock0ClearStage,
    AnthropicSystemRoleConvertStage,
    AnthropicUpstreamPathStage, AnthropicForwardStage, ReturnStage,
)

log = logging.getLogger("deepseek-tap")


# ── Pipeline 定义 ──────────────────────────────────────────────────────────

FULL_ANTHROPIC: list[type[Stage]] = [
    AnthropicParseStage,
    AnthropicBlock0ClearStage,
    CanonicalStage,
    RequestAnalysisStage,
    SysToolsLockStage,
    AnthropicSystemRoleConvertStage,
    AnthropicUpstreamPathStage,
    AnthropicForwardStage,
    ReturnStage,
]

# RequestAnalysisStage 内部按 ctx.format 分发，所有 client_type 共用同一 pipeline
ANTHROPIC_SKIP_TABLE: dict[str, set[type[Stage]]] = {
    "generic": set(),
}


def setup_anthropic_pipelines(app: web.Application) -> None:
    """启动时构建 Anthropic pipeline。由 server.py 调用。"""
    from deepseek_tap_proxy.pipeline import PipelineFactory
    factory = PipelineFactory()
    factory.build_all(FULL_ANTHROPIC, ANTHROPIC_SKIP_TABLE)
    app["anthropic_pipeline"] = factory


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

    factory = request.app.get("anthropic_pipeline")
    if factory is not None:
        pipeline = factory.get("generic")  # 单 pipeline，所有类型共用
    else:
        # Fallback for tests without setup_anthropic_pipelines
        pipeline = Pipeline(FULL_ANTHROPIC)
    ctx = await pipeline.execute(ctx)

    # ── Persist（MemoryStore + 单写者队列）──
    db = request.app.get("db")
    proxy_ctx = request.app.get("proxy_ctx", {})
    writer = proxy_ctx.get("writer")

    if db:
        persist_chain_nowait(ctx, db)  # chains/sessions → MemoryStore + SQLite

    # ── 请求元数据持久化：统一走 MemoryStore 单写者队列 ──
    store = request.app.get("store")
    if store and ctx.body_readonly is not None:
        # Set anthropic-specific metadata before snapshot
        ctx.persist.api_format = "anthropic"
        ctx.persist.msg_hash_chain = ctx.msg_hash_chain
        # Detect auto-title generation requests
        _tc = _detect_title_gen(ctx.body_readonly)
        if _tc:
            ctx.persist.trace_category = _tc

        # Phase 1: fire-and-forget insert (state='pending')
        store.enqueue_request_phase1(ctx)

        if writer:
            # JSONL: await in event loop
            try:
                from deepseek_tap_proxy.shared.log import TraceView
                record = TraceView(ctx).as_dict()
                if "sse_events" in ctx.meta:
                    record["response"]["sse_events"] = ctx.meta["sse_events"]
                jsonl_offset = await writer.write(record)
                ctx.meta["jsonl_offset"] = jsonl_offset
            except Exception:
                log.exception(f"[{ctx.req_id}] JSONL write failed")

            # Phase 2: fire-and-forget update (state='completed')
            store.enqueue_request_phase2(ctx, jsonl_path=str(writer.path))

    return ctx.response or web.Response(status=502, text="No response")
