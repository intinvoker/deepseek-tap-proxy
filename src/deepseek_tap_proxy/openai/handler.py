"""OpenAI API request handler — 使用共享 pipeline 处理 /chat/completions。

Pipeline: ParseStage → CanonicalStage → RequestAnalysisStage
          → SysToolsLockStage(opt) → PrefixLockStage(opt)
          → CompactionRewrite → ForwardStage → ReturnStage

Persist 统一走 MemoryStore 单写者队列（v7 架构），
与 Anthropic handler 对称。
"""
from __future__ import annotations

import logging

import aiohttp
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
)
from deepseek_tap_proxy.shared.consumers import run_consumers
from deepseek_tap_proxy.shared.stages.helpers import persist_chain_nowait

log: logging.Logger = logging.getLogger("deepseek-tap")


async def chat_completions_handler(request: web.Request) -> web.StreamResponse:
    """Handle /chat/completions via the shared pipeline."""
    # Resolve upstream target via router (config-driven) or fallback to --target
    rt = request.get("runtime")
    router = rt.router if rt else None
    resolved = None
    if router is not None and router.is_available:
        resolved = router.resolve(request.headers.get("Authorization"))
    if resolved is None:
        resolved = ResolvedTarget(
            target_url=getattr(rt.config, "target", "") if rt and rt.config else "",
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

    rt = request.get("runtime")
    config = rt.config if rt else {}
    lock_mode = getattr(config, "lock_mode", "prefix")
    lock_sys_tools = getattr(config, "lock_sys_tools", True)
    prefix_enabled: bool = lock_mode == "prefix"

    pipeline: Pipeline = _build_pipeline(
        prefix_enabled=prefix_enabled,
        lock_sys_tools=lock_sys_tools,
    )
    ctx = await pipeline.execute(ctx)
    ctx.topic = "chat"  # OpenAI /chat/completions 始终是 chat topic

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
    store = rt.store if rt else None
    if store and ctx.body_readonly is not None:
        ctx.persist.api_format = "openai"
        ctx.persist.msg_hash_chain = ctx.msg_hash_chain

        # Phase 1: fire-and-forget insert (state='pending')
        store.enqueue_request_phase1(ctx)

        await run_consumers(
            ctx,
            writer=writer,
            store=store,
            otlp_writer=proxy_ctx.get("otlp_log_writer"),
            run_id=proxy_ctx.get("run_id", ""),
            lock_type_snapshot=_snapshot_lock_type,
            is_modified_snapshot=_snapshot_is_modified,
            mod_note_snapshot=_snapshot_mod_note,
        )

    if ctx.response is not None:
        return ctx.response
    return web.Response(status=502, text="No response from pipeline")


async def fim_passthrough_handler(request: web.Request) -> web.Response:
    """Passthrough FIM /completions and /beta/completions requests to upstream."""
    rt = request.get("runtime")
    session = getattr(rt.transport, '_session', None) if rt else None
    if session is None:
        return web.Response(status=502, text="Transport not configured")
    target = getattr(rt.config, "target", "") if rt and rt.config else ""
    path_prefix = ""

    # Resolve upstream via router
    router = rt.router if rt else None
    if router is not None and router.is_available:
        resolved = router.resolve(request.headers.get("Authorization"))
        if resolved is not None:
            target = resolved.target_url
            if resolved.path_prefix:
                path_prefix = resolved.path_prefix

    body = await request.read()
    if path_prefix and not request.path.startswith(path_prefix):
        upstream_url = target.rstrip("/") + path_prefix + request.path
    else:
        upstream_url = target.rstrip("/") + request.path

    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host",)}
    from deepseek_tap_proxy.shared.headers import filter_hop_by_hop
    fwd_headers = filter_hop_by_hop(fwd_headers)

    try:
        upstream_resp = await session.request(
            method="POST", url=upstream_url, data=body, headers=fwd_headers,
            timeout=aiohttp.ClientTimeout(total=getattr(rt.config, "timeout", 120) if rt and rt.config else 120),
        )
    except Exception as exc:
        log.error(f"[fim] upstream error for {request.path}: {exc}")
        return web.Response(status=502, text=str(exc))

    resp_body = await upstream_resp.read()
    return web.Response(
        status=upstream_resp.status,
        headers={k: v for k, v in upstream_resp.headers.items()
                 if k.lower() not in ("transfer-encoding", "content-encoding", "content-length")},
        body=resp_body,
    )


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
