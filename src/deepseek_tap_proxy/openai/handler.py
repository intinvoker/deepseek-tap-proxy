"""OpenAI API request handler — 使用共享 pipeline 处理 /chat/completions。

Pipeline: ParseStage → CanonicalStage → RequestAnalysisStage
          → SysToolsLockStage(opt) → PrefixLockStage(opt)
          → CompactionLockStage → CompactionRewrite
          → ForwardStage → ReturnStage

Persist 统一走 MemoryStore 单写者队列（v7 架构），
与 Anthropic handler 对称。
"""
from __future__ import annotations

import logging

import aiohttp
from aiohttp import web

from deepseek_tap_proxy.pipeline import Context
from deepseek_tap_proxy.pipeline.context import resolve_request_id
from deepseek_tap_proxy.router import ResolvedTarget
from deepseek_tap_proxy.shared.consumers import run_consumers
from deepseek_tap_proxy.shared.stages.helpers import persist_chain_nowait, create_chain_moment
from deepseek_tap_proxy.shared.orchestrate import orchestrate

log: logging.Logger = logging.getLogger("deepseek-tap")


async def chat_completions_handler(request: web.Request) -> web.StreamResponse:
    """Handle /chat/completions via mode-aware dispatch (passthrough/adapter/managed)."""
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

    # ── 模式解析（预解析只读；managed 走现状完整管线）──
    from deepseek_tap_proxy.shared.passthrough import (
        execute_passthrough, read_and_parse_request, resolve_request_policy,
    )

    await read_and_parse_request(ctx)
    decision, plugin, _result = resolve_request_policy(ctx)
    ctx.persist.mode = decision.mode
    ctx.persist.kind = decision.kind or "chat"
    ctx.meta["mode"] = decision.mode
    ctx.meta["kind"] = ctx.persist.kind
    ctx.topic = "chat"

    from deepseek_tap_proxy.shared.passthrough import force_managed_for_enforcement
    forced_managed = force_managed_for_enforcement(ctx, decision)

    if decision.mode not in ("managed", "legacy") and not forced_managed:
        # passthrough / adapter：原始字节转发 + 尽力持久化（不写缓存链）
        await execute_passthrough(ctx, mode=decision.mode, plugin=plugin, topic="chat")
        return ctx.response or web.Response(status=502, text="No response from pipeline")
    if forced_managed:
        ctx.persist.mode = "managed"

    from deepseek_tap_proxy.format.openai.pipeline import OPENAI_FORMAT_PIPELINE, OPENAI_FORMAT_TAIL
    from deepseek_tap_proxy.shared.orchestrate import orchestrate
    from deepseek_tap_proxy.plugins.registry import ALL_PLUGINS

    ctx = await orchestrate(ctx, OPENAI_FORMAT_PIPELINE, ALL_PLUGINS, OPENAI_FORMAT_TAIL)
    ctx.topic = "chat"  # OpenAI /chat/completions 始终是 chat topic

    # ── 前缀链优先持久化 ──
    db = request.app.get("db")
    proxy_ctx = request.app.get("proxy_ctx", {})
    writer = proxy_ctx.get("writer")

    _snapshot_lock_type = ctx.persist.lock_type
    _snapshot_is_modified = ctx.persist.is_modified
    _snapshot_mod_note = ctx.persist.modification_note

    if db:
        await persist_chain_nowait(ctx, db)

    # ── 请求元数据持久化：统一走 MemoryStore 单写者队列 ──
    store = rt.store if rt else None
    if store and ctx.body_readonly is not None:
        ctx.persist.api_format = "openai"
        ctx.persist.msg_hash_chain = ctx.msg_hash_chain

        # Phase 1: fire-and-forget insert (state='pending')
        await store.enqueue_request_phase1(ctx)

        # ── Handler Moment creation (after pipeline, before run_consumers) ──
        await create_chain_moment(ctx, store)

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

    # Resolve upstream via router
    router = rt.router if rt else None
    if router is not None and router.is_available:
        resolved = router.resolve(request.headers.get("Authorization"))
        if resolved is not None:
            target = resolved.target_url
            # path_prefix 是 Anthropic Messages API 的路径覆盖——OpenAI 请求不应用

    body = await request.read()
    upstream_url = target.rstrip("/") + request.path

    # 转发头走唯一入口：去 Host / 去 hop-by-hop / 强制 Accept-Encoding: identity
    from deepseek_tap_proxy.shared.headers import prepare_fwd_headers
    fwd_headers = prepare_fwd_headers(request.headers)

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


async def responses_passthrough_handler(request: web.Request) -> web.StreamResponse:
    """Passthrough /v1/responses to upstream (stateless, no caching/analysis).

    Responses API（input/instructions + web_search 工具，2026-08 DeepSeek 新增）请求格式
    与 chat/completions 不同，不走共享 pipeline；web_search 注入动态搜索结果，跳过前缀缓存。
    流式 SSE 原样转发（Responses 流以 response.completed 结尾，无 [DONE]）。
    """
    rt = request.get("runtime")
    session = getattr(rt.transport, '_session', None) if rt else None
    if session is None:
        return web.Response(status=502, text="Transport not configured")
    target = getattr(rt.config, "target", "") if rt and rt.config else ""

    router = rt.router if rt else None
    if router is not None and router.is_available:
        resolved = router.resolve(request.headers.get("Authorization"))
        if resolved is not None:
            target = resolved.target_url
            # path_prefix 是 Anthropic Messages API 的路径覆盖——OpenAI 请求不应用

    body = await request.read()
    upstream_url = target.rstrip("/") + request.path

    # 转发头走唯一入口：去 Host / 去 hop-by-hop / 强制 Accept-Encoding: identity
    from deepseek_tap_proxy.shared.headers import prepare_fwd_headers
    fwd_headers = prepare_fwd_headers(request.headers)

    try:
        upstream_resp = await session.request(
            method="POST", url=upstream_url, data=body, headers=fwd_headers,
            timeout=aiohttp.ClientTimeout(
                total=getattr(rt.config, "timeout", 120) if rt and rt.config else 120),
        )
    except Exception as exc:
        log.error(f"[responses] upstream error for {request.path}: {exc}")
        return web.Response(status=502, text=str(exc))

    # 响应头回写 X-Request-Id（从 x-tap-request-id 取；非法/缺失生成
    # req_<uuid12>）。stateless passthrough —— 不写 DB。剔除上游同名头避免重复。
    req_id = resolve_request_id(request)
    resp_headers = {
        k: v for k, v in upstream_resp.headers.items()
        if k.lower() not in (
            "transfer-encoding", "content-encoding", "content-length", "x-request-id",
        )
    }
    resp_headers["X-Request-Id"] = req_id
    resp = web.StreamResponse(
        status=upstream_resp.status,
        headers=resp_headers,
    )
    await resp.prepare(request)
    async for chunk in upstream_resp.content.iter_any():
        await resp.write(chunk)
    await resp.write_eof()
    return resp
