"""Anthropic proxy handler -- pipeline orchestration only."""
from __future__ import annotations

import logging
from aiohttp import web

from deepseek_tap_proxy.pipeline import Context
from deepseek_tap_proxy.router import ResolvedTarget


# ── Request triage ────────────────────────────────────────────────────────────


def classify_anthropic(request: web.Request) -> str:
    """根据请求路径分类 Anthropic 请求的 topic。

    Returns: "chat" | "count_tokens" | "passthrough" | "rejected"
    """
    path = request.path
    if path.startswith("/anthropic"):
        path = path[len("/anthropic"):]
    clean = path.rstrip("/")

    if clean == "/v1/messages":
        return "chat"
    if clean == "/v1/messages/count_tokens":
        return "count_tokens"
    if clean in ("/v1/complete", "/v1/models"):
        return "passthrough"
    return "rejected"


def estimate_tokens(body: dict | None) -> int:
    """本地 token 估算（fallback 用），字符数 / 4。"""
    if not isinstance(body, dict):
        return 0
    total_chars = 0
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total_chars += len(block.get("text", ""))
        elif isinstance(content, str):
            total_chars += len(content)
    return max(1, total_chars // 4)


async def handle_count_tokens(ctx: Context) -> Context:
    """先尝试上游，失败则 fallback 本地估算。"""
    import json
    import orjson
    import aiohttp

    upstream = ctx.request.get("upstream")
    target = ctx.target_url
    upstream_ok = False

    if upstream and target and getattr(upstream, 'path_prefix', None):
        try:
            rt = ctx.request.get("runtime")
            session = rt.session if rt else None
            if session:
                url = target.rstrip("/") + upstream.path_prefix.rstrip("/") + "/messages/count_tokens"
                fwd_body = ctx.fwd_body or ctx.body_readonly
                if fwd_body:
                    body_bytes = orjson.dumps(fwd_body)
                else:
                    body_bytes = b"{}"
                async with session.post(
                    url, data=body_bytes, headers=ctx.fwd_headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        ctx.meta["tokens_source"] = "real"
                        ctx.resp_body = {"input_tokens": data.get("input_tokens", 0)}
                        ctx.status = 200
                        ctx.response = web.json_response(ctx.resp_body)
                        upstream_ok = True
        except Exception as exc:
            log.warning(f"[count_tokens] upstream request failed: {exc}, "
                        f"falling back to local estimate")

    if not upstream_ok:
        # 优先本地 tokenizer 精确计数（默认关闭；配置 DEEPSEEK_TAP_TOKENIZER_URL 后启用），
        # 不可用/超时再退回字符估算。见 shared/tokenizer_client.py。
        from deepseek_tap_proxy.shared.tokenizer_client import count_tokens_exact

        rt = ctx.request.get("runtime")
        session = getattr(rt, "session", None) if rt else None
        exact = await count_tokens_exact(ctx.body_readonly, session)
        if exact is not None:
            ctx.meta["tokens_source"] = "local_tokenizer"
            ctx.meta["estimated_tokens"] = exact
            ctx.resp_body = {"input_tokens": exact}
        else:
            estimated = estimate_tokens(ctx.body_readonly)
            ctx.meta["tokens_source"] = "fallback"
            ctx.meta["estimated_tokens"] = estimated
            ctx.resp_body = {"input_tokens": estimated}
        ctx.status = 200
        ctx.response = web.json_response(ctx.resp_body)

    return ctx
from deepseek_tap_proxy.shared.consumers import store_phase1, run_consumers
from deepseek_tap_proxy.shared.stages.helpers import persist_chain_nowait, create_chain_moment
from deepseek_tap_proxy.anthropic.stages import AnthropicParseStage

log = logging.getLogger("deepseek-tap")


# ── 轻量解析（非 chat topic 用）─────────────────────────────────────────────


async def _light_parse(ctx: Context) -> bool:
    """非 chat topic 的轻量解析——调用 ParseStage 只做 body parse + headers。"""
    from deepseek_tap_proxy.anthropic.stages import AnthropicParseStage
    ctx = await AnthropicParseStage().process(ctx)
    return not ctx.should_abort


async def _persist_after_pipeline(ctx: Context) -> None:
    """managed 管线后的 Phase1/Phase2 落盘（现状行为，原样保留）。"""
    request = ctx.request
    db = request.app.get("db")
    rt = request.get("runtime")
    store = rt.store if rt else None
    proxy_ctx = request.app.get("proxy_ctx", {})
    writer = proxy_ctx.get("writer")

    if db:
        await persist_chain_nowait(ctx, db)
    if store:
        ctx.persist.api_format = "anthropic"
        ctx.persist.msg_hash_chain = ctx.msg_hash_chain
        ctx.persist.model = (ctx.body_readonly or {}).get("model", "")
        ctx.persist.stream_mode = ctx.is_streaming
    await store_phase1(ctx, store)

    # ── Handler Moment creation (after pipeline, before run_consumers) ──
    if store and ctx.body_readonly is not None:
        await create_chain_moment(ctx, store)

    # ── Phase 2 落盘（管道后，JSONL + DB + log）──
    await run_consumers(
        ctx,
        writer=writer,
        store=store,
        otlp_writer=proxy_ctx.get("otlp_log_writer"),
        run_id=proxy_ctx.get("run_id", ""),
    )


async def anthropic_proxy_handler(request: web.Request) -> web.StreamResponse:
    """Anthropic 请求入口：解析 -> 模式决策 -> 分诊 -> 处理 -> Phase2落盘 -> 返回。"""
    # ── 1. Upstream 解析（现有逻辑）──
    rt = request.get("runtime")
    router = rt.router if rt is not None else None
    resolved = None
    if router is not None and router.is_available:
        resolved = router.resolve(
            request.headers.get("Authorization") or request.headers.get("x-api-key")
        )
    if resolved is None:
        target_url = getattr(rt.config, "target", "") if rt and rt.config else ""
        resolved = ResolvedTarget(
            target_url=target_url,
            upstream_name="default",
            matched_key_prefix=None,
        )
    request["upstream"] = resolved
    log.debug(
        f"[upstream] {request.method} {request.path} -> "
        f"{resolved.upstream_name} "
        f"({'prefix ' + resolved.matched_key_prefix if resolved.matched_key_prefix else 'default'})"
    )

    ctx = Context(request=request, meta={})
    ctx.target_url = resolved.target_url
    ctx.meta["upstream_base_url"] = resolved.target_url

    # ── 2. 分类 + 模式解析（预解析只读，不改写 body/response）──
    ctx.topic = classify_anthropic(request)
    from deepseek_tap_proxy.shared.passthrough import (
        execute_passthrough,
        read_and_parse_request,
        resolve_request_policy,
    )

    await read_and_parse_request(ctx)
    decision, plugin, _result = resolve_request_policy(ctx)
    ctx.persist.mode = decision.mode
    ctx.persist.kind = decision.kind
    ctx.meta["mode"] = decision.mode
    ctx.meta["kind"] = decision.kind

    # ── 3. 按 topic + mode 分流 ──
    if ctx.topic == "chat":
        from deepseek_tap_proxy.shared.passthrough import force_managed_for_enforcement

        managed = decision.mode in ("managed", "legacy") or force_managed_for_enforcement(ctx, decision)
        # 非法 JSON 的 chat 请求保持既有 400 行为（交给 ParseStage）
        must_parse = ctx.raw_body and ctx.body_readonly is None
        if managed or must_parse:
            from deepseek_tap_proxy.format.anthropic.pipeline import (
                ANTHROPIC_FORMAT_PIPELINE, ANTHROPIC_FORMAT_TAIL,
            )
            from deepseek_tap_proxy.shared.orchestrate import orchestrate
            from deepseek_tap_proxy.plugins.registry import ALL_PLUGINS

            ctx = await orchestrate(
                ctx, ANTHROPIC_FORMAT_PIPELINE, ALL_PLUGINS, ANTHROPIC_FORMAT_TAIL,
            )
            # 实际执行的是 managed 管线（模式如实标注）
            ctx.persist.mode = decision.mode if managed else "managed"
            await _persist_after_pipeline(ctx)
        else:
            # passthrough / adapter：字节级转发 + 尽力持久化（不写缓存链）
            await execute_passthrough(
                ctx, mode=decision.mode, plugin=plugin, topic="chat",
            )
        return ctx.response or web.Response(status=502, text="No response")

    # ── 非 chat topic: 需要独立 parse（因为不走 pipeline）──
    ok = await _light_parse(ctx)
    if not ok:
        # abort（blocked path / 非法 JSON）也继续落盘留痕——不提前 return
        ctx.status = getattr(ctx.response, "status", 404)
        ctx.response = ctx.response or web.Response(status=404, text="Not Found")

    if ctx.topic == "count_tokens":
        ctx = await handle_count_tokens(ctx)
    elif ctx.topic == "passthrough":
        ctx.status = 404
        ctx.response = web.Response(status=404, text="Not Implemented")
    else:  # rejected
        ctx.status = 404
        ctx.response = web.Response(status=404, text="Not Found")

    await _persist_after_pipeline(ctx)
    return ctx.response or web.Response(status=502, text="No response")
