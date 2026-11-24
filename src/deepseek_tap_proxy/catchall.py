"""通用 catch-all 透明路由 — 未知端点按 API key 转发到 upstream 并留痕。

用途：DeepSeek Search / Fetch 等**新 API** 无需改 proxy 代码即可经 proxy 留痕。
约束：

* 仅接管**未注册**路径（已注册的 handler / 内部 API 优先匹配）；
* ``/anthropic/*`` 路径由 Anthropic 路由自身做路径保护（保持既有 404 行为），
  不在 catch-all 范围；
* 一律 passthrough：不解析改写 body、不写缓存链，仅 trace/DB/OTLP 留痕。
"""
from __future__ import annotations

from aiohttp import web

from deepseek_tap_proxy.pipeline import Context
from deepseek_tap_proxy.router import ResolvedTarget


async def catchall_handler(request: web.Request) -> web.StreamResponse:
    """未知路径透明转发（POST /v1/search、/v1/fetch 等新 API）。"""
    rt = request.get("runtime")
    resolved = request.get("upstream")
    if resolved is None:
        target = getattr(rt.config, "target", "") if rt is not None and rt.config else ""
        resolved = ResolvedTarget(
            target_url=target, upstream_name="default", matched_key_prefix=None,
        )
        request["upstream"] = resolved

    ctx = Context(request=request, meta={})
    ctx.target_url = resolved.target_url
    ctx.meta["upstream_base_url"] = resolved.target_url
    ctx.topic = "passthrough"

    from deepseek_tap_proxy.shared.passthrough import (
        execute_passthrough, read_and_parse_request, resolve_request_policy,
    )

    await read_and_parse_request(ctx)
    decision, plugin, _result = resolve_request_policy(ctx)
    ctx.persist.mode = "passthrough"
    ctx.persist.kind = decision.kind or "other"
    ctx.meta["policy_source"] = "catchall"
    ctx.meta["policy_reason"] = "unknown-path"
    ctx.meta["kind"] = ctx.persist.kind

    await execute_passthrough(ctx, mode="passthrough", topic="passthrough")
    return ctx.response or web.Response(status=502, text="No response")
