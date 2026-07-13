"""统一编排函数 — OpenAI / Anthropic 共用。"""
from __future__ import annotations

from aiohttp import web

from deepseek_tap_proxy.pipeline import Pipeline, Context
from deepseek_tap_proxy.plugins.registry import detect_plugin, ALL_PLUGINS


async def orchestrate(ctx, format_pipeline, plugins, format_tail):
    """管道1（解析+context[]化）→ 检测 → 管道2（客户端处理）。

    Args:
        ctx: Context 对象（body_readonly 已设，format 已设）
        format_pipeline: 管道1 Pipeline 实例
        plugins: 检测链 Plugin 列表（默认 ALL_PLUGINS）
        format_tail: 管道2 格式收尾 Stage 列表（含 Forward/Return）

    Returns:
        ctx（管道2 执行完毕，ctx.response 就绪）
    """
    # ── 管道1：解析 + context[]化 ──
    ctx = await format_pipeline.execute(ctx)
    if ctx.should_abort:
        return ctx

    # ── 检测 ──
    plugin, result = detect_plugin(ctx, plugins or ALL_PLUGINS)
    if plugin is None or result is None:
        ctx.should_abort = True
        ctx.response = web.Response(
            status=422,
            text="Unrecognized client: no session identifier found",
        )
        return ctx

    ctx.client_type = result.client_type
    ctx.chat_type = result.chat_type
    ctx.session_id = result.session_id
    ctx.persist.session_id = result.session_id
    ctx.persist.client_type = result.client_type
    ctx.persist.chat_type = result.chat_type

    # ── 管道2：客户端处理 ──
    middle = plugin.build_stages(ctx)
    pipeline = Pipeline(middle + list(format_tail))
    return await pipeline.execute(ctx)
