"""响应端缓存分析 + 日志发射 —— 格式无关，由 ForwardStage 内部调用。"""
from __future__ import annotations

from deepseek_tap_proxy.pipeline.context import Context


def apply_cache_analysis(ctx: Context) -> None:
    """Read normalized usage from ``ctx.resp_usage`` and write to ``ctx.meta`` / ``ctx.persist``.

    Call from any ForwardStage (OpenAI / Anthropic) after upstream response is available.
    """
    usage = ctx.resp_usage
    ctx.persist.total_input_tokens = usage.get("total_input_tokens", 0)
    ctx.persist.output_tokens = usage.get("output_tokens", 0)
    ctx.persist.cache_hit_tokens = usage.get("cache_read_input_tokens", 0)
    ctx.persist.cache_miss_tokens = usage.get("cache_miss_input_tokens", 0)

    hit = ctx.persist.cache_hit_tokens
    miss = ctx.persist.cache_miss_tokens
    total = hit + miss
    ctx.persist.cache_hit_ratio = round(hit / total, 4) if total > 0 else 0.0

    ctx.persist.has_reasoning = ctx.resp_has_reasoning
    ctx.persist.has_tool_calls = ctx.resp_has_tool_calls
    ctx.persist.prompt_messages_count = len(ctx.sys) + len(ctx.messages)
    ctx.persist.prompt_tools_count = len(ctx.tools)


def emit_request_log(ctx: Context) -> None:
    """Emit per-request log: terminal + OTLP (if writer configured).

    Delegates to shared/log.py — single implementation via LogView(ctx).
    """
    from deepseek_tap_proxy.shared.log import emit_request_log as _emit
    _emit(ctx)
