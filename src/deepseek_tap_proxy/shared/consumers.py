"""Consumer 函数集合 — handler Consumer 阶段调用。

Consumer 读 ctx，写外部存储（JSONL/OTLP/DB/terminal），不改 ctx。
"""
from __future__ import annotations

import logging

log = logging.getLogger("deepseek-tap")


async def jsonl_write(ctx, writer, store=None) -> None:
    """JSONL 落盘（管道后）。

    不要求 body_readonly 非空——rejected/blocked 请求（body 解析失败）
    也写 minimal 留痕（topic=rejected），保证全请求可追溯。
    """
    from deepseek_tap_proxy.shared.log import TraceView
    if writer and store:
        try:
            record = TraceView(ctx).as_dict()
            jsonl_offset = await writer.write(record)
            ctx.meta["jsonl_offset"] = jsonl_offset
        except Exception:
            log.exception(f"[{ctx.req_id}] JSONL write failed")


async def store_phase1(ctx, store) -> None:
    """MemoryStore Phase 1 入队（管道前）。"""
    if store and ctx.body_readonly is not None:
        await store.enqueue_request_phase1(ctx)


async def store_phase2(ctx, store, writer=None, **kwargs) -> None:
    """MemoryStore Phase 2 入队（管道后）。"""
    if store and ctx.body_readonly is not None:
        jsonl_path = str(writer.path) if writer else ""
        await store.enqueue_request_phase2(ctx, jsonl_path=jsonl_path, **kwargs)


def emit_terminal(ctx) -> None:
    """终端日志发射。"""
    from deepseek_tap_proxy.shared.stages import emit_request_log
    emit_request_log(ctx)


def emit_otlp(ctx, otlp_writer, run_id: str) -> None:
    """OTLP JSONL 落盘。"""
    if otlp_writer is not None and run_id:
        otlp_writer.write(ctx, run_id)


async def run_consumers(ctx, *, writer=None, store=None,
                        otlp_writer=None, run_id="", **kwargs) -> None:
    """运行所有 Consumer。Pipeline 后调用。"""
    await jsonl_write(ctx, writer, store)
    await store_phase2(ctx, store, writer, **kwargs)
    emit_terminal(ctx)
    emit_otlp(ctx, otlp_writer, run_id)
