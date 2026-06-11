"""CanonicalStage — 激活规范形式 view，计算 hash_chain 和元数据。

在 ParseStage 之后执行。调用 ``ctx._activate_view()`` 首次计算所有 hash/count。
之后 view setter 自动增量刷新。
"""
from __future__ import annotations

from deepseek_tap_proxy.pipeline import Stage, Context


class CanonicalStage(Stage):
    """激活规范形式 view，计算 hash_chain 和元数据。

    写入 ctx.persist / ctx.meta 供 persist 和 log 消费。
    """

    def __init__(self, extensions: list[str] | None = None):
        self.extensions = extensions or []

    async def process(self, ctx: Context) -> Context:
        if not ctx.body_readonly or not ctx.fwd_body:
            return ctx

        # 首次激活 canonical view：计算三段 hash + count
        ctx._activate_view()

        total_msgs = ctx.orig_sys_len + ctx.orig_tools_len + ctx.orig_msgs_len
        ctx.persist.prompt_messages_count = total_msgs
        ctx.persist.prompt_tools_count = ctx.orig_tools_len
        ctx.meta["log_msgs_before"] = total_msgs

        # 扩展 view：同名映射 from fwd_body
        for name in self.extensions:
            ctx.ext[name] = ctx.fwd_body.get(name, {})

        return ctx
