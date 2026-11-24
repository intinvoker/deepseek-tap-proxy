"""DshCompatStage — 把 DSH 发往 DeepSeek 上游的请求体适配成可接受形态。

两条针对上游接受形态的约束（客户端 → proxy → 上游 → DeepSeek）：

1. ``role: "developer"`` 被 DeepSeek 拒绝
   （422 ``messages[0].role: unknown variant `developer```）；
   该角色的语义等同 system，这里降级为 system。
2. thinking 模式下 assistant 历史必须回传 ``reasoning_content``
   （400 ``The `reasoning_content` in the thinking mode must be passed back``）；
   DSH 在新模型/压缩/回放失效时缺失，这里补空串（空串上游接受）。

**作用范围只限 DeepSeek 模型**（``deepseek*``）：真实请求样本显示 DSH 同时在使用
``mimo-v2.6-flash``，其上游接受 ``developer`` 角色，
本 stage 对非 DeepSeek 模型直接放行，避免改动正在正常工作的流量。

刻意不接 SysToolsLock / PrefixLock：DSH 没有 session_id/chain_id，
且其 context 快照会被主动置换，锁回旧前缀会让模型看到过期上下文。
"""
from __future__ import annotations

import logging

from deepseek_tap_proxy.pipeline import Context, Stage

log: logging.Logger = logging.getLogger("deepseek-tap")

MODIFICATION_NOTE = "dsh-compat"

# 只有 DeepSeek 上游拒绝 developer 角色；其他模型（如 mimo）保持原样。
DEEPSEEK_MODEL_PREFIX = "deepseek"
# DeepSeek V4 / reasoner 默认开启 thinking，assistant 历史必须带 reasoning_content。
_THINKING_MODEL_MARKERS = ("v4", "reasoner")


def _is_deepseek_model(model: str) -> bool:
    return isinstance(model, str) and model.lower().startswith(DEEPSEEK_MODEL_PREFIX)


def _is_thinking_model(model: str) -> bool:
    lowered = model.lower()
    return any(marker in lowered for marker in _THINKING_MODEL_MARKERS)


class DshCompatStage(Stage):
    """在 Forward 之前就地规范化请求体；只在确有改动时刷新规范形式 view。"""

    async def process(self, ctx: Context) -> Context:
        body = ctx.fwd_body
        if not isinstance(body, dict):
            return ctx
        model = body.get("model", "")
        if not _is_deepseek_model(model):
            return ctx

        msgs = body.get("messages")
        if not isinstance(msgs, list):
            return ctx

        inject_reasoning = _is_thinking_model(model)
        changed = False
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "developer":
                msg["role"] = "system"
                changed = True
            elif (
                inject_reasoning
                and msg.get("role") == "assistant"
                and "reasoning_content" not in msg
            ):
                msg["reasoning_content"] = ""
                changed = True

        if changed:
            # CanonicalStage 已按旧 body 算过 hash；改写后重算 sys/tools/messages
            # 的 hash 与计数，后续 persist / 链匹配读到的是转发体本身。
            ctx._activate_view()
            ctx.persist.is_modified = 1
            ctx.persist.modification_note = MODIFICATION_NOTE
        return ctx
