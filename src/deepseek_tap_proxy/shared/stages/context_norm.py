"""ContextNormStage — 规范化 Copilot <context> 块，消除 volatile 元素对 prefix cache 的影响。

仅对 vscode-copilot-deepseek + normal chat_type 生效。

Copilot 每个 user turn 注入 <context> 块（日期、终端状态等），该块：
  - 每天日期变化 → hash 变化
  - 每条命令终端状态变化 → hash 变化
  - VSCode 重启后历史 <context> 消失 → hash 变化

剥离 <context> 和 <reminderInstructions> 块，保留 <userRequest>（用户实际输入）。

2026-06-24: 实现后通过 inspector rebuild 验证，对命中率无改善（0% delta）。
context_shift (32 events) 的根因是系统提示词变化，不是 <context> 块。
已从管道中移除，代码保留供参考。
"""
from __future__ import annotations

import logging
import re

from deepseek_tap_proxy.pipeline import Stage, Context

log = logging.getLogger("deepseek-tap")

# 匹配 <context>...</context> 和 <reminderInstructions>...</reminderInstructions>
_CONTEXT_RE = re.compile(r'<context>[\s\S]*?</context>\s*')
_REMINDER_RE = re.compile(r'<reminderInstructions>[\s\S]*?</reminderInstructions>\s*')


def _strip_context_blocks(content: str) -> str:
    """从 user 消息内容中剥离 <context> 和 <reminderInstructions> 块。"""
    result = _CONTEXT_RE.sub('', content)
    result = _REMINDER_RE.sub('', result)
    return result


class ContextNormStage(Stage):
    """规范化 Copilot <context> 块，消除 volatile 元素对 prefix cache 的影响。"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    async def process(self, ctx: Context) -> Context:
        if not self.enabled:
            return ctx

        # 仅对 vscode-copilot-deepseek + normal 生效
        if ctx.client_type != "vscode-copilot-deepseek":
            return ctx
        if ctx.chat_type != "normal":
            return ctx

        msgs = ctx.messages
        if not msgs:
            return ctx

        # 扫描 user 消息，剥离 <context> 和 <reminderInstructions>
        changed = False
        new_msgs = []
        for msg in msgs:
            if msg.get("role") != "user":
                new_msgs.append(msg)
                continue

            content = msg.get("content", "")
            if not isinstance(content, str):
                new_msgs.append(msg)
                continue

            if "<context>" not in content and "<reminderInstructions>" not in content:
                new_msgs.append(msg)
                continue

            stripped = _strip_context_blocks(content)
            if stripped.strip():
                new_msgs.append({**msg, "content": stripped})
                changed = True
            # 如果剥离后为空，跳过该消息（不应发生，但防御性处理）

        if not changed:
            return ctx

        # 写回 messages（setter 自动刷新 hash chain）
        ctx.messages = new_msgs
        log.debug("[%s] context_norm: stripped %d messages", ctx.req_id,
                  sum(1 for a, b in zip(msgs, new_msgs) if a is not b))
        return ctx
