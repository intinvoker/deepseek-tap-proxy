"""vscode-copilot-deepseek 检测逻辑 — sys 空路径（BYOK 适配器）。"""
from __future__ import annotations

from deepseek_tap_proxy.pipeline import Context
from deepseek_tap_proxy.plugins.base import DetectionResult
from deepseek_tap_proxy.plugins.consts import (
    COPILOT_NORMAL_PREFIX, COMPACTION_SIGNATURE, SESSION_UUID_RE,
)


def detect_vscode_copilot_deepseek(ctx: Context) -> DetectionResult | None:
    """检测 Agent Copilot BYOK 适配器（sys 空，mapRole System→User）。返回 None = 不匹配。"""
    # sys 必须为空（BYOK 适配器不发送 role=system）
    if ctx.sys:
        return None

    if not ctx.messages:
        return None

    m0 = ctx.messages[0].get("content", "")
    if not isinstance(m0, str):
        return None

    if not (m0.startswith(COPILOT_NORMAL_PREFIX) or
            m0.startswith(COMPACTION_SIGNATURE)):
        return None

    # 提取 session UUID（从所有消息中搜索）
    uuid = _extract_deepseek_uuid(ctx)
    if not uuid:
        return None

    chat_type = _detect_chat_type_deepseek(ctx.messages)

    return DetectionResult(
        client_type="vscode-copilot-deepseek",
        chat_type=chat_type,
        session_id=uuid,
    )


def _extract_deepseek_uuid(ctx: Context) -> str | None:
    """从 messages 中搜索 Copilot session UUID。"""
    for m in ctx.messages:
        c = m.get("content", "")
        if isinstance(c, str):
            match = SESSION_UUID_RE.search(c)
            if match:
                return match.group(1)
    return None


def _detect_chat_type_deepseek(msgs: list[dict]) -> str:
    """Compaction 优先，然后 <agents> → normal/subagent。"""
    if not msgs:
        return "normal"
    m0 = msgs[0].get("content", "")
    if isinstance(m0, str) and m0.startswith(COMPACTION_SIGNATURE):
        return "compaction"
    if isinstance(m0, str):
        if "<agents>" in m0:
            return "normal"
        else:
            return "subagent"
    return "normal"
