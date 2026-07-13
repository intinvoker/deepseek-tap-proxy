"""vscode-copilot 检测逻辑 — sys 非空路径。"""
from __future__ import annotations

from deepseek_tap_proxy.pipeline import Context
from deepseek_tap_proxy.plugins.base import DetectionResult
from deepseek_tap_proxy.plugins.consts import (
    COPILOT_NORMAL_PREFIX, COMPACTION_SIGNATURE, SESSION_UUID_RE,
)


def detect_vscode_copilot(ctx: Context) -> DetectionResult | None:
    """检测 Agent Copilot 原生 Provider（sys 非空）。返回 None = 不匹配。"""
    if not ctx.sys:
        return None

    s0 = ctx.sys[0].get("content", "")
    if not isinstance(s0, str):
        return None

    # 互斥：Hermes 特征 → 不是 Copilot
    if "Session ID:" in s0 or "# Hermes" in s0:
        return None

    # 匹配 Copilot 前缀
    if not (s0.startswith(COPILOT_NORMAL_PREFIX) or
            s0.startswith(COMPACTION_SIGNATURE)):
        return None

    # 提取 session UUID（从 sys[0] 和后续消息中搜索）
    uuid = _extract_copilot_uuid(ctx)
    if not uuid:
        return None

    # 判定 chat_type
    chat_type = _detect_chat_type(ctx.sys, ctx.messages)

    return DetectionResult(
        client_type="vscode-copilot",
        chat_type=chat_type,
        session_id=uuid,
    )


def _extract_copilot_uuid(ctx: Context) -> str | None:
    """从 sys 和 messages 中搜索 Copilot session UUID。"""
    candidates = []
    for s in ctx.sys:
        c = s.get("content", "")
        if isinstance(c, str):
            candidates.append(c)
    for m in ctx.messages:
        c = m.get("content", "")
        if isinstance(c, str):
            candidates.append(c)
    for text in candidates:
        m = SESSION_UUID_RE.search(text)
        if m:
            return m.group(1)
    return None


def _detect_chat_type(sys: list[dict], msgs: list[dict]) -> str:
    """Compaction 优先，然后 <agents> → normal/subagent。"""
    # Compaction 优先
    for source in (sys, msgs):
        if source:
            s0 = source[0].get("content", "")
            if isinstance(s0, str) and s0.startswith(COMPACTION_SIGNATURE):
                return "compaction"

    # sys 非空路径：从 sys[0] 判定
    if sys:
        s0_content = sys[0].get("content", "")
        if isinstance(s0_content, str):
            return "normal" if "<agents>" in s0_content else "subagent"
    return "normal"
