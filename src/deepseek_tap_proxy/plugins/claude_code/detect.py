"""Claude Code 检测逻辑。"""
from __future__ import annotations

from deepseek_tap_proxy.pipeline import Context
from deepseek_tap_proxy.plugins.base import DetectionResult


def detect_claude_code(ctx: Context) -> DetectionResult | None:
    """检测 Claude Code（Anthropic format + x-claude-code-session-id header）。"""
    if ctx.format != "anthropic":
        return None

    headers = {k.lower(): v for k, v in ctx.request.headers.items()}
    sid = headers.get("x-claude-code-session-id", "")
    if not sid:
        return None

    # ── 外部协议 → 内部字段（本模块是唯一的转化点）──
    # 子代理复用主会话的 session-id，靠 agent-id 区分独立的链。
    # 主代理请求不带 agent-id → chain_id 为空串，此处不做任何回落/推导。
    chain_id = headers.get("x-claude-code-agent-id", "")
    parent_chain_id = headers.get("x-claude-code-parent-agent-id", "")

    chat_type = "normal"
    # auto_title 检测（如果可用）
    try:
        from deepseek_tap_proxy.anthropic.persist import _detect_title_gen
        if _detect_title_gen(ctx.body_readonly):
            chat_type = "auto_title"
    except ImportError:
        pass

    return DetectionResult(
        client_type="claude-code",
        chat_type=chat_type,
        session_id=sid,
        chain_id=chain_id,
        parent_chain_id=parent_chain_id,
    )
