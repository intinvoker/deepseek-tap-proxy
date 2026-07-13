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
    )
