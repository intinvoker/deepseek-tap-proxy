"""Hermes 检测逻辑。"""
from __future__ import annotations

from deepseek_tap_proxy.pipeline import Context
from deepseek_tap_proxy.plugins.base import DetectionResult
from deepseek_tap_proxy.plugins.consts import (
    COPILOT_NORMAL_PREFIX, COMPACTION_SIGNATURE, HERMES_SESSION_ID_RE,
)


def detect_hermes(ctx: Context) -> DetectionResult | None:
    """检测 Hermes Agent。返回 None = 不匹配。"""
    if not ctx.sys:
        return None

    s0 = ctx.sys[0].get("content", "")
    if not isinstance(s0, str):
        return None

    # 互斥：Copilot 签名优先
    if s0.startswith(COPILOT_NORMAL_PREFIX) or s0.startswith(COMPACTION_SIGNATURE):
        return None

    if "Session ID:" not in s0 and "# Hermes" not in s0:
        return None

    m = HERMES_SESSION_ID_RE.search(s0)
    if not m:
        return None

    return DetectionResult(
        client_type="hermes",
        chat_type="normal",
        session_id=m.group(1),
    )
