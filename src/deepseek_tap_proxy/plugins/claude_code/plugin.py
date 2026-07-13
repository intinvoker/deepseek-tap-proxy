"""ClaudeCodePlugin。"""
from __future__ import annotations

from deepseek_tap_proxy.pipeline import Stage, Context
from deepseek_tap_proxy.plugins.base import DetectionResult
from deepseek_tap_proxy.plugins.claude_code.detect import detect_claude_code
from deepseek_tap_proxy.shared.stages import SysToolsLockStage


class ClaudeCodePlugin:
    name = "claude-code"
    priority = 7

    def detect(self, ctx: Context) -> DetectionResult | None:
        return detect_claude_code(ctx)

    def build_stages(self, ctx: Context) -> list[Stage]:
        return [
            SysToolsLockStage(),
        ]
