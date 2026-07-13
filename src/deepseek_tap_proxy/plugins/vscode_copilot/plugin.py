"""VSCodeCopilotPlugin — Agent Copilot 原生 Provider（sys 非空）。"""
from __future__ import annotations

from deepseek_tap_proxy.pipeline import Stage, Context
from deepseek_tap_proxy.plugins.base import DetectionResult
from deepseek_tap_proxy.plugins.vscode_copilot.detect import detect_vscode_copilot
from deepseek_tap_proxy.shared.stages import (
    SysToolsLockStage, PrefixLockStage, CompactionLockStage,
)
from deepseek_tap_proxy.plugins.vscode_copilot.stages.compaction import CompactionRewrite


class VSCodeCopilotPlugin:
    name = "vscode-copilot"
    priority = 10

    def detect(self, ctx: Context) -> DetectionResult | None:
        return detect_vscode_copilot(ctx)

    def build_stages(self, ctx: Context) -> list[Stage]:
        stages: list[Stage] = [
            SysToolsLockStage(),
            PrefixLockStage(),
            CompactionLockStage(),
        ]
        if ctx.chat_type == "compaction":
            stages.append(CompactionRewrite())
        return stages
