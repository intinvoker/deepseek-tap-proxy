"""VSCodeCopilotDeepSeekPlugin — Agent Copilot BYOK 适配器（sys 空）。"""
from __future__ import annotations

from deepseek_tap_proxy.pipeline import Stage, Context
from deepseek_tap_proxy.plugins.base import DetectionResult
from deepseek_tap_proxy.plugins.vscode_copilot_deepseek.detect import detect_vscode_copilot_deepseek
from deepseek_tap_proxy.shared.stages import (
    SysToolsLockStage, PrefixLockStage, CompactionLockStage,
)
from deepseek_tap_proxy.plugins.vscode_copilot_deepseek.stages.compaction import CompactionRewrite


class VSCodeCopilotDeepSeekPlugin:
    name = "vscode-copilot-deepseek"
    priority = 9

    def detect(self, ctx: Context) -> DetectionResult | None:
        return detect_vscode_copilot_deepseek(ctx)

    def build_stages(self, ctx: Context) -> list[Stage]:
        stages: list[Stage] = [
            # ContextNormStage 已移除 — 对命中率无改善，见 context_norm.py 注释
            SysToolsLockStage(),
            PrefixLockStage(cross_family_fallback=True),
            CompactionLockStage(),
        ]
        if ctx.chat_type == "compaction":
            stages.append(CompactionRewrite())
        return stages
