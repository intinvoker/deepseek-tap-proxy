"""HermesPlugin。"""
from __future__ import annotations

from deepseek_tap_proxy.pipeline import Stage, Context
from deepseek_tap_proxy.plugins.base import DetectionResult
from deepseek_tap_proxy.plugins.hermes.detect import detect_hermes
from deepseek_tap_proxy.shared.stages import SysToolsLockStage, PrefixLockStage


class HermesPlugin:
    name = "hermes"
    priority = 8

    def detect(self, ctx: Context) -> DetectionResult | None:
        return detect_hermes(ctx)

    def build_stages(self, ctx: Context) -> list[Stage]:
        return []  # Hermes 不需要 Lock stage
