"""DshPlugin。"""
from __future__ import annotations

from deepseek_tap_proxy.pipeline import Context, Stage
from deepseek_tap_proxy.plugins.base import DetectionResult
from deepseek_tap_proxy.plugins.dsh.detect import detect_dsh
from deepseek_tap_proxy.plugins.dsh.stages.compat import DshCompatStage


class DshPlugin:
    name = "dsh"
    # 低于 claude-code(7)：DSH 只按 UA 命中，不与既有客户端的 prompt 指纹竞争。
    priority = 6

    def detect(self, ctx: Context) -> DetectionResult | None:
        return detect_dsh(ctx)

    def build_stages(self, ctx: Context) -> list[Stage]:
        # 只做协议适配，不挂缓存锁（见 stages/compat.py 顶部说明）。
        return [DshCompatStage()]

    def build_adapter_stages(self, ctx: Context) -> list[Stage]:
        """adapter 模式（passthrough + 显式白名单适配）：仅协议兼容，不挂锁。"""
        return [DshCompatStage()]
