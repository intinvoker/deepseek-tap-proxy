"""Plugin 注册表 — 检测链和查询函数。"""
from __future__ import annotations

from deepseek_tap_proxy.plugins.base import Plugin, DetectionResult
from deepseek_tap_proxy.plugins.vscode_copilot.plugin import VSCodeCopilotPlugin
from deepseek_tap_proxy.plugins.vscode_copilot_deepseek.plugin import VSCodeCopilotDeepSeekPlugin
from deepseek_tap_proxy.plugins.hermes.plugin import HermesPlugin
from deepseek_tap_proxy.plugins.claude_code.plugin import ClaudeCodePlugin
from deepseek_tap_proxy.plugins.dsh.plugin import DshPlugin

ALL_PLUGINS: list[Plugin] = [
    VSCodeCopilotPlugin(),
    VSCodeCopilotDeepSeekPlugin(),
    HermesPlugin(),
    ClaudeCodePlugin(),
    DshPlugin(),
]


def detect_plugin(ctx, plugins=None):
    """遍历检测链，返回 (plugin, result) 或 (None, None)。"""
    for p in sorted(plugins or ALL_PLUGINS, key=lambda x: x.priority, reverse=True):
        result = p.detect(ctx)
        if result is not None:
            return p, result
    return None, None
