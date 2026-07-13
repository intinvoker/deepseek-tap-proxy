"""Anthropic 格式管道配置。"""
from __future__ import annotations

from deepseek_tap_proxy.pipeline import Pipeline
from deepseek_tap_proxy.anthropic.stages import (
    AnthropicParseStage,
    AnthropicBlock0ClearStage,
    AnthropicSystemRoleConvertStage,
    AnthropicUpstreamPathStage,
    AnthropicForwardStage,
    ReturnStage,
)
from deepseek_tap_proxy.shared.stages import CanonicalStage

ANTHROPIC_FORMAT_PIPELINE = Pipeline([
    AnthropicParseStage(),
    AnthropicBlock0ClearStage(),
    CanonicalStage(),
])

ANTHROPIC_FORMAT_TAIL = [
    AnthropicSystemRoleConvertStage(),
    AnthropicUpstreamPathStage(),
    AnthropicForwardStage(),
    ReturnStage(),
]
