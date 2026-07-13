"""OpenAI 格式管道配置。"""
from __future__ import annotations

from deepseek_tap_proxy.pipeline import Pipeline
from deepseek_tap_proxy.openai.stages import ParseStage, ForwardStage, ReturnStage
from deepseek_tap_proxy.shared.stages import CanonicalStage

OPENAI_FORMAT_PIPELINE = Pipeline([
    ParseStage(),
    CanonicalStage(),
])

OPENAI_FORMAT_TAIL = [
    ForwardStage(),
    ReturnStage(),
]
