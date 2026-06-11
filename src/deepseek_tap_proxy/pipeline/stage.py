"""Stage + Pipeline + PipelineFactory — pipeline 执行引擎。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from deepseek_tap_proxy.pipeline.context import Context


class Stage(ABC):
    """Pipeline 中的单个处理步骤。"""

    @abstractmethod
    async def process(self, ctx: Context) -> Context:
        ...


class Pipeline:
    """有序的 stage 序列，请求在此流经。"""

    def __init__(self, stages: list[type[Stage]] | list[Stage],
                 skip_set: set[type[Stage]] | None = None) -> None:
        skip = skip_set or set()
        built: list[Stage] = []
        for s in stages:
            if isinstance(s, type):
                if s not in skip:
                    built.append(s())
            else:
                built.append(s)
        self.stages = built

    async def execute(self, ctx: Context) -> Context:
        for stage in self.stages:
            ctx = await stage.process(ctx)
            if ctx.should_abort:
                break
        return ctx


class PipelineFactory:
    """启动时构建所有 pipeline variant，请求时按 client_type 查表。

    Usage:
        factory = PipelineFactory()
        factory.build_all(FULL_STAGES, {"copilot": {SkipA}, "generic": {SkipA, SkipB}})
        pipeline = factory.get("copilot")
        pipeline = factory.get("unknown")   # → 回退 generic
    """

    def __init__(self):
        self._pipelines: dict[str, Pipeline] = {}

    def build_all(self, full: list[type[Stage]],
                  skip_table: dict[str, set[type[Stage]]]) -> None:
        for client_type, skip_set in skip_table.items():
            self._pipelines[client_type] = Pipeline(full, skip_set=skip_set)

    def get(self, client_type: str) -> Pipeline:
        p = self._pipelines.get(client_type)
        if p is not None:
            return p
        p = self._pipelines.get("generic")
        if p is not None:
            return p
        raise RuntimeError("PipelineFactory: no 'generic' variant configured")
