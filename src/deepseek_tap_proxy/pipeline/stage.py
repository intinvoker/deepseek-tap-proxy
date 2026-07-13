"""Stage + Pipeline + PipelineObserver + PipelineFactory — pipeline 执行引擎。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from deepseek_tap_proxy.pipeline.context import Context


class Stage(ABC):
    """Pipeline 中的单个处理步骤。"""

    @abstractmethod
    async def process(self, ctx: Context) -> Context:
        ...


class PipelineObserver(Protocol):
    """Pipeline 执行观察者 — 纯数据采集，不参与也不影响 pipeline 执行。

    Inspector 实现此协议，在每个 stage 前后采集 ctx 快照。
    生产环境传 None，零开销。
    """

    def on_stage_start(self, stage: Stage, ctx: Context) -> None:
        """stage.process(ctx) 调用前触发。"""

    def on_stage_end(self, stage: Stage, ctx: Context) -> None:
        """stage.process(ctx) 成功返回后触发。ctx 是 stage 修改后的状态。"""

    def on_stage_error(self, stage: Stage, ctx: Context, error: Exception) -> None:
        """stage.process(ctx) 抛出异常时触发。"""

    def on_pipeline_end(self, ctx: Context) -> None:
        """所有 stage 执行完毕后触发。"""


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

    async def execute(self, ctx: Context,
                      observer: PipelineObserver | None = None) -> Context:
        """执行所有 stage。observer 非空时在每个 stage 的前后触发 hook。"""
        for stage in self.stages:
            if observer:
                observer.on_stage_start(stage, ctx)
            try:
                ctx = await stage.process(ctx)
            except Exception as e:
                if observer:
                    observer.on_stage_error(stage, ctx, e)
                raise
            if observer:
                observer.on_stage_end(stage, ctx)
            if ctx.should_abort:
                break
        if observer:
            observer.on_pipeline_end(ctx)
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
