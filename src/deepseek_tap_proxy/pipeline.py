"""Pipeline infrastructure — format-agnostic execution engine.

Context carries request/response data through stages. Format-specific
metadata is stored in the ``meta`` dict.
"""

from __future__ import annotations

import copy
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from aiohttp import web

REQUEST_ID_LENGTH = 12


@dataclass
class Context:
    """Request/response carrier that flows through pipeline stages."""

    request: web.Request
    raw_body: bytes = b""
    req_body: dict | None = None
    fwd_body: dict | None = None
    fwd_headers: dict = field(default_factory=dict)
    req_id: str = ""

    target_url: str = ""
    upstream_url: str = ""  # Final upstream URL (set by UpstreamPathStage)
    is_streaming: bool = False
    timeout: int = 600

    status: int = 0
    resp_body: dict | None = None
    resp_bytes: bytes = b""
    resp_headers: dict = field(default_factory=dict)

    should_abort: bool = False
    response: web.StreamResponse | None = None

    duration_ms: int = 0
    _t0: float = 0.0

    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._t0 = time.monotonic()
        self.req_id = f"req_{uuid.uuid4().hex[:REQUEST_ID_LENGTH]}"
        if self.fwd_body is None and self.req_body is not None:
            self.fwd_body = copy.deepcopy(self.req_body)

    @property
    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)


class Stage(ABC):
    """Single processing step in the pipeline."""

    @abstractmethod
    async def process(self, ctx: Context) -> Context:
        ...


class Pipeline:
    """Ordered sequence of stages through which a request flows."""

    def __init__(self, stages: list[Stage]) -> None:
        self.stages = stages

    async def execute(self, ctx: Context) -> Context:
        for stage in self.stages:
            ctx = await stage.process(ctx)
            if ctx.should_abort:
                break
        return ctx
