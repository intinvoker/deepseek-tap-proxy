"""Runtime — 所有 stages + handler 需要的强类型运行时依赖。

替代 app["proxy_ctx"] / app["store"] / app["config"] / request["upstream"] 等弱引用。
通过 request["runtime"] 注入，兼容 aiohttp 的 dict 式接口和 _ReplayRequest。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepseek_tap_proxy.shared.store import MemoryStore


@dataclass
class Runtime:
    """强类型运行时依赖。通过 request["runtime"] 注入。

    使用时:
        rt = ctx.request.get("runtime")
        store = rt.store if rt else None
        config = rt.config if rt else None
    """
    store: "MemoryStore | None" = None
    config: object | None = None            # ProxyConfig — 避免循环引用
    transport: object | None = None          # Transport
    router: object | None = None             # UpstreamRouter
    policies: object | None = None           # PolicyResolver（passthrough/adapter/managed）
    upstream_config: object | None = None    # ResolvedTarget
    pipeline_factory: dict | None = None    # {"generic": Pipeline} 或类似
    session: object | None = None            # aiohttp.ClientSession — 迁移过渡
    run_id: str = ""
    otlp_writer: object | None = None        # OtlpLogWriter — Consumer 依赖
