"""Plugin 基类 — client type 检测与 pipeline 构建。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

from deepseek_tap_proxy.pipeline import Stage, Context


@dataclass(frozen=True)
class DetectionResult:
    """纯检测产出。session_id 始终从请求内容提取。

    chain_id / parent_chain_id 为内部身份字段 —— 由各客户端适配器
    从外部协议转化而来，服务内一律使用这两个名字。
    """
    client_type: str
    chat_type: str
    session_id: str
    chain_id: str = ""
    parent_chain_id: str = ""


class Plugin(Protocol):
    """每种 client type 对应一个 Plugin。"""
    name: str
    priority: int

    def detect(self, ctx: Context) -> DetectionResult | None:
        """纯检测，只读 ctx。返回 None 表示不匹配。"""
        ...

    def build_stages(self, ctx: Context) -> list[Stage]:
        """返回管道2 中间段（共享策略 + client 独有 stage）。
        不含 Forward/Return — 由 Handler 追加。"""
        ...
