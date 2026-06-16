"""ChainAction — PrefixLock 的链操作决策，persist 只读此对象不做独立决策。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChainAction:
    action: str = ""                           # "create" | "update" | "none"
    chain_id: str | None = None
    msg_count: int = 0
    sys_hashes: str = ""
    tools_hashes: str = ""
    messages_hashes: str = ""
    sys: list[dict] | None = None
    tools: list[dict] | None = None
    messages: list[dict] | None = None
    safe_breakpoints: list[int] | None = None
    hit_count: int = 0                        # 三段总命中数
    sys_hit: int = 0
    tools_hit: int = 0
    chain_tokens: int | None = None
    char_count: int = 0                       # canonical 消息的字符总数
