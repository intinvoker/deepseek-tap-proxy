from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ChainMoment:
    """一次 API 请求的快照。对应 TUI ChainMoment。"""
    messages_count: int
    op: str                       # "create" | "update" | "compact"
    total_tokens: int = 0         # 上游 input_tokens（真实值）
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    char_count: int = 0           # 此时消息的 char_count
    timestamp: str = ""


@dataclass
class Chain:
    """前缀链实体。对应 TUI Chain。"""
    chain_id: str
    session_id: str
    sys: list[dict]
    tools: list[dict]
    messages: list[dict]
    created_at: str
    moments: list[ChainMoment] = field(default_factory=list)
    # Hash chain data for prefix matching (comma-separated strings)
    sys_hashes: str = ""
    tools_hashes: str = ""
    messages_hashes: str = ""
