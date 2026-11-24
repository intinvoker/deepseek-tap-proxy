from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ChainMoment:
    """一次 API 请求的快照。"""
    messages_count: int
    op: str                       # "create" | "update" | "compact"
    total_tokens: int = 0         # 上游 input_tokens（真实值）
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    char_count: int = 0           # 此时消息的 char_count
    timestamp: str = ""


@dataclass
class Chain:
    """前缀链实体。"""
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


def find_anchor_moment(moments: list[ChainMoment], pref_hit: int) -> ChainMoment | None:
    """找到 messages_count ≤ pref_hit 的最新 Moment。

    用于 est 计算：moment 的全部消息都被当前前缀覆盖时，
    moment.total_tokens ≈ 当前请求的缓存命中 token。

    moments 必须按 messages_count 升序排列。
    """
    best = None
    for m in moments:
        if m.messages_count <= pref_hit:
            best = m
        else:
            break
    return best
