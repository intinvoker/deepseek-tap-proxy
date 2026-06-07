"""Persist data contract — unified data structure written to DB.

All pipeline stages write persistence-bound data to ``ctx.persist``.
Persist functions read from ``ctx.persist``.  This is the only data
path to the database — no stage writes ``ctx.meta`` keys for persist
purposes, and no persist function reads ``ctx.meta`` directly.

Layers
    Raw Data  → View (normalized)  → Persist (snapshot contract)  → DB
    Raw Data  → JSONL (raw archive, persist meta only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from deepseek_tap_proxy.pipeline import ChainAction


@dataclass
class PersistRecord:
    """Persist snapshot contract — accumulated during pipeline, frozen at persist time.

    Stage writes field → PersistRecord (pipeline R/W)
    Persist reads field → PersistRecord → DB (post-pipeline R/O)
    """
    # ── 请求标识 ──
    request_id: str = ""
    api_format: str = ""           # "openai" | "anthropic"
    model: str = ""
    stream_mode: bool = False
    timestamp: str = ""

    # ── 会话 ──
    session_id: str = ""
    session_type: str = ""         # match | uuid | create
    session_fingerprint: str = ""

    # ── 请求元数据 ──
    prompt_messages_count: int = 0
    prompt_tools_count: int = 0
    msg_hash_chain: str = ""

    # ── 前缀锁定 ──
    lock_type: str = ""
    prefix_chain_id: str = ""

    # ── 链操作（引用 chain_action，不拷贝重量数据） ──
    chain_id: str = ""
    chain_action: str = ""         # "create" | "update" | ""
    chain_msg_count: int = 0
    chain_tokens: int = 0
    chain_ref: Optional['ChainAction'] = None  # 引用，不拷贝，persist 函数内部读

    # ── 修改标记 ──
    is_modified: bool = False
    modification_note: str = ""

    # ── 归一化 Usage（apply_cache_analysis 写入） ──
    total_input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_hit_ratio: float = 0.0
    has_reasoning: bool = False
    has_tool_calls: bool = False

    # ── Anthropic 专属（始终可选） ──
    trace_category: str = ""
    system_role_converted: bool = False
