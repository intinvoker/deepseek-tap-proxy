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
    session_method: str = ""     # uuid | match | create — 解析方式
    client_type: str = ""        # copilot | copilot-agent | hermes | claude-code | generic
    chat_type: str = ""          # normal | compaction | subagent

    # ── 请求元数据 ──
    prompt_messages_count: int = 0
    prompt_tools_count: int = 0
    msg_hash_chain: str = ""

    # ── 前缀锁定 ──
    lock_type: str = ""
    prefix_chain_id: str = ""

    # ── 链操作（引用 chain_action，不拷贝重量数据） ──
    # ⚠️ 与 ctx.chain_id（Context 上的身份字段）同名不同义 ——
    #    本字段当前是「前缀链」标识，由 sys/tools/messages 哈希派生；
    #    而 ctx.chain_id 是「请求身份」，由适配器从外部协议转化。
    #    收敛后本字段改由身份写入；在此之前两者
    #    在一次请求上取值不同，写入/读取时必须指明是哪一个。
    chain_id: str = ""
    chain_action: str = ""         # "create" | "update" | ""
    chain_msg_count: int = 0
    chain_tokens: int = 0
    chain_ref: Optional['ChainAction'] = None  # 引用，不拷贝，persist 函数内部读

    # ═══════════════════════════════════════════════════════════════
    #  Modification trace — 每列对应一个 Stage 的操作结果。
    #  规则：枚举值（≤4 个），禁止放数字/摘要。详细数据见 OTLP / JSONL。
    # ═══════════════════════════════════════════════════════════════
    is_modified: bool = False
    modification_note: str = ""          # 人类可读（禁止 SQL 查询依赖此字段）
    system_role_converted: bool = False  # AnthropicSystemRoleConvertStage
    sys_tools_locked: str = ""           # SysToolsLockStage: init|pass|lock|skip

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

    # ── 请求分类 ──
    topic: str = ""                  # "chat" | "count_tokens" | "passthrough" | "rejected" | ""

    # ── 模式 / 端点类型 / 干预 / usage 可信度（passthrough-policy 引入）──
    # mode: passthrough | adapter | managed —— 决定走哪条管线；
    # kind: chat | search | fetch | other —— 端点语义（搜索/抓取新 API 留痕用）；
    # intervention: 是否对请求做过改写（与 is_modified 同源，见 DB 列）；
    # usage_source: parsed | unparsed —— usage 解析是否成功（尽力而为）。
    mode: str = "managed"
    kind: str = ""
    usage_source: str = ""
