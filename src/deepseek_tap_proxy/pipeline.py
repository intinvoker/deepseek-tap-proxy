"""Pipeline infrastructure — format-agnostic execution engine.

Context carries request/response data through stages. The canonical form
(``sys`` / ``tools`` / ``messages``) provides a format-agnostic view
backed by ``fwd_body``.  Property accessors transparently map reads and
writes to the appropriate API-specific fields::

    Format       sys location          messages location
    ──────       ────────────          ─────────────────
    openai        messages[0]           messages[1:]
    anthropic     system field          messages[]

All pipeline stages operate on the canonical form — they don't need to
know whether the request arrived as OpenAI or Anthropic format.
"""

from __future__ import annotations

import copy
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from aiohttp import web
from typing import Optional

from deepseek_tap_proxy.shared.persist import PersistRecord


@dataclass
class ChainAction:
    """PrefixLock 的链操作决策——persist 只读此对象，不做独立决策。"""
    action: str                           # "create" | "update" | "none"
    chain_id: str | None = None
    msg_count: int = 0
    hash_chain: str = ""
    sys: list[dict] | None = None
    tools: list[dict] | None = None
    messages: list[dict] | None = None
    safe_breakpoints: list[int] | None = None
    hit_count: int = 0
    chain_tokens: int | None = None
    char_count: int = 0                   # canonical 消息的字符总数
    is_frozen: bool = False               # 公共前缀链（冻结，不 UPDATE）


REQUEST_ID_LENGTH = 12


@dataclass
class Context:
    """Request/response carrier that flows through pipeline stages.

    Canonical form (format-agnostic view)::

        ctx.sys       — system prompt messages
        ctx.tools     — tool definitions
        ctx.messages  — conversation messages (user / assistant / tool)

    These are *properties* backed by ``fwd_body``.  Reading from them
    extracts from the API-specific structure; writing to them maps back
    transparently.  Pipeline stages operate on these, never on
    ``fwd_body`` directly.
    """

    request: web.Request
    body_readonly: dict | None = None  # 只读：ParseStage 设置后永不修改。如需修改说明你写错了地方。
    fwd_body: dict | None = None
    fwd_headers: dict = field(default_factory=dict)
    req_id: str = ""
    format: str = ""  # "openai" | "anthropic" — set by ParseStage

    target_url: str = ""
    upstream_url: str = ""  # Final upstream URL (set by UpstreamPathStage)
    is_streaming: bool = False
    timeout: int = 600

    status: int = 0
    resp_body: dict | None = None  # 原始响应体（格式相关），由 ForwardStage 设置
    resp_bytes: bytes = b""
    resp_headers: dict = field(default_factory=dict)

    should_abort: bool = False
    response: web.StreamResponse | None = None

    duration_ms: int = 0
    _t0: float = 0.0

    meta: dict = field(default_factory=dict)
    chain_action: Optional['ChainAction'] = None  # PrefixLock 产出，persist 消费
    ext: dict = field(default_factory=dict)  # 扩展 view（同名映射 from fwd_body）
    persist: PersistRecord = field(default_factory=PersistRecord)

    def __post_init__(self) -> None:
        self._t0 = time.monotonic()
        self.req_id = f"req_{uuid.uuid4().hex[:REQUEST_ID_LENGTH]}"
        if self.fwd_body is None and self.body_readonly is not None:
            self.fwd_body = copy.deepcopy(self.body_readonly)

    @property
    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    # ═══════════════════════════════════════════════════════════════
    #  Canonical form: sys / tools / messages
    #
    #  These properties are the *only* way pipeline stages should
    #  access message content.  They transparently map to the
    #  API-specific fwd_body structure.
    # ═══════════════════════════════════════════════════════════════

    # -- sys --------------------------------------------------------

    @property
    def sys(self) -> list[dict]:
        """System prompt — format-agnostic canonical view."""
        if not self.fwd_body:
            return []
        if self.format == "openai":
            msgs = self.fwd_body.get("messages", [])
            if msgs and msgs[0].get("role") == "system":
                return [msgs[0]]
            return []
        elif self.format == "anthropic":
            system = self.fwd_body.get("system", "")
            if isinstance(system, str):
                return [{"role": "system", "content": system}] if system else []
            elif isinstance(system, list):
                result = []
                for item in system:
                    if isinstance(item, dict) and item.get("type") == "text":
                        txt = item.get("text", "")
                        if txt:
                            result.append({"role": "system", "content": txt})
                return result
            return []
        return []

    @sys.setter
    def sys(self, value: list[dict]):
        if not self.fwd_body:
            return
        if self.format == "openai":
            msgs: list = self.fwd_body.setdefault("messages", [])
            if msgs and msgs[0].get("role") == "system":
                if value:
                    msgs[0] = value[0]
                else:
                    msgs.pop(0)
            elif value:
                msgs.insert(0, value[0])
        elif self.format == "anthropic":
            if not value:
                self.fwd_body.pop("system", None)
            elif len(value) == 1:
                # Single system message → store as string (simple case)
                self.fwd_body["system"] = value[0].get("content", "")
            else:
                # Multiple system blocks → store as list of text blocks
                self.fwd_body["system"] = [
                    {"type": "text", "text": m.get("content", "")}
                    for m in value
                ]

    # -- tools ------------------------------------------------------

    @property
    def tools(self) -> list[dict]:
        """Tool definitions — format-agnostic canonical view."""
        return (self.fwd_body or {}).get("tools", []) or []

    @tools.setter
    def tools(self, value: list[dict]):
        if self.fwd_body is not None:
            self.fwd_body["tools"] = value

    # -- messages ---------------------------------------------------

    @property
    def messages(self) -> list[dict]:
        """Conversation messages (user/assistant/tool) — WITHOUT sys."""
        if not self.fwd_body:
            return []
        if self.format == "openai":
            msgs = self.fwd_body.get("messages", [])
            if msgs and msgs[0].get("role") == "system":
                return msgs[1:]
            return list(msgs)
        elif self.format == "anthropic":
            return self.fwd_body.get("messages", [])
        return []

    @messages.setter
    def messages(self, value: list[dict]):
        if not self.fwd_body:
            return
        if self.format == "openai":
            existing = self.fwd_body.get("messages", [])
            if existing and existing[0].get("role") == "system":
                self.fwd_body["messages"] = [existing[0]] + list(value)
            else:
                self.fwd_body["messages"] = list(value)
        elif self.format == "anthropic":
            self.fwd_body["messages"] = list(value)

    # ═══════════════════════════════════════════════════════════════
    #  Response canonical view: format-agnostic response properties
    #
    #  Read-only views over ``resp_body`` — the raw upstream response.
    #  Pipeline stages should use these instead of digging into
    #  format-specific response structures.
    # ═══════════════════════════════════════════════════════════════

    @property
    def resp_usage(self) -> dict:
        """归一化 usage：{input_tokens, output_tokens, cache_read_input_tokens, ...}"""
        if not self.resp_body:
            return {}
        from deepseek_tap_proxy.shared.usage import normalize_usage
        return normalize_usage(
            self.resp_body.get("usage", {}),
            self.format,
        )

    @property
    def resp_model(self) -> str:
        """模型名（来自响应体，回退到请求体）。"""
        if self.resp_body:
            m = self.resp_body.get("model")
            if m:
                return m
        return (self.body_readonly or {}).get("model", "")

    @property
    def resp_has_tool_calls(self) -> bool:
        """响应中是否有 tool_calls（OpenAI）或 tool_use（Anthropic）。"""
        if not self.resp_body:
            return False
        if self.format == "openai":
            choices = self.resp_body.get("choices", [])
            if choices:
                return bool(choices[0].get("message", {}).get("tool_calls"))
            return False
        elif self.format == "anthropic":
            content = self.resp_body.get("content", [])
            return any(c.get("type") == "tool_use" for c in content)
        return False

    @property
    def resp_has_reasoning(self) -> bool:
        """响应中是否有 reasoning_content / thinking 内容。"""
        if not self.resp_body:
            return False
        if self.format == "openai":
            choices = self.resp_body.get("choices", [])
            if choices:
                return bool(choices[0].get("message", {}).get("reasoning_content"))
            return False
        elif self.format == "anthropic":
            content = self.resp_body.get("content", [])
            return any(c.get("type") == "thinking" for c in content)
        return False


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
