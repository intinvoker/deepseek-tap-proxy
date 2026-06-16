"""Context — 请求/响应数据载体，在 pipeline stages 之间流转。

Canonical form（格式无关视图）::

    ctx.sys       — system prompt messages
    ctx.tools     — tool definitions
    ctx.messages  — conversation messages (user/assistant/tool)

这些是 *properties*，底层由 ``fwd_body`` 支持。读取时从 API 特定结构提取，
写入时透明映射回去。Pipeline stages 操作这些属性，而不是直接接触 ``fwd_body``。
"""
from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field

import xxhash
from aiohttp import web
from typing import Optional

from deepseek_tap_proxy.shared.chain import Chain, ChainMoment
from deepseek_tap_proxy.shared.persist import PersistRecord
from deepseek_tap_proxy.shared.hash import msg_hash
from deepseek_tap_proxy.pipeline.chain_action import ChainAction
from deepseek_tap_proxy.pipeline.snapshot import SysToolsSnapshot

REQUEST_ID_LENGTH = 12


@dataclass
class Context:
    """请求/响应载体，流经 pipeline 各 stage。"""

    request: web.Request
    body_readonly: dict | None = None   # 只读：ParseStage 设置后永不修改
    fwd_body: dict | None = None
    fwd_headers: dict = field(default_factory=dict)
    req_id: str = ""
    format: str = ""                    # "openai" | "anthropic"

    target_url: str = ""
    upstream_url: str = ""
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
    chain_action: Optional['ChainAction'] = None
    sys_tools_snap: SysToolsSnapshot = field(default_factory=SysToolsSnapshot)
    ext: dict = field(default_factory=dict)
    persist: PersistRecord = field(default_factory=PersistRecord)

    # ── 请求身份（RequestAnalysisStage 设置）──
    client_type: str = ""
    chat_type: str = ""
    session_id: str = ""
    session_method: str = ""

    # ── 规范形式分段长度（view setter 自动刷新）──
    orig_sys_len: int = 0
    orig_tools_len: int = 0
    orig_msgs_len: int = 0

    # ── 三段 hash（原生 list[str]，view setter 自动刷新）──
    _sys_hashes: list[str] = field(default_factory=list)
    _tools_hashes: list[str] = field(default_factory=list)
    _messages_hashes: list[str] = field(default_factory=list)

    # ── 分支族标识（sys_hashes + tools_hashes → streaming xxh3_64）──
    family_key: str = ""

    # ── 链匹配结果（PrefixLockStage 设置）──
    matched_chain: Chain | None = None
    anchor_moment: ChainMoment | None = None
    new_char_count: int = 0

    # ── 内部：是否已激活 view ──
    _view_activated: bool = False

    def __post_init__(self) -> None:
        self._t0 = time.monotonic()
        self.req_id = f"req_{uuid.uuid4().hex[:REQUEST_ID_LENGTH]}"
        if self.fwd_body is None and self.body_readonly is not None:
            self.fwd_body = copy.deepcopy(self.body_readonly)

    @property
    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    # -- topic --------------------------------------------------------

    @property
    def topic(self) -> str:
        return self.persist.topic

    @topic.setter
    def topic(self, value: str) -> None:
        self.persist.topic = value

    # ═══════════════════════════════════════════════════════════════
    #  Canonical form: sys / tools / messages
    # ═══════════════════════════════════════════════════════════════

    # -- sys --------------------------------------------------------

    @property
    def sys(self) -> list[dict]:
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
            else:
                self.fwd_body["system"] = [
                    {"type": "text", "text": m.get("content", "")}
                    for m in value
                ]
        self._on_sys_changed()

    # -- tools ------------------------------------------------------

    @property
    def tools(self) -> list[dict]:
        raw = (self.fwd_body or {}).get("tools", []) or []
        return list(raw)

    @tools.setter
    def tools(self, value: list[dict]):
        if self.fwd_body is not None:
            self.fwd_body["tools"] = value
        self._on_tools_changed()

    # -- messages ---------------------------------------------------

    @property
    def messages(self) -> list[dict]:
        if not self.fwd_body:
            return []
        if self.format == "openai":
            msgs = self.fwd_body.get("messages", [])
            if msgs and msgs[0].get("role") == "system":
                return msgs[1:]
            return list(msgs)
        elif self.format == "anthropic":
            return list(self.fwd_body.get("messages", []))
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
        self._on_messages_changed()

    # -- hash: list[str] (原生) + chain str (派生，DB/JSONL 用) --------

    @property
    def sys_hashes(self) -> list[str]:
        return self._sys_hashes

    @property
    def tools_hashes(self) -> list[str]:
        return self._tools_hashes

    @property
    def messages_hashes(self) -> list[str]:
        return self._messages_hashes

    @property
    def sys_hash_chain(self) -> str:
        return ",".join(self._sys_hashes) if self._sys_hashes else ""

    @property
    def tools_hash_chain(self) -> str:
        return ",".join(self._tools_hashes) if self._tools_hashes else ""

    @property
    def messages_hash_chain(self) -> str:
        return ",".join(self._messages_hashes) if self._messages_hashes else ""

    @property
    def msg_hash_chain(self) -> str:
        return ",".join(
            p for p in [self.sys_hash_chain, self.tools_hash_chain, self.messages_hash_chain] if p
        )

    # -- 内部：view → hash/count 自动刷新 -----------------------------

    def _on_sys_changed(self):
        s = self.sys
        self.orig_sys_len = len(s)
        self._sys_hashes = [msg_hash(m) for m in s]
        self._recompute_family_key()

    def _on_tools_changed(self):
        t = self.tools
        self.orig_tools_len = len(t)
        self._tools_hashes = [msg_hash(m) for m in t]
        self._recompute_family_key()

    def _on_messages_changed(self):
        m = self.messages
        self.orig_msgs_len = len(m)
        self._messages_hashes = [msg_hash(m) for m in m]

    def _recompute_family_key(self):
        """Streaming xxh3_64 over sys_hashes + tools_hashes → 16-char hex."""
        h = xxhash.xxh3_64()
        for s in self._sys_hashes:
            h.update(s.encode())
        for t in self._tools_hashes:
            h.update(t.encode())
        self.family_key = h.hexdigest()[:16] if (self._sys_hashes or self._tools_hashes) else ""

    def _activate_view(self):
        """首次激活 canonical view：计算全部 hash/count + family_key。"""
        self._on_sys_changed()
        self._on_tools_changed()
        self._on_messages_changed()
        self._view_activated = True

    # ═══════════════════════════════════════════════════════════════
    #  Response canonical view
    # ═══════════════════════════════════════════════════════════════

    @property
    def resp_usage(self) -> dict:
        if not self.resp_body:
            return {}
        from deepseek_tap_proxy.shared.usage import normalize_usage
        return normalize_usage(self.resp_body.get("usage", {}), self.format)

    @property
    def resp_model(self) -> str:
        if self.resp_body:
            m = self.resp_body.get("model")
            if m:
                return m
        return (self.body_readonly or {}).get("model", "")

    @property
    def req_model(self) -> str:
        """原始请求中的模型名（不受上游响应影响）。"""
        return (self.body_readonly or {}).get("model", "")

    @property
    def resp_has_tool_calls(self) -> bool:
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
