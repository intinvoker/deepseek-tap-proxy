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
import json
import orjson
import re
import time
import uuid
from dataclasses import dataclass, field

import xxhash
from aiohttp import web
from typing import Optional

from deepseek_tap_proxy.shared.chain import Chain, ChainMoment
from deepseek_tap_proxy.shared.persist import PersistRecord
from deepseek_tap_proxy.pipeline.chain_action import ChainAction
from deepseek_tap_proxy.pipeline.snapshot import SysToolsSnapshot

REQUEST_ID_LENGTH = 12

# 外部客户端外传的请求级关联 id。合法形态 = 32 位 hex（大小写不敏感，trim 后），
# 与客户端 RequestId（Guid "N" 格式）同构；proxy 直接复用，实现两端日志按同一键关联。
EXTERNAL_REQUEST_ID_HEADER = "x-tap-request-id"
_EXTERNAL_REQUEST_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")


def external_request_id(request) -> str:
    """读请求头 ``x-tap-request-id``（客户端外传）→ 统一小写；非法/缺失返回 ""。

    宽松防御：请求对象可能没有 headers（单测用 dict/MagicMock），头值可能非 str——
    任何异常形态都回退到生成逻辑，绝不因坏头阻断转发。

    与 ``resolve_request_id`` 同为请求 id 解析唯一入口（Context 与 Responses
    passthrough handler 共用，避免两处形态判定漂移）。
    """
    headers = getattr(request, "headers", None)
    if headers is None:
        return ""
    try:
        raw = headers.get(EXTERNAL_REQUEST_ID_HEADER)
    except (AttributeError, TypeError):
        return ""
    if not isinstance(raw, str):
        return ""
    raw = raw.strip()
    if not _EXTERNAL_REQUEST_ID_RE.match(raw):
        return ""
    return raw.lower()


def resolve_request_id(request) -> str:
    """请求 id 唯一解析入口：合法外传 id 优先，否则生成 ``req_<uuid12>``。"""
    return external_request_id(request) or (
        f"req_{uuid.uuid4().hex[:REQUEST_ID_LENGTH]}")


class MessageIndex:
    """dict 对象 → (hash, char_count) 的附属映射。

    同一个 dict 对象只做一次 json.dumps，后续零开销读取。
    使用 id(dict) 索引，同时持有 dict 引用防止 id 复用。
    生命周期 = 一次 HTTP 请求（挂在 Context 上，请求结束自动释放）。
    """

    def __init__(self):
        # id → (msg_ref, hash, char_count)
        # msg_ref 保持引用，防止 id 复用导致碰撞
        self._idx: dict[int, tuple[dict, str, int]] = {}
        self.hits: int = 0       # 缓存命中次数
        self.misses: int = 0     # json.dumps 次数

    def get(self, msg: dict) -> tuple[str, int]:
        """返回 (hash_hex, char_count)。首次 json.dumps，后续读缓存。"""
        mid = id(msg)
        entry = self._idx.get(mid)
        if entry is not None and entry[0] is msg:
            self.hits += 1
            return entry[1], entry[2]
        # 新 dict 或 id 碰撞（极罕见）：计算并缓存
        # hash 前剥离 cache_control（DeepSeek 忽略该字段——对齐 msg_hash 语义）
        self.misses += 1
        from deepseek_tap_proxy.shared.hash import _strip_cache_control
        raw = orjson.dumps(_strip_cache_control(msg))
        h = xxhash.xxh3_64(raw).hexdigest()[:16]
        cc = len(raw)
        self._idx[mid] = (msg, h, cc)
        return h, cc


@dataclass
class Context:
    """请求/响应载体，流经 pipeline 各 stage。"""

    request: web.Request
    body_readonly: dict | None = None   # 只读：ParseStage 设置后永不修改
    fwd_body: dict | None = None
    raw_body: bytes = b""               # 客户端原始请求字节（passthrough 字节级转发用）
    fwd_raw_bytes: bytes | None = None  # 非 None 时 Transport 直接用它作为转发 body
    fwd_headers: dict = field(default_factory=dict)
    req_id: str = ""
    format: str = ""                    # "openai" | "anthropic"

    target_url: str = ""
    upstream_url: str = ""
    is_streaming: bool = False
    timeout: int = 600

    status: int = 0
    resp_body_max_bytes: int = 0      # 0=不截断；>0 时 setter 结构内截断 resp_body
    _resp_body: dict | None = field(default=None, repr=False)
    resp_bytes: bytes = b""
    resp_headers: dict = field(default_factory=dict)

    @property
    def resp_body(self) -> dict | None:
        return self._resp_body

    @resp_body.setter
    def resp_body(self, value: dict | None) -> None:
        """赋值时按 resp_body_max_bytes 截断（聚合副本，转发流不受影响）。

        大响应体（MB 级）是单请求内存峰值主因——截断使峰值有界。
        截断语义见 shared.log.truncate_resp_body（viewer 展示层同构）。
        """
        if value is not None and self.resp_body_max_bytes > 0:
            from deepseek_tap_proxy.shared.log import truncate_resp_body
            value, _ = truncate_resp_body(value, self.resp_body_max_bytes)
        self._resp_body = value

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

    # ── 内部身份（由适配器从外部协议转化，禁止外部词汇）──
    # chain_id 标识「谁在说话」：跨请求稳定，主代理与各子代理各持一个值。
    #
    # ⚠️ 与 ctx.persist.chain_id 同名不同义 —— 同一次请求上两者取值不同：
    #    · 本字段        = 身份，由适配器从外部协议转化而来；
    #    · persist.chain_id = 前缀链标识，由 sys/tools/messages 哈希派生。
    #    收敛后两者归到同一概念，但字段仍分处两层，
    #    跨层传递时必须指明用哪一个 —— 取错会静默查不到链（不报错）。
    chain_id: str = ""
    parent_chain_id: str = ""      # 父身份；主代理与一级子代理均为空串

    # ── 规范形式分段长度（view setter 自动刷新）──
    orig_sys_len: int = 0
    orig_tools_len: int = 0
    orig_msgs_len: int = 0

    # ── 对象的附属索引：dict 对象 → (hash, char_count)，零拷贝 ──
    _msg_index: MessageIndex = field(default_factory=MessageIndex)

    # ── 三段 hash（原生 list[str]，view setter 自动刷新）──
    _sys_hashes: list[str] = field(default_factory=list)
    _tools_hashes: list[str] = field(default_factory=list)
    _messages_hashes: list[str] = field(default_factory=list)

    # ── 三段 char_count（与 hash 同步计算，用于替代 compute_char_count）──
    _sys_char_counts: list[int] = field(default_factory=list)
    _tools_char_counts: list[int] = field(default_factory=list)
    _messages_char_counts: list[int] = field(default_factory=list)

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
        self.req_id = resolve_request_id(self.request)
        if self.fwd_body is None and self.body_readonly is not None:
            # json roundtrip 比 deepcopy 快 ~2.5×（body 是 JSON 原生类型）
            self.fwd_body = orjson.loads(orjson.dumps(self.body_readonly))

    def _external_request_id(self) -> str:
        """兼容入口：委托模块级 ``external_request_id``（唯一实现）。"""
        return external_request_id(self.request)

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
        self._sys_hashes = []
        self._sys_char_counts = []
        for m in s:
            h, cc = self._msg_index.get(m)
            self._sys_hashes.append(h)
            self._sys_char_counts.append(cc)
        self._recompute_family_key()

    def _on_tools_changed(self):
        t = self.tools
        self.orig_tools_len = len(t)
        self._tools_hashes = []
        self._tools_char_counts = []
        for m in t:
            h, cc = self._msg_index.get(m)
            self._tools_hashes.append(h)
            self._tools_char_counts.append(cc)
        self._recompute_family_key()

    def _on_messages_changed(self):
        m = self.messages
        self.orig_msgs_len = len(m)
        self._messages_hashes = []
        self._messages_char_counts = []
        for m in m:
            h, cc = self._msg_index.get(m)
            self._messages_hashes.append(h)
            self._messages_char_counts.append(cc)

    def char_count_of(self, items: list[dict]) -> int:
        """返回任意 list[dict] 的 char_count，使用 MessageIndex 缓存。"""
        return sum(self._msg_index.get(m)[1] for m in items)

    @property
    def total_char_count(self) -> int:
        """三段 char_count 总和（替代 compute_char_count）。"""
        return (sum(self._sys_char_counts) +
                sum(self._tools_char_counts) +
                sum(self._messages_char_counts))

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
        if not isinstance(self.resp_body, dict):
            return {}
        # Usage 抽象层统一入口：厂商从 upstream_name（路由配置名）探测，
        # 缺失时按响应特征（billing_usage 等）兜底
        from deepseek_tap_proxy.shared.usage import REGISTRY, detect_vendor
        raw = self.resp_body.get("usage", {})
        vendor = self.meta.get("upstream_name", "") or detect_vendor(raw)
        return REGISTRY.extract(raw, self.format, vendor)

    @property
    def resp_model(self) -> str:
        if isinstance(self.resp_body, dict):
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
        if not isinstance(self.resp_body, dict):
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
        if not isinstance(self.resp_body, dict):
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
