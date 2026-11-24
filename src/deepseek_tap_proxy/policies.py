"""请求模式（policy）解析 — passthrough / adapter / managed。

设计目标：

* ``passthrough``（默认）：只路由 + 持久化；body/response 字节级不变；
* ``adapter``：passthrough + 显式白名单协议适配（dsh compat）；
* ``managed``：现状完整管线（缓存链 / 上下文处理）。

选择顺序：显式 override（path/tool/header）> clients 映射 > default。
配置来源：``--policies-config`` / ``DEEPSEEK_TAP_POLICIES_CONFIG`` 指向的 YAML；
未配置时使用与设计稿一致的**内置默认映射**（无需配置文件即可工作）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("deepseek-tap")

PASSTHROUGH = "passthrough"
ADAPTER = "adapter"
MANAGED = "managed"
LEGACY = "legacy"  # 旧版管道兼容：managed 管线 + 旧序列化，用于切流前 byte-compat
MODES = frozenset({PASSTHROUGH, ADAPTER, MANAGED, LEGACY})

KIND_CHAT = "chat"
KIND_SEARCH = "search"
KIND_FETCH = "fetch"
KIND_OTHER = "other"
KINDS = frozenset({KIND_CHAT, KIND_SEARCH, KIND_FETCH, KIND_OTHER})

# 内置默认策略：与设计稿 §2 的客户端映射一致。
# 未知客户端 / 未知端点 → passthrough（默认值）。
DEFAULT_POLICY: dict[str, Any] = {
    "default": PASSTHROUGH,
    "clients": {
        "claude-code": MANAGED,
        "hermes": MANAGED,
        "vscode-copilot": MANAGED,
        "vscode-copilot-deepseek": MANAGED,
        "dsh": ADAPTER,
    },
    "overrides": [
        # 搜索 / 抓取类工具：无论客户端，强制 passthrough（不写缓存链）。
        {
            "match": {"body_tools_type_prefix": ["web_search", "deepseek_web_search"]},
            "mode": PASSTHROUGH,
            "kind": KIND_SEARCH,
            "reason": "search-tool",
        },
        {
            "match": {"body_tools_type_prefix": ["deepseek_read_url", "read_url", "fetch"]},
            "mode": PASSTHROUGH,
            "kind": KIND_FETCH,
            "reason": "fetch-tool",
        },
        # 搜索 / 抓取新 API 端点：透明转发（不依赖 body 结构）。
        {
            "match": {"path_prefix": ["/v1/search", "/search"]},
            "mode": PASSTHROUGH,
            "kind": KIND_SEARCH,
            "reason": "search-path",
        },
        {
            "match": {"path_prefix": ["/v1/fetch", "/fetch", "/v1/read_url", "/read_url"]},
            "mode": PASSTHROUGH,
            "kind": KIND_FETCH,
            "reason": "fetch-path",
        },
    ],
    "strict_passthrough": False,
}

# chat 端点（用于 kind 推断）
_CHAT_SUFFIXES = (
    "/chat/completions",
    "/messages",
    "/responses",
    "/completions",
)


@dataclass(frozen=True)
class PolicyDecision:
    """一次请求的模式决策（不可变，便于日志/测试断言）。"""

    mode: str = PASSTHROUGH
    kind: str = KIND_OTHER
    source: str = "default"      # override | client | default
    reason: str = ""
    strict: bool = False

    @property
    def intervention(self) -> bool:
        """模式本身不构成干预；实际干预由 stage 的 is_modified 标记。"""
        return False


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return []


def detect_kind(path: str, body: dict | None) -> str:
    """从 path / body.tools[].type 推断 kind（chat/search/fetch/other）。

    优先级：显式工具类型 > 路径关键词 > 聊天端点 > other。
    """
    for tool_type in _tool_types(body):
        lowered = tool_type.lower()
        if "web_search" in lowered or lowered.endswith("_search"):
            return KIND_SEARCH
        if "read_url" in lowered or "fetch" in lowered:
            return KIND_FETCH

    lowered_path = (path or "").lower()
    if "search" in lowered_path:
        return KIND_SEARCH
    if "read_url" in lowered_path or "fetch" in lowered_path:
        return KIND_FETCH
    if _is_chat_path(path):
        return KIND_CHAT
    return KIND_OTHER


def _is_chat_path(path: str) -> bool:
    clean = (path or "").split("?", 1)[0].rstrip("/")
    if clean.startswith("/anthropic"):
        clean = clean[len("/anthropic"):]
    # count_tokens 是附属端点（kind=other），不算 chat
    if clean.endswith("/messages/count_tokens"):
        return False
    return any(clean.endswith(suffix) for suffix in _CHAT_SUFFIXES)


def _tool_types(body: dict | None) -> list[str]:
    """提取 ``body.tools[].type``（只认 type，不认 function.name）。

    设计稿约定按 ``tools[].type`` 识别搜索/抓取 API 工具（如
    ``{"type": "web_search"}``）。**不能**把 ``function.name`` 也纳入匹配：
    Copilot/Claude Code 声明的普通工具里就有 ``WebFetch``/``WebSearch``，
    按名字匹配会把正常 chat 流量误判成 fetch 而强制 passthrough。
    """
    if not isinstance(body, dict):
        return []
    tools = body.get("tools")
    if not isinstance(tools, list):
        return []
    out: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        t = tool.get("type")
        if isinstance(t, str) and t:
            out.append(t)
    return out


class PolicyResolver:
    """模式策略解析器（纯内存，热加载原子替换）。"""

    def __init__(self, config_path: str | None = None) -> None:
        self._config_path = config_path or ""
        self._default = PASSTHROUGH
        self._clients: dict[str, str] = {}
        self._overrides: list[dict] = []
        self._strict = False
        self._apply(DEFAULT_POLICY)  # 内置默认：即使无配置文件也工作
        if self._config_path:
            ok, message = self.reload()
            if not ok:
                log.warning("[policy] %s（保持内置默认策略）", message)
        else:
            log.info("[policy] 未配置 policies 文件，使用内置默认策略: %s", self.describe())

    # ── 配置加载 ────────────────────────────────────────────────

    def _apply(self, data: dict) -> None:
        if not isinstance(data, dict):
            raise ValueError("policy config 必须是 mapping")
        default = str(data.get("default", PASSTHROUGH)).strip().lower()
        if default not in MODES:
            raise ValueError(f"非法 default mode: {default!r}")
        clients = data.get("clients") or {}
        if not isinstance(clients, dict):
            raise ValueError("clients 必须是 mapping")
        normalized_clients: dict[str, str] = {}
        for name, mode in clients.items():
            mode_s = str(mode).strip().lower()
            if mode_s not in MODES:
                raise ValueError(f"client {name!r} 的 mode 非法: {mode!r}")
            normalized_clients[str(name).strip()] = mode_s
        overrides = data.get("overrides") or []
        if not isinstance(overrides, list):
            raise ValueError("overrides 必须是 list")
        normalized_overrides: list[dict] = []
        for item in overrides:
            if not isinstance(item, dict):
                continue
            match = item.get("match")
            if not isinstance(match, dict) or not match:
                continue
            mode_s = str(item.get("mode", "")).strip().lower()
            if mode_s and mode_s not in MODES:
                raise ValueError(f"override mode 非法: {item.get('mode')!r}")
            kind_s = str(item.get("kind", "")).strip().lower()
            if kind_s and kind_s not in KINDS:
                raise ValueError(f"override kind 非法: {item.get('kind')!r}")
            normalized_overrides.append({
                "match": match,
                "mode": mode_s or None,
                "kind": kind_s or None,
                "reason": str(item.get("reason", "")),
            })
        # 原子替换（GIL 下引用替换对读线程安全，与 UpstreamRouter.reload 同构）
        self._default = default
        self._clients = normalized_clients
        self._overrides = normalized_overrides
        self._strict = bool(data.get("strict_passthrough", False))

    def _load_file(self) -> dict:
        import yaml

        path = Path(self._config_path)
        if not path.exists():
            raise FileNotFoundError(f"policy config not found: {self._config_path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("policy config 根节点必须是 mapping")
        return data

    def reload(self) -> tuple[bool, str]:
        """热加载配置文件；失败保持旧配置继续服务。"""
        if not self._config_path:
            return False, "no policies config path (built-in defaults in use)"
        try:
            data = self._load_file()
            self._apply(data)
            log.info("[policy] reloaded: %s", self.describe())
            return True, f"reloaded: {self.describe()}"
        except Exception as exc:  # noqa: BLE001 — reload 绝不破坏运行中请求
            return False, f"failed to reload policy config: {exc}"

    def describe(self) -> str:
        return (
            f"default={self._default}, clients={self._clients}, "
            f"overrides={len(self._overrides)}, strict={self._strict}"
        )

    @property
    def config_path(self) -> str:
        return self._config_path

    # ── 解析 ────────────────────────────────────────────────────

    def resolve(
        self,
        client_type: str = "",
        path: str = "",
        body: dict | None = None,
        headers: Any = None,
    ) -> PolicyDecision:
        """按 override > client 映射 > default 的优先级给出模式决策。"""
        inferred_kind = detect_kind(path, body)
        headers_map = _headers_map(headers)

        for override in self._overrides:
            if not _match(override["match"], path, body, headers_map):
                continue
            mode = override["mode"] or self._default
            kind = override["kind"] or inferred_kind
            return PolicyDecision(
                mode=mode, kind=kind, source="override",
                reason=override["reason"] or "override", strict=self._strict,
            )

        mode = self._clients.get(client_type or "")
        if mode is not None:
            return PolicyDecision(
                mode=mode, kind=inferred_kind, source="client",
                reason=f"client={client_type}", strict=self._strict,
            )

        return PolicyDecision(
            mode=self._default, kind=inferred_kind, source="default",
            reason="default", strict=self._strict,
        )


def _headers_map(headers: Any) -> dict[str, str]:
    if headers is None:
        return {}
    try:
        return {str(k).lower(): str(v) for k, v in dict(headers).items()}
    except (TypeError, ValueError, AttributeError):
        return {}


def _match(
    match: dict, path: str, body: dict | None, headers: dict[str, str],
) -> bool:
    """override match 语义：不同 key 之间 AND，同 key 列表内 OR。"""
    matched_any = False

    if "path_prefix" in match:
        prefixes = _as_list(match.get("path_prefix"))
        if not prefixes or not any(path.startswith(p) for p in prefixes):
            return False
        matched_any = True

    if "path_contains" in match:
        needles = _as_list(match.get("path_contains"))
        if not needles or not any(n in path for n in needles):
            return False
        matched_any = True

    if "body_tools_type_prefix" in match:
        prefixes = _as_list(match.get("body_tools_type_prefix"))
        tool_types = [t.lower() for t in _tool_types(body)]
        if not prefixes or not any(
            t.startswith(p.lower()) or p.lower() in t
            for t in tool_types for p in prefixes
        ):
            return False
        matched_any = True

    if "header" in match:
        expected = match.get("header") or {}
        if not isinstance(expected, dict):
            return False
        for name, prefix in expected.items():
            actual = headers.get(str(name).lower())
            if actual is None:
                return False
            if not actual.startswith(str(prefix)):
                return False
        matched_any = True

    return matched_any
