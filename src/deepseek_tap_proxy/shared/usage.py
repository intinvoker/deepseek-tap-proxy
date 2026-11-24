"""Cross-format usage normalization.

Normalizes both OpenAI and Anthropic usage dicts into a common set of keys:

    input_tokens, output_tokens,
    cache_read_input_tokens, cache_creation_input_tokens,
    cache_miss_input_tokens
"""
from __future__ import annotations


_OPENAI_MAP: dict[str, str] = {
    "input_tokens": "input_tokens",
    "prompt_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "completion_tokens": "output_tokens",
    "cache_read_input_tokens": "cache_read_input_tokens",
    "cache_creation_input_tokens": "cache_creation_input_tokens",
    "prompt_cache_hit_tokens": "cache_read_input_tokens",
    "prompt_cache_miss_tokens": "cache_miss_input_tokens",
    "cached_tokens": "cache_read_input_tokens",
    "cached_content_token_count": "cache_read_input_tokens",
    "cachedContentTokenCount": "cache_read_input_tokens",
    "promptTokenCount": "input_tokens",
    "candidatesTokenCount": "output_tokens",
}


def _extract_nested_cached_tokens(usage: dict) -> int | None:
    """Extract cache_read tokens from nested usage structures.

    Handles vendor-specific nested formats where cache information
    is inside ``input_tokens_details`` or ``prompt_tokens_details``
    rather than at the top level.

    Returns the cached token count, or ``None`` if no nested
    cache information is found.
    """
    for details_key in ("input_tokens_details", "prompt_tokens_details"):
        details = usage.get(details_key)
        if isinstance(details, dict):
            cached = details.get("cached_tokens")
            if cached is not None:
                return cached
    return None


def normalize_openai(usage: dict) -> dict:
    """Normalize OpenAI-format usage dict.

    Handles flat key mapping (``_OPENAI_MAP``) plus nested fallback
    for ``prompt_tokens_details.cached_tokens`` etc.  Computes
    ``cache_miss_input_tokens`` and ``total_input_tokens``.
    """
    if not isinstance(usage, dict):
        return {}
    if not usage:
        return {}

    # ── 1. Flat key mapping ──
    normalized: dict = {}
    for raw_key, value in usage.items():
        mapped_key = _OPENAI_MAP.get(raw_key, raw_key)
        # Prefer non-zero value when multiple raw keys map to same normalized key
        # (e.g. input_tokens=0 and prompt_tokens=5 — the 5 should win)
        if mapped_key in normalized and normalized[mapped_key] and not value:
            continue
        normalized[mapped_key] = value

    # ── 2. Nested fallback for cache_read ──
    if "cache_read_input_tokens" not in normalized:
        nested = _extract_nested_cached_tokens(usage)
        if nested is not None:
            normalized["cache_read_input_tokens"] = nested

    # ── 3. Defaults ──
    normalized.setdefault("cache_read_input_tokens", 0)

    # ── 4. Miss: flat key priority, else compute ──
    miss = usage.get("prompt_cache_miss_tokens")
    if miss is not None:
        normalized["cache_miss_input_tokens"] = miss
    else:
        raw_input = normalized.get("input_tokens", 0) or 0
        cache_read = normalized.get("cache_read_input_tokens", 0) or 0
        normalized["cache_miss_input_tokens"] = max(0, raw_input - cache_read)

    # ── 5. Creation: upstream 优先，缺省 0（不 miss 回退——两者语义不同）──
    normalized.setdefault("cache_creation_input_tokens", 0)

    # ── 6. total_input_tokens ──
    input_t = normalized.get("input_tokens", 0) or 0
    normalized["total_input_tokens"] = input_t

    return normalized


def normalize_anthropic(usage: dict) -> dict:
    """Normalize Anthropic-format usage dict.

    标准 Anthropic 格式：
      input_tokens                 = 总输入 token
      cache_read_input_tokens      = 缓存命中 token
      cache_creation_input_tokens  = 缓存创建 token
      cache_miss_input_tokens      = 未命中 token = input_tokens - cache_read

    总输入 = cache_miss + cache_read = input_tokens
    """
    if not isinstance(usage, dict):
        return {}

    normalized = dict(usage)

    if "input_tokens" not in normalized:
        normalized["input_tokens"] = usage.get("prompt_tokens", 0)

    if "output_tokens" not in normalized:
        normalized["output_tokens"] = usage.get("completion_tokens", 0)

    if "cache_read_input_tokens" not in normalized:
        cached = usage.get("cached_tokens")
        if cached is None:
            cached = usage.get("cachedContentTokenCount")
        if cached is None:
            cached = _extract_nested_cached_tokens(usage)
        if cached is not None:
            normalized["cache_read_input_tokens"] = cached

    normalized.setdefault("cache_read_input_tokens", 0)

    # miss: 优先用 prompt_cache_miss_tokens，否则 input_tokens - cache_read
    miss = usage.get("prompt_cache_miss_tokens")
    if miss is not None:
        normalized["cache_miss_input_tokens"] = miss
    else:
        raw_input = normalized.get("input_tokens", 0) or 0
        cache_read = normalized.get("cache_read_input_tokens", 0) or 0
        if cache_read > raw_input:
            # DeepSeek 语义防御：DeepSeek 的 input_tokens =
            # 未命中部分（cache_read 独立计），而 Anthropic 官方语义下
            # cache_read ≤ input_tokens 恒成立——read > input 时必为 DeepSeek 语义。
            normalized["cache_miss_input_tokens"] = raw_input
        else:
            normalized["cache_miss_input_tokens"] = max(0, raw_input - cache_read)

    # creation（同 normalize_openai）
    normalized.setdefault("cache_creation_input_tokens", 0)

    # 总输入：Anthropic 官方语义 = input_tokens；DeepSeek 语义 = input + read
    raw_input = normalized.get("input_tokens", 0) or 0
    cache_read = normalized.get("cache_read_input_tokens", 0) or 0
    if cache_read > raw_input:
        normalized["total_input_tokens"] = raw_input + cache_read
    else:
        normalized["total_input_tokens"] = raw_input

    return normalized


def normalize_usage(usage: dict | None, fmt: str) -> dict:
    """Normalize usage dict from any format into shared key set.

    Args:
        usage: Raw usage dict from response body (or None).
        fmt: ``"openai"`` or ``"anthropic"``.

    Returns:
        Dict with keys: input_tokens, output_tokens,
        cache_read_input_tokens, cache_creation_input_tokens,
        cache_miss_input_tokens.
    """
    if not isinstance(usage, dict):
        return {}
    if fmt == "openai":
        return normalize_openai(usage)
    elif fmt == "anthropic":
        return normalize_anthropic(usage)
    return dict(usage)


# ═══════════════════════════════════════════════════════════════════════
#  Usage 抽象层——统一契约 + 厂商级读取器 + 事件级聚合
#
#  不同渠道的 cache_read 可能出现位置/语义差异（message_delta 增量、
#  billing_usage 嵌套镜像等）。本层固化：UsageData 统一契约 +
#  ExtractorRegistry（按格式/厂商注册读取器）+ StreamingUsageAccumulator
#  （事件级聚合显式化）。
# ═══════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class UsageData:
    """统一 usage 契约（跨格式/厂商）。落库与观测的单一出口。"""

    total_input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    cache_creation_tokens: int = 0

    @classmethod
    def from_normalized(cls, norm: dict) -> "UsageData":
        return cls(
            total_input_tokens=int(norm.get("total_input_tokens") or norm.get("input_tokens") or 0),
            output_tokens=int(norm.get("output_tokens") or 0),
            cache_hit_tokens=int(norm.get("cache_read_input_tokens") or 0),
            cache_miss_tokens=int(norm.get("cache_miss_input_tokens") or 0),
            cache_creation_tokens=int(norm.get("cache_creation_input_tokens") or 0),
        )

    @property
    def cache_hit_ratio(self) -> float:
        return round(self.cache_hit_tokens / self.total_input_tokens, 4) if self.total_input_tokens else 0.0

    def to_persist(self) -> dict:
        """requests 表落库字段（与 apply_cache_analysis 写入一致）。"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "output_tokens": self.output_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_hit_ratio": self.cache_hit_ratio,
        }


def _extract_deepseek_anthropic(usage: dict) -> dict:
    """deepseek 官方直连（Anthropic 兼容层）usage 语义 extractor。

    DeepSeek 直连时 cache_read 可能大于 input_tokens：
    DeepSeek 的 input_tokens = 未命中（新增）部分，cache_read_input_tokens =
    命中部分，二者独立（total = input + read）；与 Anthropic 官方语义
    （input=总输入，miss=input-read）不同。

    真实请求体 token 校验：input + read ≈ body(+tools)。
    read ≤ input 时（官方语义下也可能）无法靠数值判别 → 必须按 vendor 显式注册。
    """
    norm = normalize_anthropic(usage)
    if "prompt_cache_miss_tokens" not in usage:
        raw_input = norm.get("input_tokens", 0) or 0
        cache_read = norm.get("cache_read_input_tokens", 0) or 0
        norm["cache_miss_input_tokens"] = raw_input
        norm["total_input_tokens"] = raw_input + cache_read
    return norm


def _extract_newapi_nested(usage: dict) -> dict:
    """网关风格 extractor：顶层优先，billing_usage 嵌套兜底。

    部分渠道 message_delta 顶层有 cache_read
    （reconstructor 已合并）；``billing_usage.openai_usage.prompt_tokens_details.
    cached_tokens`` 是 OpenAI 语义镜像——顶层缺失时兜底。
    """
    norm = normalize_anthropic(usage)
    if norm.get("cache_read_input_tokens"):
        return norm
    billing = (usage.get("billing_usage") or {}).get("openai_usage") or {}
    details = billing.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is not None:
        norm["cache_read_input_tokens"] = cached
        norm["cache_miss_input_tokens"] = max(0, (norm.get("input_tokens") or 0) - cached)
    return norm


class ExtractorRegistry:
    """按 (api_format, vendor) 注册 usage 读取器。vendor="*" 为通配兜底。"""

    def __init__(self) -> None:
        self._extractors: dict[tuple[str, str], Callable[[dict], dict]] = {}

    def register(self, api_format: str, vendor: str, fn: Callable[[dict], dict]) -> None:
        self._extractors[(api_format, vendor)] = fn

    def extract(self, usage: dict, api_format: str, vendor: str = "") -> dict:
        if not isinstance(usage, dict):
            return {}
        fn = self._extractors.get((api_format, vendor))
        if fn is None:
            fn = self._extractors.get((api_format, "*"))
        if fn is None:
            return normalize_usage(usage, api_format)
        return fn(usage)


REGISTRY = ExtractorRegistry()
REGISTRY.register("anthropic", "*", normalize_anthropic)
REGISTRY.register("anthropic", "deepseek", _extract_deepseek_anthropic)
REGISTRY.register("anthropic", "new-api", _extract_newapi_nested)
REGISTRY.register("openai", "*", normalize_openai)


def detect_vendor(usage: dict) -> str:
    """响应特征探测厂商（无 upstream_name 时兜底）。"""
    if isinstance(usage, dict) and "billing_usage" in usage:
        return "new-api"
    return ""


def extract_usage(usage: dict | None, api_format: str, vendor: str = "") -> UsageData:
    """统一入口：原始 usage dict → UsageData（厂商缺失时特征探测）。"""
    if not isinstance(usage, dict) or not usage:
        return UsageData()
    if not vendor:
        vendor = detect_vendor(usage)
    norm = REGISTRY.extract(usage, api_format, vendor)
    return UsageData.from_normalized(norm)


class StreamingUsageAccumulator:
    """流式 usage 事件聚合器（按格式注册事件处理器，显式化事件级合并）。

    anthropic: message_start → 基线（input）；message_delta → 增量合并
               （cache_read/cache_creation/output 完整值覆盖，dict 直存）
    openai:    事件处理由 OpenAIReconstructor 的 chunk usage 归一化承担
               （chunk usage 是完整值非增量）——本类 openai 分支留空，
               接入方自行调用 feed 或直接构造 usage dict。
    """

    def __init__(self, api_format: str) -> None:
        self._format = api_format
        self._usage: dict = {}

    def _on_anthropic_start(self, data: dict) -> None:
        msg = data.get("message") or {}
        u = msg.get("usage") or {}
        if isinstance(u, dict):
            self._usage.update(u)

    def _on_anthropic_delta(self, data: dict) -> None:
        u = data.get("usage") or {}
        if not isinstance(u, dict):
            return
        for k, v in u.items():
            if isinstance(v, dict):
                self._usage[k] = v
            elif isinstance(v, (int, float)) and v:
                self._usage[k] = v
            # 0 值不覆盖（message_delta 只报有变化的字段；0 覆盖会丢 message_start 基线）

    def feed(self, event_type: str, data: dict) -> None:
        if self._format == "anthropic":
            if event_type == "message_start":
                self._on_anthropic_start(data)
            elif event_type == "message_delta":
                self._on_anthropic_delta(data)

    @property
    def usage(self) -> dict:
        return dict(self._usage)
