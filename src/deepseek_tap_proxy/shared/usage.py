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


def normalize_openai(usage: dict) -> dict:
    """Normalize OpenAI-format usage dict.

    OpenAI 的 input_tokens(=prompt_tokens) 已经是总输入，
    不需要额外计算 total_input_tokens。
    """
    if not isinstance(usage, dict):
        return {}
    if not usage:
        return {}

    normalized: dict = {}
    for raw_key, value in usage.items():
        mapped_key = _OPENAI_MAP.get(raw_key, raw_key)
        normalized[mapped_key] = value
    # OpenAI 的 input_tokens 已经是总 token
    input_t = normalized.get("input_tokens", 0) or 0
    normalized["total_input_tokens"] = input_t
    return normalized


def normalize_anthropic(usage: dict) -> dict:
    """Normalize Anthropic-format usage dict.

    Anthropic 格式的 input_tokens 是"未命中部分"（需计算的 token），
    不是总输入。总输入 = input_tokens + cache_read_input_tokens。
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
            for details_key in ("input_tokens_details", "prompt_tokens_details"):
                details = usage.get(details_key)
                if isinstance(details, dict):
                    cached = details.get("cached_tokens")
                    if cached is not None:
                        break
        if cached is not None:
            normalized["cache_read_input_tokens"] = cached

    normalized.setdefault("cache_creation_input_tokens", 0)

    # Anthropic 的 input_tokens 本身就是 cache_miss（未命中部分）
    if "cache_miss_input_tokens" not in normalized:
        miss = usage.get("prompt_cache_miss_tokens")
        if miss is not None:
            normalized["cache_miss_input_tokens"] = miss
        elif usage.get("input_tokens") is not None:
            # DeepSeek Anthropic: input_tokens 就是 miss 部分
            normalized["cache_miss_input_tokens"] = usage["input_tokens"]

    # 总输入 token = input_tokens + cache_read_input_tokens
    raw_input = normalized.get("input_tokens", 0) or 0
    cache_read = normalized.get("cache_read_input_tokens", 0) or 0
    normalized["total_input_tokens"] = raw_input + cache_read

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
