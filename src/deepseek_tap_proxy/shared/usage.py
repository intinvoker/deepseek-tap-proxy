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
    normalized.setdefault("cache_creation_input_tokens", 0)
    normalized.setdefault("cache_read_input_tokens", 0)

    # ── 4. Miss: flat key priority, else compute ──
    miss = usage.get("prompt_cache_miss_tokens")
    if miss is not None:
        normalized["cache_miss_input_tokens"] = miss
    else:
        raw_input = normalized.get("input_tokens", 0) or 0
        cache_read = normalized.get("cache_read_input_tokens", 0) or 0
        normalized["cache_miss_input_tokens"] = max(0, raw_input - cache_read)

    # ── 5. total_input_tokens ──
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

    normalized.setdefault("cache_creation_input_tokens", 0)
    normalized.setdefault("cache_read_input_tokens", 0)

    # miss: 优先用 prompt_cache_miss_tokens，否则 input_tokens - cache_read
    miss = usage.get("prompt_cache_miss_tokens")
    if miss is not None:
        normalized["cache_miss_input_tokens"] = miss
    else:
        raw_input = normalized.get("input_tokens", 0) or 0
        cache_read = normalized.get("cache_read_input_tokens", 0) or 0
        normalized["cache_miss_input_tokens"] = max(0, raw_input - cache_read)

    # 总输入 = input_tokens（标准 Anthropic 格式的 input_tokens 就是总输入）
    raw_input = normalized.get("input_tokens", 0) or 0
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
