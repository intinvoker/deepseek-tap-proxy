from __future__ import annotations


def normalize_anthropic_usage(usage: dict) -> dict:
    """Normalize Anthropic/DeepSeek Anthropic usage into shared field names."""
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

    if "cache_miss_input_tokens" not in normalized:
        miss = usage.get("prompt_cache_miss_tokens")
        if miss is not None:
            normalized["cache_miss_input_tokens"] = miss

    return normalized
