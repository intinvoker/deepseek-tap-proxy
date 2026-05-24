from __future__ import annotations

USAGE_NORMALIZATION_MAP: dict[str, str] = {
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


def normalize_usage(usage: dict) -> dict:
    normalized: dict = {}
    for raw_key, value in usage.items():
        mapped_key = USAGE_NORMALIZATION_MAP.get(raw_key, raw_key)
        normalized[mapped_key] = value
    return normalized
