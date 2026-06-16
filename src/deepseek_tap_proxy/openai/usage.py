"""OpenAI usage normalization — thin wrapper over shared/usage.py.

``normalize_usage`` delegates to ``shared.usage.normalize_openai``.
``USAGE_NORMALIZATION_MAP`` is re-exported for backward compatibility.
"""
from __future__ import annotations

from deepseek_tap_proxy.shared.usage import normalize_openai  # noqa: F401

# Re-export for test backward compatibility
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
    """Normalize OpenAI-format usage dict via shared/usage.py.

    Signature preserved (1 arg) for backward compatibility with
    ``openai/sse.py`` and existing tests.
    """
    return normalize_openai(usage)
