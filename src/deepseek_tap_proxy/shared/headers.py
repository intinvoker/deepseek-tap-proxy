from __future__ import annotations

HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade",
})

SENSITIVE_HEADERS = frozenset({
    "authorization", "cookie", "set-cookie", "set-cookie2", "x-api-key",
})

PREFIX_REDACTED = frozenset({"authorization", "x-api-key"})


def filter_hop_by_hop(headers: dict) -> dict:
    """Remove hop-by-hop headers from a header dict."""
    return {
        k: v for k, v in headers.items()
        if k.lower() not in HOP_BY_HOP
    }


def redact(headers: dict) -> dict:
    """Return a copy with sensitive values masked.

    Prefix-redactable keys (e.g. Authorization) are truncated to first
    12 chars + '...'. Fully sensitive keys are replaced with '***'.
    """
    out = {}
    for k, v in headers.items():
        kl = k.lower()
        if kl in PREFIX_REDACTED and isinstance(v, str) and len(v) > 12:
            out[k] = v[:12] + "..."
        elif kl in SENSITIVE_HEADERS:
            out[k] = "***"
        else:
            out[k] = v
    return out
