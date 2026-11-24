from __future__ import annotations

from collections.abc import Mapping

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


def prepare_fwd_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """转发上游前的唯一入口：去 Host → 去 hop-by-hop → 强制 identity 编码。

    Accept-Encoding 必须**显式覆盖**（而非保留客户端值）：透传客户端的
    zstd 声明会让上游压缩响应，而 aiohttp 在解析响应头阶段即抛
    ContentEncodingError（HAS_ZSTD=False），表现为与上游真实状态无关的 502。

    不修改入参。
    """
    out = {k: v for k, v in headers.items() if k.lower() != "host"}
    out = filter_hop_by_hop(out)
    out = {k: v for k, v in out.items() if k.lower() != "accept-encoding"}
    out["Accept-Encoding"] = "identity"
    return out


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
