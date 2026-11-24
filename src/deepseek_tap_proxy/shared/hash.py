"""Cross-format hash chain computation for session tracking.

Uses xxHash3 (xxh3_64) for fast, collision-resistant message hashing.
orjson for serialization (~3x faster than stdlib json).
"""

import orjson
import xxhash


def _strip_cache_control(msg: dict) -> dict:
    """剥离 cache_control 字段（顶层 + content 块内）。

    DeepSeek 忽略 cache_control，不参与前缀匹配——对齐回放侧 raw 语义
    语义（_strip_cache_control）。Claude Code 对消息的 cache_control 标记
    不稳定（同一消息有时带有时不带），不剥离会导致前缀 hash 断裂、
    上游缓存全 miss。
    """
    if "cache_control" not in msg and not any(
        isinstance(b, dict) and "cache_control" in b
        for b in (msg.get("content") or [])
        if isinstance(msg.get("content"), list)
    ):
        return msg  # 快速路径：无 cache_control
    out = {k: v for k, v in msg.items() if k != "cache_control"}
    if isinstance(out.get("content"), list):
        out["content"] = [
            {k: v for k, v in b.items() if k != "cache_control"}
            if isinstance(b, dict) else b
            for b in out["content"]
        ]
    return out


def msg_hash(msg: dict) -> str:
    """Return first 16 hex chars of xxh3_64 of a message dict.

    xxh3_64 provides 64-bit quality (~2× fewer collisions than the old
    32-bit MD5 truncation) at 3-5× the speed.  The first 16 hex chars
    encode the full 64-bit value.

    hash 前剥离 cache_control（DeepSeek 忽略该字段，不参与前缀匹配）。
    """
    raw = orjson.dumps(_strip_cache_control(msg))  # bytes, keys auto-sorted
    return xxhash.xxh3_64(raw).hexdigest()[:16]


def compute_msg_hash_chain(messages: list) -> str:
    """Compute comma-separated hash chain for a list of messages."""
    return ",".join(msg_hash(m) for m in messages)


def compute_canonical_hash_chain(
    sys: list[dict], tools: list[dict], messages: list[dict],
) -> str:
    """Compute hash chain from canonical form: sys + tools[] + msg[].

    Each component item contributes one hash — sys, tools, and msg are
    equal citizens.  No synthetic messages are created::

        parts = [hash(s) for s in sys]
              + [hash(t) for t in tools]
              + [hash(m) for m in messages]
    """
    parts: list[str] = []
    for s in sys:
        parts.append(msg_hash(s))
    for t in tools:
        parts.append(msg_hash(t))
    for m in messages:
        parts.append(msg_hash(m))
    return ",".join(parts)
