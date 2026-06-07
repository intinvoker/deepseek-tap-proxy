"""Cross-format hash chain computation for session tracking.

Uses xxHash3 (xxh3_64) for fast, collision-resistant message hashing.
"""

import json
import xxhash


def msg_hash(msg: dict) -> str:
    """Return first 16 hex chars of xxh3_64 of a message dict.

    xxh3_64 provides 64-bit quality (~2× fewer collisions than the old
    32-bit MD5 truncation) at 3-5× the speed.  The first 16 hex chars
    encode the full 64-bit value.
    """
    raw = json.dumps(msg, sort_keys=True, ensure_ascii=False)
    return xxhash.xxh3_64(raw.encode()).hexdigest()[:16]


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
