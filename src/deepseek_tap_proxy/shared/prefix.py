"""Prefix chain safe breakpoint computation.

Ensures prefix truncation never breaks tool_call chains.
"""

from __future__ import annotations


def compute_safe_breakpoints(messages: list[dict]) -> list[int]:
    """Compute all safe breakpoint positions in a message list.

    A safe breakpoint is a position where ALL declared tool_calls
    have received their responses (no dangling tool_call chains).

    Args:
        messages: List of message dicts with 'role' and optional
                  'tool_calls' / 'tool_call_id' fields.

    Returns:
        List of indices such that messages[:i] is a complete chain.
        Always includes 0 (empty messages are safe).
    """
    breakpoints = [0]  # empty messages are always safe
    pending: set[str] = set()

    for i, msg in enumerate(messages):
        role = msg.get("role", "")

        if role == "assistant" and "tool_calls" in msg:
            # Declare new tool_calls — NOT safe to break here
            for tc in msg["tool_calls"]:
                pending.add(tc.get("id", ""))

        elif role == "tool" and "tool_call_id" in msg:
            # Respond to a tool_call
            pending.discard(msg["tool_call_id"])

        # Safe to break only when ALL tool_calls are resolved
        if not pending:
            breakpoints.append(i + 1)

    return breakpoints


def find_safe_truncation(
    safe_breakpoints: list[int],
    target_len: int,
) -> int:
    """Find the largest safe breakpoint <= target_len.

    Uses binary search for O(log N) performance.

    Args:
        safe_breakpoints: Sorted list from compute_safe_breakpoints.
        target_len: Desired truncation length.

    Returns:
        Index to truncate at (safe_breakpoints value).
    """
    lo, hi = 0, len(safe_breakpoints) - 1
    result = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if safe_breakpoints[mid] <= target_len:
            result = safe_breakpoints[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return result


def validate_tool_call_chain(messages: list[dict]) -> bool:
    """Validate tool_call chain integrity.

    Checks both directions:
    1. No DANGLING tool_calls — every assistant(tool_calls) has corresponding
       tool response(s).
    2. No ORPHANED tool messages — every tool response has a preceding
       assistant message with a matching tool_call_id.

    Args:
        messages: List of message dicts.

    Returns:
        True if chain is complete and consistent.
    """
    pending: set[str] = set()
    known_ids: set[str] = set()  # all tool_call_ids declared by assistant messages
    for msg in messages:
        role = msg.get("role", "")
        if role == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id", "")
                pending.add(tc_id)
                known_ids.add(tc_id)
        elif role == "tool" and "tool_call_id" in msg:
            tc_id = msg["tool_call_id"]
            # Orphaned tool: no assistant message ever declared this tool_call_id
            if tc_id not in known_ids:
                return False
            pending.discard(tc_id)
    return len(pending) == 0


def make_chain_id(hash_chain: list[str]) -> str:
    """从 hash_chain 内容衍生链 ID。

    xxhash3_64 保证确定性和低碰撞率。
    同 hash_chain = 同 chain_id（CREATE 路径下不冲突）。
    """
    import xxhash
    return xxhash.xxh3_64("|".join(hash_chain).encode()).hexdigest()[:16]
