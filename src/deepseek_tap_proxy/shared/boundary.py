"""Stable prefix boundary detection for PrefixLock.

Determines which portion of a request's messages is "stable history"
(used for chain matching) vs "current round" (preserved as-is).

The algorithm:
  Going backward from end of messages array (skipping position 0 which
  is always the GlobalAgentContext / role=user ctx), find the
  second-to-last ``role=user`` message. Everything from that message
  onward is the *current round* — it must be preserved without
  modification. Everything before it is the *stable history* — it may be
  locked to an existing chain.

  When there is only one non-ctx user message, the entire array is
  treated as stable (first-ever round or intermediate tool-calling
  continuation).
"""

from __future__ import annotations


def extract_stable_prefix(messages: list[dict]) -> int:
    """Return the exclusive boundary index for the stable prefix.

    ``messages[:boundary]`` is the stable history (used for chain matching).
    ``messages[boundary:]`` is the current round (preserved as-is).

    Rules:
    - msg[0] (role=user, GlobalAgentContext) is never counted as a round
      boundary — it always belongs to the stable prefix.
    - If >= 2 non-ctx role=user messages exist going backward, the
      second-to-last one marks the current round start.
    - If <= 1 non-ctx user exists, the entire array is stable
      (first request or tool-calling continuation).
    """
    if not messages:
        return 0

    # Going backward from end, skip position 0 (ctx is role=user)
    user_positions: list[int] = []
    for i in range(len(messages) - 1, 0, -1):
        if messages[i].get("role") == "user":
            user_positions.append(i)
            if len(user_positions) >= 2:
                break

    if len(user_positions) >= 2:
        # Second-to-last user going backward = start of current round
        return user_positions[1]

    # <= 1 non-ctx user → entire array is stable
    return len(messages)
