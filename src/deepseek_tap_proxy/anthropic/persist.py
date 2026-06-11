"""Anthropic-specific persist helpers (pure functions, no I/O)."""
from __future__ import annotations

# ── Title generation detection ─────────────────────────────────────────

_TITLE_GEN_KEYWORDS = (
    "sentence-case title",
    "Generate a concise",
    "3-7 words",
    "session in a list",
)


def _detect_title_gen(req_body: dict | None) -> str | None:
    """Detect if this request is an auto-generated title request.

    Returns "auto_title" if the system prompt contains title-generation
    keywords AND the request carries no tools. Returns None otherwise.
    This is a pure function — no IO, runs in async fire-and-forget.
    """
    if not isinstance(req_body, dict):
        return None

    system = req_body.get("system")
    if not system:
        return None

    # Extract text from system prompt (string or block array)
    texts = []
    if isinstance(system, str):
        texts = [system]
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                for key in ("text", "thinking"):
                    txt = block.get(key)
                    if isinstance(txt, str):
                        texts.append(txt)

    combined = " ".join(texts)
    if not any(kw in combined for kw in _TITLE_GEN_KEYWORDS):
        return None

    # Title generation requests don't carry tools
    if req_body.get("tools"):
        return None

    return "auto_title"


# ── Phase1/Phase2 persist moved to MemoryStore.enqueue_request_phase* ──
# (called directly from handler.py via the single-writer queue)

