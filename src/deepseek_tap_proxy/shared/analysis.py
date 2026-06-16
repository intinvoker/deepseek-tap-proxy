"""Request analysis: client_type, intent, session — format-aware, with optional DB persistence."""

from __future__ import annotations
import hashlib
import re

from deepseek_tap_proxy.shared.session import Session
from deepseek_tap_proxy.shared.hash import msg_hash


# ── Constants ──

COMPACTION_SIGNATURE = "Your task is to create a comprehensive, detailed summary"

_SESSION_UUID_RE = re.compile(
    r'GitHub\.copilot-chat[\\/](?:debug-logs|transcripts)[\\/]'
    r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})'
)

# ── Client type detection ──

def detect_client_type_openai(msgs: list[dict]) -> str:
    """.. deprecated:: v7
    Replaced by ``RequestAnalysisStage._detect_client_type``.
    Kept for test backward compatibility.
    """
    if len(msgs) < 2:
        return "generic"
    c0 = msgs[0].get("content", "")
    c1 = msgs[1].get("content", "")
    # <environment_info> is the strongest Copilot signal (present in normal
    # and compaction requests alike).  The first message may be either the
    # agent prompt or the compaction instruction.
    if isinstance(c1, str) and "<environment_info>" in c1:
        return "copilot-agent"
    if isinstance(c0, str) and msgs[0].get("role") == "system":
        return "hermes"
    return "generic"


# ── Intent detection ──

def detect_intent_openai(client_type: str, msgs: list[dict]) -> str:
    """Detect request intent. Only copilot + specific msg[0] triggers compaction."""
    if client_type not in ("copilot", "copilot-agent"):
        return "normal"
    if not msgs:
        return "normal"
    c0 = msgs[0].get("content", "")
    if isinstance(c0, str) and c0.startswith(COMPACTION_SIGNATURE):
        return "compaction"
    return "normal"


# ── Session resolution ──
_HERMES_SESSION_ID_RE = re.compile(
    r'(?:^|\n)\s*(?:Current )?Session\s+ID[:\s]+(\S+)', re.IGNORECASE
)


def extract_copilot_uuid(text: str) -> str | None:
    m = _SESSION_UUID_RE.search(text)
    return m.group(1) if m else None


def extract_hermes_uuid(text: str) -> str | None:
    """Extract Hermes session ID from msg[0] content."""
    m = _HERMES_SESSION_ID_RE.search(text)
    return m.group(1) if m else None


def _compute_fingerprint(msgs: list[dict], start: int, count: int) -> str | None:
    parts = []
    for i in range(start, min(len(msgs), start + count)):
        c = msgs[i].get("content", "")
        if isinstance(c, str):
            parts.append(c)
        else:
            return None
    if not parts:
        return None
    return hashlib.md5(("<<<MSG>>>".join(parts)).encode()).hexdigest()[:16]


def resolve_session_openai(msgs: list[dict]) -> Session:
    """Resolve Session from OpenAI-format messages.

    Copilot uses UUID as primary key (extracted from msg[0] path),
    with fingerprint fallback. Hermes/generic always use fingerprint.
    """
    client_type = detect_client_type_openai(msgs)

    if client_type in ("copilot", "copilot-agent"):
        uuid = extract_copilot_uuid(msgs[0].get("content", "") if msgs else "")
        fingerprint = _compute_fingerprint(msgs, 1, 2) or "unknown"
        key = uuid or fingerprint
    elif client_type == "hermes":
        fingerprint = _compute_fingerprint(msgs, 1, 4) or "unknown"
        key = fingerprint
    else:
        fingerprint = _compute_fingerprint(msgs, 0, 4) or "unknown"
        key = fingerprint

    return Session(key=key, client_type=client_type, fingerprint=fingerprint)


# ── Anthropic detection ──


def detect_client_type_anthropic(headers: dict) -> str:
    """Anthropic client detection. Only Claude Code has a reliable fingerprint."""
    sid = headers.get("x-claude-code-session-id", "")
    if sid:
        return "claude-code"
    return "generic"


def resolve_session_anthropic(system, headers: dict) -> Session:
    """Resolve Session from Anthropic-format request."""
    client_type = detect_client_type_anthropic(headers)
    msgs_for_fp = _system_to_msgs(system)
    if client_type == "claude-code":
        key = headers.get("x-claude-code-session-id", "")
    else:
        key = ""
    fingerprint = _compute_fingerprint(msgs_for_fp, 0, len(msgs_for_fp)) or "unknown"
    if not key:
        key = fingerprint
    return Session(key=key, client_type=client_type, fingerprint=fingerprint)


def _system_to_msgs(system) -> list[dict]:
    """Convert Anthropic system field to list[dict] for fingerprint computation."""
    if isinstance(system, str):
        return [{"role": "system", "content": system}]
    if isinstance(system, list):
        return [{"role": "system", "content": str(b)} for b in system]
    return []


# ── Hash chain matching (used by RequestAnalysisStage._match_or_create) ──


def extract_curr_hashes(msgs: list[dict], anchor_idx: int = 0) -> list[str]:
    """Extract hash chain from all messages for session matching."""
    if not msgs:
        return []
    return [msg_hash(m) for m in msgs]


def is_same_session(curr_hashes: list[str], max_hashes: list[str], threshold: int) -> bool:
    """Check if two hash chains have >= threshold consecutive matching hashes.

    Uses cross-alignment sliding window to handle compaction shifts.
    """
    if not curr_hashes or not max_hashes:
        return False
    for ci in range(len(curr_hashes)):
        for mi in range(len(max_hashes)):
            consecutive = 0
            remaining = min(len(curr_hashes) - ci, len(max_hashes) - mi)
            for k in range(remaining):
                if curr_hashes[ci + k] == max_hashes[mi + k]:
                    consecutive += 1
                    if consecutive >= threshold:
                        return True
                else:
                    consecutive = 0
    return False


def find_cross_alignment(
    pure_hashes: list[str],
    chain_hashes: list[str],
    threshold: int = 3,
) -> int | None:
    """在 chain_hashes 中找 pure_hashes 的交叉对齐位置。

    滑动窗口：pure_hashes[0] 对齐 chain_hashes[offset]，
    pure_hashes[1] 对齐 chain_hashes[offset+1]，
    连续匹配 >= threshold 即返回 offset。

    Returns:
        偏移量（chain 中 pure[0] 的位置），或 None（无匹配）。
    """
    if not pure_hashes or not chain_hashes:
        return None
    if len(pure_hashes) < threshold:
        return None

    max_start = len(chain_hashes) - threshold
    for start in range(max_start + 1):
        match_len = 0
        max_match = min(len(pure_hashes), len(chain_hashes) - start)
        for j in range(max_match):
            if pure_hashes[j] == chain_hashes[start + j]:
                match_len += 1
            else:
                break
        if match_len >= threshold:
            return start
    return None
