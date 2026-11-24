"""Request analysis: client_type, intent, hash-chain matching — format-aware."""
from __future__ import annotations

from deepseek_tap_proxy.shared.hash import msg_hash


# ── Constants ──

COMPACTION_SIGNATURE = "Your task is to create a comprehensive, detailed summary"

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
        return "vscode-copilot"
    if isinstance(c0, str) and msgs[0].get("role") == "system":
        return "hermes"
    return "generic"


# ── Intent detection ──

def detect_intent_openai(client_type: str, msgs: list[dict]) -> str:
    """Detect request intent. Only copilot + specific msg[0] triggers compaction."""
    if client_type not in ("vscode-copilot", "vscode-copilot-deepseek"):
        return "normal"
    if not msgs:
        return "normal"
    c0 = msgs[0].get("content", "")
    if isinstance(c0, str) and c0.startswith(COMPACTION_SIGNATURE):
        return "compaction"
    return "normal"


# ── Anthropic detection ──


def detect_client_type_anthropic(headers: dict) -> str:
    """Anthropic client detection. Only Claude Code exposes a session-id header."""
    sid = headers.get("x-claude-code-session-id", "")
    if sid:
        return "claude-code"
    return "generic"


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
