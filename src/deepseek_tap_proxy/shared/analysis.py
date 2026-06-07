"""Request analysis: client_type, intent, session — format-aware, with optional DB persistence."""

from __future__ import annotations
import re
import xxhash
from datetime import datetime, timezone
from typing import NamedTuple

from deepseek_tap_proxy.shared.session import Session
from deepseek_tap_proxy.shared.db import (
    find_candidate_sessions,
    get_or_create_session_by_explicit_id,
    update_session_hash_chain,
)
from deepseek_tap_proxy.shared.hash import msg_hash


class LockDecision(NamedTuple):
    """PrefixLock 决策结果。

    n:           链中最后一个匹配到 curr 的索引 (0-based)
    should_lock: 是否锁定（n >= min_match）
    lock_type:   "layer1_perfect" | "layer2_partial" | "none"
    """
    n: int
    should_lock: bool
    lock_type: str


def _get_conn(db):
    """Extract sqlite3.Connection from db object."""
    return db.conn if hasattr(db, 'conn') else db


# ── Constants ──

COMPACTION_SIGNATURE = "Your task is to create a comprehensive, detailed summary"

_SESSION_UUID_RE = re.compile(
    r'GitHub\.copilot-chat[\\/](?:debug-logs|transcripts)[\\/]'
    r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})'
)

# ── Client type detection ──

def detect_client_type_openai(msgs: list[dict]) -> str:
    """Detect client type from OpenAI-format messages."""
    if not msgs:
        return "generic"
    c0 = msgs[0].get("content", "")
    # Compaction signature is unique to Copilot — check it first so
    # compaction requests are still identified as copilot even when
    # <environment_info> isn't in msg[1].
    if isinstance(c0, str) and c0.startswith(COMPACTION_SIGNATURE):
        return "copilot"
    if len(msgs) < 2:
        return "generic"
    c1 = msgs[1].get("content", "")
    # <environment_info> is the strongest Copilot signal
    if isinstance(c1, str) and "<environment_info>" in c1:
        return "copilot"
    if isinstance(c0, str) and msgs[0].get("role") == "system":
        return "hermes"
    return "generic"


# ── Intent detection ──

def detect_intent_openai(client_type: str, msgs: list[dict]) -> str:
    """Detect request intent. Only copilot + specific msg[0] triggers compaction."""
    if client_type != "copilot":
        return "normal"
    if not msgs:
        return "normal"
    c0 = msgs[0].get("content", "")
    if isinstance(c0, str) and c0.startswith(COMPACTION_SIGNATURE):
        return "compaction"
    return "normal"


# ── Session resolution ──
_HERMES_SESSION_ID_RE = re.compile(
    r'Current Session ID[:\s]+(\S+)', re.IGNORECASE
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
    return xxhash.xxh3_64(("<<<MSG>>>".join(parts)).encode()).hexdigest()[:16]


def resolve_session_openai(
    msgs: list[dict],
    db=None,
) -> Session:
    """Resolve Session from OpenAI-format messages.

    If db is provided, tries hash chain matching as fallback.
    """
    client_type = detect_client_type_openai(msgs)
    total_msgs = len(msgs)
    threshold = 3 if total_msgs > 5 else 2

    # Anchor index per client type
    if client_type == "copilot":
        anchor_idx = 2  # first user message
        uuid = extract_copilot_uuid(msgs[0].get("content", "") if msgs else "")
        # Compaction 请求的 msgs[0] 是总结指令，UUID 在 msg[1:] 的 <context> 中
        if not uuid and len(msgs) > 1:
            for m in msgs[1:]:
                c = m.get("content", "")
                if isinstance(c, str):
                    uuid = extract_copilot_uuid(c)
                    if uuid:
                        break
        if uuid:
            fingerprint = _compute_fingerprint(msgs, anchor_idx, min(4, total_msgs - anchor_idx)) or "unknown"
            if db:
                conn = _get_conn(db)
                get_or_create_session_by_explicit_id(
                    conn, uuid, client_type, anchor_idx,
                    extract_curr_hashes(msgs, anchor_idx),
                )
            return Session(key=uuid, client_type=client_type, fingerprint=fingerprint, method="uuid")
    elif client_type == "hermes":
        anchor_idx = 1
        uuid = extract_hermes_uuid(msgs[0].get("content", "") if msgs else "")
        if uuid:
            fingerprint = _compute_fingerprint(msgs, anchor_idx, min(4, total_msgs - anchor_idx)) or "unknown"
            if db:
                conn = _get_conn(db)
                get_or_create_session_by_explicit_id(
                    conn, uuid, client_type, anchor_idx,
                    extract_curr_hashes(msgs, anchor_idx),
                )
            return Session(key=uuid, client_type=client_type, fingerprint=fingerprint, method="uuid")
    else:
        anchor_idx = 0

    # Fallback: fingerprint
    fingerprint = _compute_fingerprint(msgs, anchor_idx, min(4, total_msgs - anchor_idx)) or "unknown"

    # Detect compaction: copilot's msg[0] is a compaction instruction,
    # not the normal system prompt.  Skip max_hash_chain update so the
    # chain isn't polluted with a compaction-instruction hash at [0].
    is_compaction = (
        client_type == "copilot"
        and msgs
        and isinstance(msgs[0].get("content", ""), str)
        and msgs[0]["content"].startswith(COMPACTION_SIGNATURE)
    )

    # Hash chain matching (if db provided)
    if db:
        conn = _get_conn(db)
        curr_hashes = extract_curr_hashes(msgs, anchor_idx)
        candidates = find_candidate_sessions(conn, client_type)
        for candidate in candidates:
            max_chain = candidate["max_hash_chain"].split(",") if candidate["max_hash_chain"] else []
            if is_same_session(curr_hashes, max_chain, threshold):
                if not is_compaction:
                    update_session_hash_chain(conn, candidate["session_id"], curr_hashes)
                key = candidate["session_id"]
                return Session(key=key, client_type=client_type, fingerprint=fingerprint, method="match")

        # No match → create new session
        new_id = f"syn_{xxhash.xxh3_64(fingerprint.encode()).hexdigest()[:16]}"
        now = datetime.now(timezone.utc).isoformat()
        get_or_create_session_by_explicit_id(
            conn, new_id, client_type, anchor_idx, curr_hashes, now=now,
        )
        key = new_id
    else:
        key = fingerprint

    return Session(key=key, client_type=client_type, fingerprint=fingerprint, method="create")


# ── Anthropic detection ──


def detect_client_type_anthropic(headers: dict) -> str:
    """Anthropic client detection. Only Claude Code has a reliable fingerprint."""
    sid = headers.get("x-claude-code-session-id", "")
    if sid:
        return "claude-code"
    return "generic"


def resolve_session_anthropic(
    system, headers: dict, db=None
) -> Session:
    """Resolve Session from Anthropic-format request."""
    client_type = detect_client_type_anthropic(headers)
    msgs_for_fp = _system_to_msgs(system)
    total_msgs = len(msgs_for_fp)
    threshold = 3 if total_msgs > 5 else 2
    anchor_idx = 0

    if client_type == "claude-code":
        key = headers.get("x-claude-code-session-id", "")
        if key:
            if db:
                conn = _get_conn(db)
                get_or_create_session_by_explicit_id(
                    conn, key, client_type, anchor_idx,
                    extract_curr_hashes(msgs_for_fp, anchor_idx),
                )
            fingerprint = _compute_fingerprint(msgs_for_fp, 0, len(msgs_for_fp)) or "unknown"
            return Session(key=key, client_type=client_type, fingerprint=fingerprint, method="uuid")

    fingerprint = _compute_fingerprint(msgs_for_fp, 0, len(msgs_for_fp)) or "unknown"

    if db:
        conn = _get_conn(db)
        curr_hashes = extract_curr_hashes(msgs_for_fp, anchor_idx)
        candidates = find_candidate_sessions(conn, client_type)
        for candidate in candidates:
            max_chain = candidate["max_hash_chain"].split(",") if candidate["max_hash_chain"] else []
            if is_same_session(curr_hashes, max_chain, threshold):
                update_session_hash_chain(conn, candidate["session_id"], curr_hashes)
                key = candidate["session_id"]
                return Session(key=key, client_type=client_type, fingerprint=fingerprint, method="match")

        # No match → create new
        new_id = f"syn_{xxhash.xxh3_64(fingerprint.encode()).hexdigest()[:16]}"
        now = datetime.now(timezone.utc).isoformat()
        get_or_create_session_by_explicit_id(
            conn, new_id, client_type, anchor_idx, curr_hashes, now=now,
        )
        key = new_id
    else:
        key = fingerprint

    return Session(key=key, client_type=client_type, fingerprint=fingerprint)


def _system_to_msgs(system) -> list[dict]:
    """Convert Anthropic system field to list[dict] for fingerprint computation."""
    if isinstance(system, str):
        return [{"role": "system", "content": system}]
    if isinstance(system, list):
        return [{"role": "system", "content": str(b)} for b in system]
    return []


# ── Hash chain matching ──


def extract_curr_hashes(msgs: list[dict], anchor_idx: int = 0) -> list[str]:
    """Extract hash chain from all messages for session matching.

    Uses ALL messages (from idx 0) rather than from anchor_idx to ensure
    the chain is long enough for threshold matching. Short chains (2 entries)
    can never meet threshold=2 when the last message changes between turns.
    The anchor_idx is still used for fingerprint computation.
    """
    if not msgs:
        return []
    return [msg_hash(m) for m in msgs]


def is_same_session(curr_hashes: list[str], max_hashes: list[str], threshold: int) -> bool:
    """Check if two hash chains have >= threshold consecutive matching hashes.

    Uses cross-alignment sliding window (not strict positional) to handle
    compaction requests where msg[0] (and possibly msg[1]) have been
    inserted/removed, shifting the entire hash chain.
    """
    if not curr_hashes or not max_hashes:
        return False

    # Try every alignment between the two chains.  This is O(N³) but chain
    # lengths are < 100, so it's effectively instant.
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


def should_lock(curr_hashes: list[str], prefix_hashes: list[str],
                user_positions: list[int] | None = None,
                threshold: int = 3) -> LockDecision:
    """判断当前请求是否应被前缀锁定，并返回锁定边界 n。

    委托给 match_prefix 单链模式；保留此函数用于向后兼容的调用方。

    返回 LockDecision(n, should_lock, lock_type)。
    锁定 = chain[:n+1] + curr[n+1:]
    """
    from deepseek_tap_proxy.shared.match import _compute_boundary, _build_lookup, _scan_chain

    if not curr_hashes or not prefix_hashes:
        return LockDecision(n=-1, should_lock=False, lock_type="none")

    # Layer 1: 完美延续
    if len(curr_hashes) >= len(prefix_hashes) \
       and curr_hashes[:len(prefix_hashes)] == prefix_hashes:
        return LockDecision(
            n=len(prefix_hashes) - 1,
            should_lock=True,
            lock_type="layer1_perfect",
        )

    # Layer 2: 贪心前向扫描
    up = user_positions or []
    boundary = _compute_boundary(len(curr_hashes), up)
    lookup = _build_lookup(curr_hashes, boundary)
    n = _scan_chain(prefix_hashes, lookup)

    if n >= threshold:
        return LockDecision(n=n, should_lock=True, lock_type="layer2_partial")

    return LockDecision(n=-1, should_lock=False, lock_type="none")
