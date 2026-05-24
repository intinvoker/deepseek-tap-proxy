from __future__ import annotations

import hashlib
import re

_ANTHROPIC_USER_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def normalize_metadata(req_body: dict) -> dict:
    """Hash non-standard user_id in metadata for DeepSeek compatibility.

    DeepSeek's Anthropic endpoint rejects user_id that don't match
    ``^[a-zA-Z0-9_-]+$``.  Hash them to avoid 400 errors.
    """
    metadata = req_body.get("metadata")
    if not isinstance(metadata, dict):
        return req_body

    user_id = metadata.get("user_id")
    if not isinstance(user_id, str) or _ANTHROPIC_USER_ID_RE.fullmatch(user_id):
        return req_body

    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]
    modified = dict(req_body)
    new_meta = dict(metadata)
    new_meta["user_id"] = f"tap_{digest}"
    modified["metadata"] = new_meta
    return modified
