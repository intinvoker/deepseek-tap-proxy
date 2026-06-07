from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Session:
    """Session identity, opaque to consumers.

    key: stable unique identifier (UUID preferred, fingerprint fallback).
    client_type: 'copilot' | 'hermes' | 'claude-code' | 'generic'.
    fingerprint: content-based hash for topic-level isolation (always set).
    """
    key: str
    client_type: str
    fingerprint: str = ""
    method: str = ""  # 'uuid' | 'match' | 'create'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Session):
            return NotImplemented
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)


def split_session_fields(session: Session) -> tuple[str, str]:
    """Return (session_id, session_fingerprint).

    session_id: the session identifier (UUID, header ID, or fingerprint fallback).
    session_fingerprint: content-based hash (always set).
    """
    key = session.key
    fp = session.fingerprint or key
    return (key, fp)
