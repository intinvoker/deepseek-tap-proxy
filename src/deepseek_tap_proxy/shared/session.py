from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Session:
    """Session identity, opaque to consumers.

    key: stable unique identifier (UUID preferred; content fingerprint 仅作无稳定 ID 客户端的兜底).
    client_type: 'copilot' | 'hermes' | 'claude-code' | 'generic'.
    fingerprint: .. deprecated:: fingerprint-retirement V1
        早期 content-based topic hash，仅保留兼容字段；**不作为身份/主键使用**。
    """
    key: str
    client_type: str
    fingerprint: str = ""  # defaults to empty for backward compat

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Session):
            return NotImplemented
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)
