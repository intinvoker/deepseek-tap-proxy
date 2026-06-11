"""SysToolsSnapshot + LockSegment — SysToolsLock 产出，挂在 ctx 上供 LogView / trace 直接读。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LockSegment:
    """SysToolsLock 作用于单个 canonical 段的快照。

    仅 action="lock" 时 before != after（体现锁定修正量）。
    """
    before: int = 0
    match: int = 0
    after: int = 0


@dataclass
class SysToolsSnapshot:
    """SysToolsLock 的产出。"""
    action: str = ""                          # "init" | "pass" | "lock" | "skip" | ""
    sys: LockSegment = field(default_factory=LockSegment)
    tools: LockSegment = field(default_factory=LockSegment)
