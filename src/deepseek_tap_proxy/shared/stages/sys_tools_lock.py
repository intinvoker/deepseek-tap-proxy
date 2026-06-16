"""SysToolsLockStage — 同 session 锁定 sys + tools 到 canonical 基线。

通过 MemoryStore 读写（0 I/O），SQLite 落盘为 fire-and-forget。
"""
from __future__ import annotations

import json

from deepseek_tap_proxy.pipeline import Stage, Context, SysToolsSnapshot, LockSegment
from deepseek_tap_proxy.shared.store import MemoryStore


def _match_prefix_items(curr: list[dict], target: list[dict]) -> int:
    count = 0
    for i, item in enumerate(curr):
        if i >= len(target) or item != target[i]:
            break
        count = i + 1
    return count


class SysToolsLockStage(Stage):
    """锁定 sys+tools 到 canonical 值（0 I/O via MemoryStore）。"""

    def __init__(self, enabled: bool = True) -> None:
        self._default_enabled = enabled

    async def process(self, ctx: Context) -> Context:
        rt = ctx.request.get("runtime")
        config = rt.config if rt else {}
        lock_enabled = getattr(config, "lock_sys_tools", self._default_enabled)
        if not lock_enabled:
            return ctx
        if ctx.chat_type in ("subagent", "auto_title"):
            ctx.meta["sys_tools_locked"] = "skip"
            ctx.persist.sys_tools_locked = "skip"
            return ctx

        session_id = ctx.session_id
        if not session_id:
            return ctx
        if ctx.client_type not in ("copilot", "copilot-agent", "claude-code"):
            return ctx

        store: MemoryStore | None = rt.store if rt else None
        if not store:
            return ctx

        cur_sys = ctx.sys
        cur_tools = ctx.tools
        sys_before_len = len(cur_sys)
        tools_before_len = len(cur_tools)

        cur_sys_json = json.dumps(cur_sys, separators=(",", ":"), ensure_ascii=False)
        cur_tools_json = json.dumps(cur_tools, separators=(",", ":"), ensure_ascii=False)

        # Read canonical from MemoryStore (0 I/O, session-level)
        canon = store.get_canonical(session_id)
        has_canon_sys = bool(canon.get("sys"))
        has_canon_tools = bool(canon.get("tools"))

        if not has_canon_sys and not has_canon_tools:
            if not cur_sys and not cur_tools:
                return ctx
            # init: write canonical to MemoryStore
            store.set_canonical(session_id,
                                sys=cur_sys_json if cur_sys else "",
                                tools=cur_tools_json if cur_tools else "")
            # Fire-and-forget SQLite
            _schedule_canonical_init(store, session_id, cur_sys_json, cur_tools_json)
            ctx.meta["sys_tools_locked"] = "init"
            ctx.persist.sys_tools_locked = "init"
            ctx.sys_tools_snap = SysToolsSnapshot(
                action="init",
                sys=LockSegment(before=sys_before_len, match=0, after=sys_before_len),
                tools=LockSegment(before=tools_before_len, match=0, after=tools_before_len),
            )
            return ctx

        restored_sys = list(cur_sys)
        restored_tools = list(cur_tools)
        needs_lock = False

        if has_canon_sys:
            canon_sys = json.loads(canon["sys"])
            if cur_sys != canon_sys:
                restored_sys = canon_sys
                needs_lock = True
        if has_canon_tools:
            canon_tools = json.loads(canon["tools"])
            if cur_tools != canon_tools:
                restored_tools = canon_tools
                needs_lock = True

        if not needs_lock:
            ctx.meta["sys_tools_locked"] = "pass"
            ctx.persist.sys_tools_locked = "pass"
            ctx.sys_tools_snap = SysToolsSnapshot(
                action="pass",
                sys=LockSegment(before=sys_before_len, match=sys_before_len, after=sys_before_len),
                tools=LockSegment(before=tools_before_len, match=tools_before_len, after=tools_before_len),
            )
            return ctx

        # lock: restore canonical
        sys_match = _match_prefix_items(cur_sys, restored_sys) if restored_sys else sys_before_len
        tools_match = _match_prefix_items(cur_tools, restored_tools) if restored_tools else tools_before_len

        if restored_sys:
            ctx.sys = restored_sys
        if restored_tools:
            ctx.tools = restored_tools

        ctx.meta["sys_tools_locked"] = "lock"
        ctx.persist.sys_tools_locked = "lock"
        ctx.sys_tools_snap = SysToolsSnapshot(
            action="lock",
            sys=LockSegment(before=sys_before_len, match=sys_match, after=len(ctx.sys)),
            tools=LockSegment(before=tools_before_len, match=tools_match, after=len(ctx.tools)),
        )
        ctx.persist.is_modified = True
        ctx.persist.modification_note = "sys_tools_locked"
        return ctx


def _schedule_canonical_init(store: MemoryStore, session_id: str,
                             sys_json: str, tools_json: str) -> None:
    """Fire-and-forget: persist canonical to SQLite."""
    from deepseek_tap_proxy.shared.db import set_session_canonical

    def _write(conn, sid, s, t):
        set_session_canonical(conn, sid, sys=s, tools=t)

    store.enqueue_sqlite_write(_write, session_id, sys_json, tools_json,
                               op_name="canonical_init")
