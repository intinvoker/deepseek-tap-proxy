"""SysToolsLockStage — 同 session 锁定 sys + tools 到 canonical 基线。

通过 MemoryStore 读写（0 I/O），SQLite 落盘为 fire-and-forget。
"""
from __future__ import annotations

import json
import orjson

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
        # 显式 stream:false（title 生成/批量）旁路锁定，原样透传；缺省按流式处理
        if ctx.body_readonly is not None and isinstance(ctx.body_readonly, dict) \
                and ctx.body_readonly.get("stream") is False:
            return ctx
        if ctx.chat_type in ("subagent", "auto_title"):
            ctx.meta["sys_tools_locked"] = "skip"
            ctx.persist.sys_tools_locked = "skip"
            return ctx

        session_id = ctx.session_id
        if not session_id:
            return ctx
        if ctx.client_type not in ("vscode-copilot", "vscode-copilot-deepseek", "claude-code"):
            return ctx

        store: MemoryStore | None = rt.store if rt else None
        if not store:
            return ctx

        cur_sys = ctx.sys
        cur_tools = ctx.tools
        sys_before_len = len(cur_sys)
        tools_before_len = len(cur_tools)

        cur_sys_json = orjson.dumps(cur_sys).decode()
        cur_tools_json = orjson.dumps(cur_tools).decode()

        # Read canonical from MemoryStore (0 I/O, 按身份 (session_id, chain_id) 隔离)
        # Claude Code 子代理复用主会话 session-id → 只按 session 查会让子代理
        # 读到主代理的冻结面（子代理工具集不同会造成误锁）。
        canon = store.get_canonical(session_id, ctx.chain_id)
        has_canon_sys = bool(canon.get("sys"))
        has_canon_tools = bool(canon.get("tools"))

        if not has_canon_sys and not has_canon_tools:
            if not cur_sys and not cur_tools:
                # 无 sys/tools 且无 canonical → 无操作（不设置标记，
                # 与"已处理但无内容"区分；OTLP 输出为空字符串）
                return ctx
            # init: write canonical to MemoryStore（归属该身份自己的链）
            store.set_canonical(session_id, ctx.chain_id,
                                sys=cur_sys_json if cur_sys else "",
                                tools=cur_tools_json if cur_tools else "")
            # Fire-and-forget SQLite（chain_id 归一由 _schedule_canonical_init
            # 内部完成 —— 与上面的 set_canonical 对称，均传原值）
            await _schedule_canonical_init(store, session_id, ctx.chain_id,
                                           cur_sys_json, cur_tools_json)
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
            canon_sys = orjson.loads(canon["sys"])
            if cur_sys != canon_sys:
                restored_sys = canon_sys
                needs_lock = True
        if has_canon_tools:
            canon_tools = orjson.loads(canon["tools"])
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


async def _schedule_canonical_init(store: MemoryStore, session_id: str,
                                   chain_id: str, sys_json: str,
                                   tools_json: str) -> None:
    """Fire-and-forget: 把 canonical 落到 chain_canonical 表（按身份归属）。

    ``chain_id`` 在组合键中位于 ``session_id`` 之后 —— 与直觉相反，勿调换。

    空 chain_id 在**此处**归一为该 session 的默认链（``chain_id or session_id``），
    与 ``MemoryStore._canonical_key`` 同一规则 —— 落盘存的就是内存键，两个入口
    传原值即可（与紧邻的 ``store.set_canonical(session_id, ctx.chain_id, ...)`` 对称）。

    不归一会让主代理那行存成 ``(sid, '')``：与 ``get_canonical(sid, "")`` 经归一
    查的 ``(sid, sid)`` 错位 ⇒ 重启装载后永远查不回（主代理静默重设基线、前缀缓存
    断链），且与 checkpoint 投影出的 ``(sid, sid)`` 行构成同链两行。
    """
    from deepseek_tap_proxy.shared.db import set_chain_canonical

    def _write(conn, sid, cid, s, t):
        set_chain_canonical(conn, sid, cid, sys=s, tools=t, commit=False)

    await store.enqueue_sqlite_write(
        _write, session_id, chain_id or session_id, sys_json, tools_json,
        op_name="canonical_init")
