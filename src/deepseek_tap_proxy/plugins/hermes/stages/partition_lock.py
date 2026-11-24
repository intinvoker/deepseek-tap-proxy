"""HermesPartitionLockStage — hermes sys 分段锁定。

为何「分段」而不是「整体锁」: 该客户端的 sys 由稳定段 + 易变段组成，
易变段会随技能/记忆更新而漂移。整体锁会把易变段一起冻结 → 模型看到过时内容；
分段锁只冻结「本应稳定」的部分，既保住前缀缓存，又不阻断内容更新。

锁定语义:
  - sys  : 稳定段漂移时回滚为 canonical，易变尾部保留当前值
  - tools: 整体回滚为 canonical
  - 找不到易变标记（hermes 结构变化）→ 保守放行 sys，只锁 tools
"""
from __future__ import annotations

import orjson

from deepseek_tap_proxy.pipeline import Context, LockSegment, Stage, SysToolsSnapshot
from deepseek_tap_proxy.shared.stages.sys_tools_lock import (
    _match_prefix_items,
    _schedule_canonical_init,
)

# 易变段标记：出现即视为易变区起点
VOLATILE_MARKERS = ("## Skills (mandatory)", "# Honcho Memory")


def split_stable_prefix(content: str) -> tuple[str, str] | None:
    """切出「稳定段 + 易变尾部」。

    以最早的易变标记为界。无任何标记时返回 None
    （调用方据此保守放行，不锁 sys）。
    """
    idx = len(content)
    for marker in VOLATILE_MARKERS:
        i = content.find(marker)
        if i != -1:
            idx = min(idx, i)
    if idx >= len(content):
        return None
    return content[:idx], content[idx:]


class HermesPartitionLockStage(Stage):
    """hermes 分段锁定：sys 只锁稳定段，tools 整体锁。"""

    def __init__(self, enabled: bool = True) -> None:
        self._default_enabled = enabled

    async def process(self, ctx: Context) -> Context:
        rt = ctx.request.get("runtime")
        config = rt.config if rt else None
        if not getattr(config, "lock_sys_tools", self._default_enabled):
            return ctx
        # 显式 stream:false（title 生成/批量）旁路，原样透传
        if isinstance(ctx.body_readonly, dict) \
                and ctx.body_readonly.get("stream") is False:
            return ctx
        if ctx.chat_type in ("subagent", "auto_title"):
            ctx.meta["sys_tools_locked"] = "skip"
            ctx.persist.sys_tools_locked = "skip"
            return ctx

        session_id = ctx.session_id
        if not session_id:
            return ctx
        store = rt.store if rt else None
        if not store:
            return ctx

        cur_sys = ctx.sys
        cur_tools = ctx.tools
        if not cur_sys and not cur_tools:
            return ctx
        sys_before_len = len(cur_sys)
        tools_before_len = len(cur_tools)

        cur_content = cur_sys[0].get("content", "") if cur_sys else ""
        cur_split = split_stable_prefix(cur_content) if cur_content else None

        # 按身份 (session_id, chain_id) 取冻结面。hermes 适配器不设 chain_id
        # → ctx.chain_id 恒为 ""，归一后仍是默认链（与改造前等价）。
        canon = store.get_canonical(session_id, ctx.chain_id)
        has_canon_sys = bool(canon.get("sys"))
        has_canon_tools = bool(canon.get("tools"))

        if not has_canon_sys and not has_canon_tools:
            sys_json = orjson.dumps(cur_sys).decode() if cur_sys else ""
            tools_json = orjson.dumps(cur_tools).decode() if cur_tools else ""
            store.set_canonical(session_id, ctx.chain_id,
                                sys=sys_json, tools=tools_json)
            # chain_id 归一由 _schedule_canonical_init 内部完成（与上面一行对称）
            await _schedule_canonical_init(store, session_id, ctx.chain_id,
                                           sys_json, tools_json)
            ctx.meta["sys_tools_locked"] = "init"
            ctx.persist.sys_tools_locked = "init"
            ctx.sys_tools_snap = SysToolsSnapshot(
                action="init",
                sys=LockSegment(before=sys_before_len, match=0, after=sys_before_len),
                tools=LockSegment(before=tools_before_len, match=0,
                                  after=tools_before_len),
            )
            return ctx

        restored_sys = list(cur_sys)
        restored_tools = list(cur_tools)
        needs_lock = False

        # ── sys: 只回滚稳定段，尾部保留当前值 ──
        if has_canon_sys and cur_split is not None and cur_sys:
            canon_sys = orjson.loads(canon["sys"])
            canon_content = canon_sys[0].get("content", "") if canon_sys else ""
            canon_split = split_stable_prefix(canon_content) if canon_content else None
            if canon_split is not None:
                canon_stable, _ = canon_split
                cur_stable, cur_tail = cur_split
                if cur_stable != canon_stable:
                    restored_sys = [dict(cur_sys[0], content=canon_stable + cur_tail)]
                    needs_lock = True

        # ── tools: 整体回滚 ──
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
                sys=LockSegment(before=sys_before_len, match=sys_before_len,
                                after=sys_before_len),
                tools=LockSegment(before=tools_before_len, match=tools_before_len,
                                  after=tools_before_len),
            )
            return ctx

        if restored_sys:
            ctx.sys = restored_sys
        if restored_tools:
            ctx.tools = restored_tools

        sys_match = (_match_prefix_items(cur_sys, restored_sys)
                     if restored_sys else sys_before_len)
        tools_match = (_match_prefix_items(cur_tools, restored_tools)
                       if restored_tools else tools_before_len)

        ctx.meta["sys_tools_locked"] = "lock"
        ctx.persist.sys_tools_locked = "lock"
        ctx.persist.is_modified = True
        ctx.persist.modification_note = "hermes_partition_lock"
        ctx.sys_tools_snap = SysToolsSnapshot(
            action="lock",
            sys=LockSegment(before=sys_before_len, match=sys_match,
                            after=len(ctx.sys)),
            tools=LockSegment(before=tools_before_len, match=tools_match,
                              after=len(ctx.tools)),
        )
        return ctx
