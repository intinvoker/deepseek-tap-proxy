"""CompactionLockStage — Standard Copilot 压缩请求前缀锁定。

激活: chat_type=compaction, client_type=copilot

步骤:
  1. 剥离包装: pure_msgs = msgs[1:-1]
  2. 交叉对齐: find_cross_alignment(pure_hashes, chain.messages_hashes)
  3. 还原头部: locked = chain.messages[:shift] + pure_msgs + [msgs[-1]]
  4. 锚点定位: find_anchor(chain.moments, pref_hit)
"""
from __future__ import annotations
import json
import logging
from deepseek_tap_proxy.pipeline import Stage, Context
from deepseek_tap_proxy.shared.chain import ChainMoment, find_anchor_moment
from deepseek_tap_proxy.shared.stages.helpers import compute_char_count

log = logging.getLogger("deepseek-tap")


class CompactionLockStage(Stage):
    """Standard Copilot 压缩请求前缀锁定。"""

    async def process(self, ctx: Context) -> Context:
        if ctx.chat_type != "compaction":
            return ctx
        if ctx.client_type not in ("copilot", "copilot-agent"):
            return ctx

        session_id = ctx.session_id
        if not session_id:
            return ctx

        rt = ctx.request.get("runtime")
        store = rt.store if rt else None
        if not store:
            return ctx

        msgs_c = ctx.messages
        if len(msgs_c) < 3:
            return ctx

        # 1. 剥离压缩包装
        pure_msgs = msgs_c[1:-1]
        pure_hashes = ctx.messages_hashes[1:-1]

        if len(pure_hashes) < 3:
            return ctx

        # 2. 交叉对齐匹配
        from deepseek_tap_proxy.shared.analysis import find_cross_alignment
        chains = store.get_session_chains(session_id)
        matched_chain = None
        shift = 0
        for chain in chains:
            hash_idx = store.get_chain_hashes(chain.chain_id)
            chain_hashes = hash_idx.messages_hashes if hash_idx else []
            if not chain_hashes:
                continue
            offset = find_cross_alignment(
                pure_hashes, chain_hashes, threshold=3)
            if offset is not None and offset >= 0:
                matched_chain = chain
                shift = offset
                break

        if matched_chain is None:
            return ctx

        # 3. 还原头部 + 锁定
        chain_msgs = matched_chain.messages
        locked_msgs = (
            list(chain_msgs[:shift]) +
            list(pure_msgs) +
            [msgs_c[-1]]
        )
        # 写回规范形式 — setter 自动刷新 hash chain
        ctx.messages = locked_msgs
        ctx.matched_chain = matched_chain

        # 4. 锚点定位
        pref_hit = shift + len(pure_hashes)
        ctx.meta["pref_hit_count"] = pref_hit
        # 修复：压缩请求 sys/tools 与旧链一致（SysToolsLock 保证），显示为全命中
        ctx.meta["pref_sys_hit"] = len(ctx.sys)
        ctx.meta["pref_tools_hit"] = len(ctx.tools)
        ctx.meta["pref_msgs_hit"] = pref_hit
        ctx.anchor_moment = find_anchor_moment(matched_chain.moments, pref_hit)
        ctx.new_char_count = compute_char_count([msgs_c[-1]])

        log.info("[%s] compaction locked: shift=%d, pref_hit=%d, chain=%s",
                 ctx.req_id, shift, pref_hit, matched_chain.chain_id[:16])
        return ctx


def _find_anchor_from_dict(chain: dict, pref_hit: int) -> ChainMoment | None:
    """从 chain dict 的 moments JSON 中找到 messages_count ≤ pref_hit 的最大 Moment。"""
    moments_json = chain.get("moments", "[]")
    if isinstance(moments_json, str):
        try:
            moments_data = json.loads(moments_json)
        except (json.JSONDecodeError, TypeError):
            moments_data = []
    elif isinstance(moments_json, list):
        moments_data = moments_json
    else:
        moments_data = []

    moments = []
    for m in moments_data:
        if isinstance(m, dict):
            moments.append(ChainMoment(
                messages_count=m.get("messages_count", 0),
                op=m.get("op", ""),
                total_tokens=m.get("total_tokens", 0),
                cache_hit_tokens=m.get("cache_hit_tokens", 0),
                cache_miss_tokens=m.get("cache_miss_tokens", 0),
                char_count=m.get("char_count", 0),
                timestamp=m.get("timestamp", ""),
            ))

    best = None
    for m in moments:
        if m.messages_count <= pref_hit:
            best = m
        else:
            break
    return best


# Backward-compat alias (logic moved to chain.find_anchor_moment)
_find_anchor = find_anchor_moment
