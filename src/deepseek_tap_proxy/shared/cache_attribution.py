"""缓存命中归因字段（从 prefix_lock 的 sys/tools/msgs 命中数与 usage 派生，纯函数无副作用）。

用途：回答「这次 cache miss 是什么原因」——把 PrefixLock 匹配结果（fc/bc）与
上游返回的 usage（cache_hit_tokens）放在同一行里，供只读分析。

字段语义（写入 OTLP chat attrs）：
- ``cache_match_reason``：客户端侧前缀匹配结论
  - ``new_chain``        新建链（无历史前缀可复用）
  - ``prefix_continue``  layer1 完美前缀延续（应命中缓存）
  - ``system_changed``   system 段首条即不匹配
  - ``tools_changed``    tools 段首条即不匹配
  - ``msg0_changed``     messages 首条即不匹配
  - ``partial_prefix``   中段/尾部变动（layer2 部分匹配）
  - ``no_lock``          未进入锁定路径（如旁路直通）
- ``cache_expected_hit_tokens``：按匹配前缀（anchor moment）推算的期望命中 token
- ``cache_hit_gap``：``expected - actual``（>0 表示期望命中但上游未返回，可能上游驱逐/TTL）
"""

from __future__ import annotations


def classify_cache_match(ctx) -> str:
    """根据 PrefixLock 匹配结果给出客户端侧命中原因（不含上游行为）。"""
    persist = getattr(ctx, "persist", None)
    meta = getattr(ctx, "meta", None) or {}
    lock_type = getattr(persist, "lock_type", "") or ""
    chain_action = getattr(ctx, "chain_action", None)
    action = getattr(chain_action, "action", "") if chain_action else ""

    if not lock_type:
        return "no_lock"
    if action == "create":
        return "new_chain"
    if lock_type == "layer1_perfect":
        return "prefix_continue"

    # layer2_partial：按各段命中数定位首个变化段
    if not meta.get("pref_sys_hit", 1):
        return "system_changed"
    if not meta.get("pref_tools_hit", 1):
        return "tools_changed"
    if not meta.get("pref_msgs_hit", 1):
        return "msg0_changed"
    return "partial_prefix"


def cache_expected_hit_tokens(ctx) -> int:
    """匹配前缀（anchor moment）的 token 数；无 anchor 时 0。"""
    moment = getattr(ctx, "anchor_moment", None)
    return int(getattr(moment, "total_tokens", 0) or 0)


def cache_actual_hit_tokens(ctx) -> int:
    persist = getattr(ctx, "persist", None)
    return int(getattr(persist, "cache_hit_tokens", 0) or 0)


def cache_attribution(ctx) -> dict:
    """返回写入 OTLP 的归因三字段。"""
    expected = cache_expected_hit_tokens(ctx)
    actual = cache_actual_hit_tokens(ctx)
    return {
        "cache_match_reason": classify_cache_match(ctx),
        "cache_expected_hit_tokens": expected,
        "cache_hit_gap": max(0, expected - actual),
    }
