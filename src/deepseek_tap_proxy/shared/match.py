"""前缀链匹配引擎 — 双向位置化匹配。

为当前请求的 **稳定前缀** hash 链在已有前缀链中定位最佳匹配。
封装全部匹配逻辑，PrefixLockStage 只负责执行锁定。

匹配策略（双向）:
  1. 前向: 从位置 0 开始逐位置比较，遇不同即停 → fc
  2. 后向: 从末尾往 fc 方向逐位置比较，遇不同即停 → bc
  3. 判定: 前后各达到阈值 + 总占比达到阈值 → 确认同一链
  4. 锁定: chain[:fc] + curr[fc:n-bc] + chain[n-bc:] + 当前轮

对比旧版贪心扫描（_scan_chain）:
  - ❌ hash 存在性匹配 (跨位置跳跃 → NewRound 被替换)
  - ✅ 位置化匹配 (同索引严格相等 → NewRound 完整保留)
"""

from __future__ import annotations
from typing import NamedTuple


class MatchResult(NamedTuple):
    """匹配结果。

    chain_id:   匹配到的链 ID
    fc:         前向匹配条数（0-based count，从位置 0 开始有几条匹配）
    bc:         后向匹配条数（0-based count，从末尾向前有几条匹配）
                锁定: chain[:fc] + curr[fc:n-bc] + chain[n-bc:]
    lock_type:  "layer1_perfect" | "layer2_partial"
    """

    chain_id: str
    fc: int
    bc: int
    lock_type: str


# ── 内部匹配函数 ──────────────────────────────────────────


def _match_front(
    chain_hashes: list[str], curr_hashes: list[str], n: int
) -> int:
    """前向位置化匹配。

    从位置 0 开始，逐位置比较 curr[i] == chain[i]，
    遇到第一个不匹配的位置或到达 n-1 为止。
    Returns 匹配的条数 (0..n)。
    """
    for i in range(n):
        if curr_hashes[i] != chain_hashes[i]:
            return i
    return n


def _match_back(
    chain_hashes: list[str], curr_hashes: list[str], n: int, fc: int
) -> int:
    """后向位置化匹配。

    从位置 n-1 向 fc 方向（不重叠），逐位置比较 curr[i] == chain[i]，
    遇到第一个不匹配即停。
    Returns 匹配的条数 (0..n-fc)。
    """
    bc = 0
    for i in range(n - 1, fc - 1, -1):
        if curr_hashes[i] != chain_hashes[i]:
            break
        bc += 1
    return bc


# ── 阈值判定 ────────────────────────────────────────────

_SHORT_CHAIN_THRESHOLD = 6


def _passes_threshold(fc: int, bc: int, match_len: int) -> bool:
    """判定是否通过双向匹配阈值。

    长链 (match_len >= 6):  fc >= 3 AND bc >= 3
    短链 (match_len < 6):   fc >= 1 AND bc >= 1
    """
    if match_len < _SHORT_CHAIN_THRESHOLD:
        return fc >= 1 and bc >= 1
    return fc >= 3 and bc >= 3


# ── 主入口 ──────────────────────────────────────────────


def match_prefix(
    chains: list[dict],
    curr_hashes: list[str],
) -> MatchResult | None:
    """为当前请求的 hash 链在已有前缀链中定位最佳匹配。

    Args:
        chains:       get_session_chains 返回的链列表。
                      每个链需包含 ``chain_id`` 和 ``hash_chain``。
        curr_hashes:  当前请求的**稳定前缀** hash 链
                      （由 extract_stable_prefix 预先提取，
                      不含当前轮的新消息）。

    Returns:
        MatchResult(chain_id, fc, bc, lock_type)，
        或 None（无满足阈值的匹配 → 应 CREATE 新链）。
    """
    if not chains or not curr_hashes:
        return None

    n = len(curr_hashes)

    best: MatchResult | None = None
    best_total = -1

    for chain in chains:
        chain_id = chain["chain_id"]
        chain_hashes_str = chain.get("hash_chain", "")
        chain_hashes = chain_hashes_str.split(",") if chain_hashes_str else []

        if not chain_hashes:
            continue

        match_len = min(len(chain_hashes), n)
        if match_len == 0:
            continue

        # Layer 1: 完美匹配 — curr[:match_len] 与 chain[:match_len] 逐位置全等
        if curr_hashes[:match_len] == chain_hashes[:match_len]:
            total_matched = match_len
            if best is None or total_matched > best_total:
                best = MatchResult(
                    chain_id=chain_id,
                    fc=match_len,
                    bc=0,
                    lock_type="layer1_perfect",
                )
                best_total = total_matched
            continue

        # Layer 2: 双向位置化匹配
        fc = _match_front(chain_hashes, curr_hashes, match_len)
        bc = _match_back(chain_hashes, curr_hashes, match_len, fc)

        if not _passes_threshold(fc, bc, match_len):
            continue

        total_matched = fc + bc
        if best is None or total_matched > best_total:
            best = MatchResult(
                chain_id=chain_id,
                fc=fc,
                bc=bc,
                lock_type="layer2_partial",
            )
            best_total = total_matched

    return best
