"""前缀链匹配引擎。

为当前请求的 hash 链在已有前缀链中定位最佳匹配。
封装全部匹配逻辑（判定 + 边界索引），PrefixLockStage 只负责执行锁定。

性能：O(N + K×M)，其中 N = curr 长度，M = 链长度，K = 链数量。
链数量通常 ≤3，总体 < 150 元素，开销可忽略。
"""

from __future__ import annotations
from typing import NamedTuple


class MatchResult(NamedTuple):
    """匹配结果。

    chain_id:   匹配到的链 ID
    n:          链中最后一个能匹配到 curr 旧内容的索引 (0-based)
                锁定 = chain[:n+1] + curr[n+1:]
    lock_type:  "layer1_perfect" | "layer2_partial" | "none"
    """
    chain_id: str
    n: int
    lock_type: str


def _compute_boundary(curr_len: int, user_positions: list[int],
                      *, disable_user_anchor: bool = False) -> int:
    """确定旧内容边界（搜索上限）。

    ≥2 user 且未禁用锚点时：保护最后一个 user 及之后 = msg_new。
    禁用锚点时（compaction / 首轮 / 子代理）：回退到全量。
    """
    if not disable_user_anchor and len(user_positions) >= 2:
        b = user_positions[-1] - 1
        if b >= 0:
            return b
    return curr_len - 1


def _build_lookup(hashes: list[str], boundary: int) -> dict[str, list[int]]:
    """构建 hash → 所有出现位置 的查找表（递增有序）。

    只索引 boundary 及之前的位置（旧内容区域）。
    存所有位置支持同一 hash 多次出现时的顺序匹配。
    """
    import bisect
    lookup: dict[str, list[int]] = {}
    for i in range(boundary + 1):
        h = hashes[i]
        lookup.setdefault(h, []).append(i)
    return lookup


def _find_pos(lookup: dict[str, list[int]], h: str, curr_ptr: int) -> int | None:
    """在 lookup 中找 h 的第一个 ≥ curr_ptr 的位置。O(log K) 其中 K 是 h 的出现次数。"""
    positions = lookup.get(h)
    if not positions:
        return None
    # 二分查找
    import bisect
    idx = bisect.bisect_left(positions, curr_ptr)
    if idx < len(positions):
        return positions[idx]
    return None


def _scan_chain(
    chain_hashes: list[str],
    lookup: dict[str, list[int]],
) -> int:
    """贪心前向扫描：找链中最后一个能匹配到 curr 旧内容的索引。

    curr_ptr 维护"下次匹配必须在 curr 中 >= 此位置"的约束，
    保证匹配是单调向前的（不交叉、不回退）。
    篡改/删减的 hash 在 lookup 中不存在，自然跳过。
    """
    curr_ptr = 0
    n = -1

    for chain_i, h in enumerate(chain_hashes):
        pos = _find_pos(lookup, h, curr_ptr)
        if pos is not None:
            curr_ptr = pos + 1
            n = chain_i

    return n


def match_prefix(
    chains: list[dict],
    curr_hashes: list[str],
    user_positions: list[int],
    *,
    min_match: int = 2,
    disable_user_anchor: bool = False,
) -> MatchResult | None:
    """为当前请求的 hash 链在已有前缀链中定位最佳匹配。

    Args:
        chains:           get_session_chains 返回的链列表。
        curr_hashes:      当前请求的 hash 链（完整，sys+tools+msg）。
        user_positions:   curr 中所有 role=user 的位置索引。
        min_match:        最少需匹配的链元素数（默认 2）。
        disable_user_anchor: 禁用 user 锚点（compaction / 首轮 / 子代理）。

    Returns:
        MatchResult(chain_id, n, lock_type)，或 None（无匹配 → CREATE 新链）。
    """
    if not chains or not curr_hashes:
        return None

    boundary = _compute_boundary(len(curr_hashes), user_positions,
                                  disable_user_anchor=disable_user_anchor)
    lookup = _build_lookup(curr_hashes, boundary)

    best: MatchResult | None = None

    for chain in chains:
        chain_id = chain["chain_id"]
        chain_hashes = chain["hash_chain"].split(",") if chain["hash_chain"] else []

        if not chain_hashes:
            continue

        # Layer 1: 完美延续 — 当前请求严格承接链的全部内容
        if len(curr_hashes) >= len(chain_hashes) \
           and curr_hashes[:len(chain_hashes)] == chain_hashes:
            candidate_n = len(chain_hashes) - 1
            if best is None or candidate_n > best.n:
                best = MatchResult(
                    chain_id=chain_id,
                    n=candidate_n,
                    lock_type="layer1_perfect",
                )
            continue

        # Layer 2: 贪心前向扫描
        n = _scan_chain(chain_hashes, lookup)
        if n >= min_match:
            if best is None or n > best.n:
                best = MatchResult(
                    chain_id=chain_id,
                    n=n,
                    lock_type="layer2_partial",
                )

    return best
