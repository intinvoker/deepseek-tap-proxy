"""生产级压力测试 — 基于 Inspector 真实数据 (94K reqs / 360 sessions)。

数据来源: deepseek-tap-data/inspector/inspector.db (6.3GB, 33M context segments)

真实生产模式:
  Client:    claude-code 72%, vscode-copilot 7%, hermes 7%, other 14%
  Sessions:  每 session 平均 288 reqs (claude-code), 83 (hermes), 162 (copilot)
  Chains:    每 session 1-18 chains (median=1), 89% of chains have 1-10 reqs
  Messages:  多数 0-500B (50%), 500B-2KB (31%), 10-50KB (3.5%), >50KB (0.7%)
  Content:   296K unique elements, avg=2KB, max=138KB
"""
from __future__ import annotations

import asyncio
import gc
import os
import random
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone

import orjson
import pytest

from deepseek_tap_proxy.shared.store import MemoryStore, SessionRecord, ChainRecord
from deepseek_tap_proxy.shared.chain import ChainMoment
from deepseek_tap_proxy.shared.memory_governance import estimate_bytes


# ═══════════════════════════════════════════════════════════════════════════
#  Real data distributions (from inspector.db)
# ═══════════════════════════════════════════════════════════════════════════

# Content size distribution (buckets with probability weights)
CONTENT_BUCKETS = [
    (0, 500, 0.50),       # 50%  → 0-500B (tool calls, small messages)
    (500, 2000, 0.31),    # 31%  → 500B-2KB (typical messages)
    (2000, 10000, 0.14),  # 14%  → 2-10KB (tool outputs)
    (10000, 50000, 0.04), # 4%   → 10-50KB (code blocks)
    (50000, 150000, 0.01),# 1%   → 50KB+ (large context)
]

# Chain length distribution (request_count per chain)
CHAIN_LENGTH_WEIGHTS = [
    (1, 10, 0.89),        # 89%  → 1-10 requests
    (11, 50, 0.06),       # 6%   → 11-50
    (51, 100, 0.02),      # 2%   → 51-100
    (101, 500, 0.03),     # 3%   → 101-500 (long Copilot sessions)
]

# Chains per session distribution
CHAINS_PER_SESSION_WEIGHTS = [
    (1, 0.54),    # 54%  → 1 chain
    (2, 0.17),    # 17%  → 2
    (3, 0.04),    # 4%   → 3
    (4, 0.05),    # 5%   → 4
    (5, 0.05),    # 5%   → 5-9
    (10, 0.08),   # 8%   → 10-20
    (20, 0.07),   # 7%   → 20+
]


def _weighted_choice(rng: random.Random, weights: list) -> int:
    """从加权桶中随机选择一个值。"""
    total = sum(w[2] if len(w) > 2 else w[1] for w in weights)
    r = rng.random() * total
    cumulative = 0.0
    for w in weights:
        if len(w) > 2:
            cumulative += w[2]
        else:
            cumulative += w[1]
        if r <= cumulative:
            lo, hi = w[0], w[1]
            return rng.randint(lo, hi) if lo < hi else lo
    return weights[-1][0]


def _random_content(rng: random.Random) -> str:
    """生成符合真实分布的随机内容。"""
    lo, hi = 0, 500
    r = rng.random()
    cumulative = 0.0
    for bucket_lo, bucket_hi, prob in CONTENT_BUCKETS:
        cumulative += prob
        if r <= cumulative:
            lo, hi = bucket_lo, bucket_hi
            break
    size = rng.randint(lo, hi)
    # 生成接近真实 JSON 的内容
    if rng.random() < 0.3:
        return '{"role":"' + rng.choice(["user","assistant","tool"]) + '","content":"' + "x" * max(0, size - 40) + '"}'
    else:
        return "x" * size


def _random_hash_chain(n: int, rng: random.Random) -> str:
    return ",".join([f"{rng.randint(0, 2**64-1):016x}" for _ in range(n)])


# ═══════════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def frozen_now():
    return datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def rng():
    return random.Random(42)  # 固定种子，可复现


class TestProductionPattern:
    """基于生产数据的关键时序模式。"""

    def test_claude_code_heavy_session(self, frozen_now, rng):
        """模拟最极端场景：Claude Code 单 session 11,505 reqs。

        生产数据: claude-code max=11,505 reqs/session, 每 chain avg=6 reqs
        → ~1,900 chains per session (极端), 正常: chains=1-18/session
        """
        store = MemoryStore(
            ttl_hours=168,
            max_memory_bytes=200 * 1024 * 1024,
        )
        store._now = lambda: frozen_now
        store._last_eviction_check = 0

        sid = "claude_heavy"
        store._sessions[sid] = SessionRecord(
            session_id=sid,
            last_seen_time=frozen_now.isoformat(),
        )

        # 模拟 200 chains（每 chain 平均 6 个 request 一轮）
        n_chains = 200
        total_messages = 0
        t0 = time.monotonic()

        for ci in range(n_chains):
            chain_len = _weighted_choice(rng, CHAIN_LENGTH_WEIGHTS)
            msgs = []
            for _ in range(chain_len):
                msgs.append({"role": rng.choice(["user", "assistant"]),
                             "content": _random_content(rng)})
            total_messages += chain_len

            c = ChainRecord(
                chain_id=f"{sid}_c{ci}",
                session_id=sid,
                sys=_random_content(rng),
                tools="[" + ",".join([_random_content(rng) for _ in range(rng.randint(1, 50))]) + "]",
                messages=orjson.dumps(msgs).decode(),
                sys_hashes=_random_hash_chain(1, rng),
                tools_hashes=_random_hash_chain(rng.randint(10, 50), rng),
                messages_hashes=_random_hash_chain(chain_len, rng),
                msg_count=chain_len,
                last_seen=frozen_now.isoformat(),
            )
            fk = store.compute_family_key(c.sys_hashes, c.tools_hashes)
            store._chains.setdefault(sid, {}).setdefault(fk, []).append(c)

        t_insert = time.monotonic() - t0
        est = estimate_bytes(store._sessions, store._chains)
        chain_count = sum(len(clist) for fmap in store._chains.values() for clist in fmap.values())

        print(f"\n  Insert {n_chains} chains / {total_messages} msgs: {t_insert*1000:.0f}ms")
        print(f"  Memory: {est/1024/1024:.1f}MB, chains={chain_count}")

        # 关键断言：200 chains + 生产级消息 → 应 <200MB
        assert est < 200 * 1024 * 1024, \
            f"Memory {est/1024/1024:.1f}MB exceeds 200MB cap"

        # maybe_maintain 不应阻塞
        t_m = time.monotonic()
        store.maybe_maintain()
        t_maintain = (time.monotonic() - t_m) * 1000
        print(f"  maybe_maintain: {t_maintain:.1f}ms")
        assert t_maintain < 100, f"maybe_maintain blocked: {t_maintain:.0f}ms"

    def test_realistic_cluster_7days(self, frozen_now, rng):
        """模拟 7 天生产流量：300 sessions × 多种 client。

        生产数据:
          claude-code:      236 sessions, avg 288 reqs, max 11,505 reqs
          vscode-copilot:    42 sessions, avg 162 reqs, max 655 reqs
          hermes:            75 sessions, avg  83 reqs, max 819 reqs
        """
        store = MemoryStore(
            ttl_hours=168,
            max_memory_bytes=200 * 1024 * 1024,
        )
        store._now = lambda: frozen_now
        store._last_eviction_check = 0

        clients = {
            "claude-code": {"sessions": 236, "avg_reqs": 288, "copies": 30},
            "vscode-copilot-deepseek": {"sessions": 42, "avg_reqs": 162, "copies": 8},
            "hermes": {"sessions": 75, "avg_reqs": 83, "copies": 12},
        }

        total_chains = 0
        total_reqs = 0
        t0 = time.monotonic()
        peak_memory = 0
        memory_samples = []

        for client_type, cfg in clients.items():
            for si in range(cfg["copies"]):
                sid = f"{client_type}_{si}"
                # 随机化 session 时间：0-7天前
                days_ago = rng.randint(0, 7)
                session_time = frozen_now - timedelta(days=days_ago, hours=rng.randint(0, 23))
                store._now = lambda t=session_time: t  # noqa: B023

                store._sessions[sid] = SessionRecord(
                    session_id=sid,
                    last_seen_time=session_time.isoformat(),
                )

                # 每 session 的 chains（基于真实分布）
                n_chains = _weighted_choice(rng, CHAINS_PER_SESSION_WEIGHTS)
                for ci in range(n_chains):
                    chain_len = _weighted_choice(rng, CHAIN_LENGTH_WEIGHTS)
                    msgs = []
                    for _ in range(chain_len):
                        msgs.append({"role": rng.choice(["user", "assistant"]),
                                     "content": _random_content(rng)})
                    total_reqs += chain_len

                    n_tools = rng.randint(1, min(100, client_type == "claude-code" and 200 or 80))
                    c = ChainRecord(
                        chain_id=f"{sid}_c{ci}",
                        session_id=sid,
                        sys='[{"role":"system","content":"' + _random_content(rng)[:200] + '"}]',
                        tools="[" + ",".join([_random_content(rng) for _ in range(n_tools)]) + "]",
                        messages=orjson.dumps(msgs).decode(),
                        sys_hashes=_random_hash_chain(1, rng),
                        tools_hashes=_random_hash_chain(n_tools, rng),
                        messages_hashes=_random_hash_chain(chain_len, rng),
                        msg_count=chain_len,
                        last_seen=session_time.isoformat(),
                    )
                    fk = store.compute_family_key(c.sys_hashes, c.tools_hashes)
                    store._chains.setdefault(sid, {}).setdefault(fk, []).append(c)
                    total_chains += 1

                # 每次 session 后采样内存
                est = estimate_bytes(store._sessions, store._chains)
                memory_samples.append(est)
                if est > peak_memory:
                    peak_memory = est

        # 恢复时间
        store._now = lambda: frozen_now

        t_insert = time.monotonic() - t0

        # TTL 淘汰 7 天外的 session
        store._last_eviction_check = 0
        stats = store.maybe_maintain()

        final_est = estimate_bytes(store._sessions, store._chains)
        final_chains = sum(len(clist) for fmap in store._chains.values() for clist in fmap.values())

        print(f"\n  === 7-day cluster simulation ===")
        print(f"  Total: {total_reqs} reqs, {total_chains} chains")
        print(f"  Insert time: {t_insert:.1f}s")
        print(f"  Peak memory: {peak_memory/1024/1024:.1f}MB")
        print(f"  Final memory: {final_est/1024/1024:.1f}MB ({final_chains} chains)")
        print(f"  TTL evicted: {stats.stale_evicted} sessions")
        print(f"  LRU evicted: {stats.lru_evicted} chains")
        print(f"  Memory samples: {[s/1024/1024 for s in memory_samples[:5]]}...")

        # 断言
        assert peak_memory < 200 * 1024 * 1024 * 1.2, \
            f"Peak {peak_memory/1024/1024:.1f}MB exceeds cap"
        assert t_insert < 60, f"Insert too slow: {t_insert:.1f}s"

    def test_power_law_message_sizes(self, frozen_now, rng):
        """验证对不同大小消息的处理稳定性。

        生产数据: 50% <500B, 31% 500-2KB, 14% 2-10KB, 4% 10-50KB, 1% 50KB+
        """
        store = MemoryStore(max_memory_bytes=50 * 1024 * 1024)
        store._now = lambda: frozen_now
        store._last_eviction_check = 0

        # 创建不同大小的 chains
        sizes = []
        for _ in range(100):
            msgs = []
            for _ in range(_weighted_choice(rng, CHAIN_LENGTH_WEIGHTS)):
                msgs.append({"role": "user", "content": _random_content(rng)})
            msg_str = orjson.dumps(msgs).decode()
            sizes.append(len(msg_str))

            c = ChainRecord(
                chain_id=f"c{_}",
                messages=msg_str,
                sys_hashes=_random_hash_chain(1, rng),
                tools_hashes=_random_hash_chain(10, rng),
                messages_hashes=_random_hash_chain(len(msgs), rng),
                last_seen=frozen_now.isoformat(),
            )
            fk = store.compute_family_key(c.sys_hashes, c.tools_hashes)
            store._chains.setdefault("s1", {}).setdefault(fk, []).append(c)

        # 统计
        size_buckets = {"<1KB": 0, "1-10KB": 0, "10-50KB": 0, "50-100KB": 0, ">100KB": 0}
        for s in sizes:
            if s < 1024: size_buckets["<1KB"] += 1
            elif s < 10240: size_buckets["1-10KB"] += 1
            elif s < 51200: size_buckets["10-50KB"] += 1
            elif s < 102400: size_buckets["50-100KB"] += 1
            else: size_buckets[">100KB"] += 1

        print(f"\n  Message size distribution (100 chains):")
        for bucket, cnt in size_buckets.items():
            print(f"    {bucket:10s}: {cnt}")

        est = estimate_bytes(store._sessions, store._chains)
        print(f"  Total memory: {est/1024/1024:.1f}MB")

        # estimate_bytes 应对所有大小都稳定（不抛异常）
        assert est > 0

    async def test_moments_growth_over_long_session(self, frozen_now, rng):
        """验证超长 session 的 moments 不爆炸。

        生产数据: claude-code 单 chain 最多 435 reqs → 435 moments
        max_moments=100 → 内存保留 100 条，不随请求数增长
        """
        store = MemoryStore(max_moments=100)
        store._now = lambda: frozen_now

        rec = ChainRecord(
            chain_id="long_chain", session_id="s1",
            moments="[]",
            sys_hashes="aaaa,bbbb",
            tools_hashes="cccc",
            last_seen=frozen_now.isoformat(),
        )
        fk = store.compute_family_key(rec.sys_hashes, rec.tools_hashes)
        store._chains.setdefault("s1", {}).setdefault(fk, []).append(rec)

        max_observed_moments_len = 0
        moments_sizes = []

        for i in range(500):  # > 真实 max=435
            await store.add_moment("long_chain", ChainMoment(
                messages_count=i, op="update",
                total_tokens=rng.randint(1000, 50000),
                cache_hit_tokens=rng.randint(0, 50000),
                cache_miss_tokens=rng.randint(0, 1000),
                char_count=rng.randint(500, 50000),
                timestamp=frozen_now.isoformat(),
            ))
            m = orjson.loads(rec.moments)
            max_observed_moments_len = max(max_observed_moments_len, len(m))
            moments_sizes.append(len(rec.moments))

        print(f"\n  After 500 moments: {max_observed_moments_len} in memory (cap=100)")
        print(f"  Moments string sizes: {moments_sizes[0]}→{moments_sizes[-1]} bytes")
        print(f"  Truncated total: {store._moments_truncated_total}")

        assert max_observed_moments_len <= 100
        assert store._moments_truncated_total > 0
        # 验证收敛：最后 100 次写入不增长
        last_100_sizes = moments_sizes[-100:]
        assert max(last_100_sizes) - min(last_100_sizes) < 1000, \
            f"Moments should converge, range: {min(last_100_sizes)}-{max(last_100_sizes)}"

    def test_tools_change_creates_family_branches(self, frozen_now, rng):
        """验证 tools 变化产生的 family_key 分支不影响内存控制。

        生产数据: 部分 session 有 10-20 chains（多分支）。
        每个 family_key 是一个独立分支，LRU 淘汰跨分支排序。
        """
        store = MemoryStore(max_memory_bytes=2 * 1024 * 1024)
        store._now = lambda: frozen_now
        store._last_eviction_check = 0

        sid = "multi_branch"
        store._sessions[sid] = SessionRecord(
            session_id=sid,
            last_seen_time=frozen_now.isoformat(),
        )

        # 模拟连续的 tools 变更：每 10 个 request 产生一个新分支
        n_branches = 15  # 生产数据极端: 15-20 chains/session
        branches_created = 0

        for bi in range(n_branches):
            # 每次 tools 变更，sys 不变，tools hash 变 → 新 family_key
            sys_hash = "aaaa"  # sys 稳定
            tools_hashes = _random_hash_chain(rng.randint(10, 80), rng)

            c = ChainRecord(
                chain_id=f"{sid}_b{bi}",
                session_id=sid,
                sys=_random_content(rng)[:500],
                tools="[" + ",".join([_random_content(rng) for _ in range(rng.randint(1, 30))]) + "]",
                messages=orjson.dumps([
                    {"role": "user", "content": _random_content(rng)}
                    for _ in range(rng.randint(5, 50))
                ]).decode(),
                sys_hashes=sys_hash,
                tools_hashes=tools_hashes,
                messages_hashes=_random_hash_chain(50, rng),
                msg_count=50,
                last_seen=frozen_now.isoformat(),
            )
            fk = store.compute_family_key(c.sys_hashes, c.tools_hashes)
            store._chains.setdefault(sid, {}).setdefault(fk, []).append(c)
            branches_created += 1

        # 统计 family_key 数量
        n_families = len(store._chains.get(sid, {}))
        print(f"\n  Branches created: {branches_created}, families: {n_families}")

        # 即使分支多，LRU 也能跨家族淘汰
        store.maybe_maintain()
        est = estimate_bytes(store._sessions, store._chains)
        remaining_chains = sum(len(clist) for clist in store._chains.get(sid, {}).values())

        print(f"  After LRU: {remaining_chains} chains, {est/1024/1024:.1f}MB")
        assert est < store.max_memory_bytes * 1.5

    def test_restart_recovery_with_realistic_db(self, frozen_now, rng):
        """模拟重启：DB 中有 30 天数据，启动只加载近期窗口。

        生产数据: 94K reqs across ~360 sessions over 30 days
        启动窗口 7 天 → 只加载活跃 session
        """
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        try:
            conn = sqlite3.connect(db_path)
            from deepseek_tap_proxy.shared.db import (
                create_sessions_table, create_prefix_chains_table,
                create_requests_table, create_chain_token_history_table,
            )
            create_sessions_table(conn)
            create_prefix_chains_table(conn)
            create_requests_table(conn)
            create_chain_token_history_table(conn)

            # 模拟 30 天数据：360 sessions × 94K reqs
            n_old = 0
            n_recent = 0

            for day_offset in range(30):
                ts = (frozen_now - timedelta(days=day_offset)).isoformat()
                # 每天 ~12 sessions，~3000 reqs
                for si in range(12):
                    sid = f"day{day_offset}_s{si}"
                    conn.execute(
                        "INSERT INTO sessions (session_id, client_type, last_seen_time, start_time)"
                        " VALUES (?, ?, ?, ?)",
                        (sid, "claude-code", ts, ts),
                    )
                    # 每 session ~8 chains
                    for ci in range(rng.randint(1, 8)):
                        conn.execute(
                            "INSERT INTO prefix_chains"
                            " (session_id, chain_id, sys, tools, messages,"
                            "  sys_hashes, tools_hashes, messages_hashes,"
                            "  msg_count, created_at, last_seen, moments)"
                            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (sid, f"{sid}_c{ci}", "[]", "[]", "[]",
                             "aaaa", "bbbb", "cccc", 10, ts, ts, "[]"),
                        )
                    if day_offset < 7:
                        n_recent += 1
                    else:
                        n_old += 1
            conn.commit()
            conn.close()

            from deepseek_tap_proxy.db import Database
            db = Database(db_path)
            db.init_db()
            db.close()

            # 启动：7 天窗口
            store = asyncio.run(MemoryStore.load_from(
                db, _db=db, load_window_hours=168,
            ))

            session_count = len(store._sessions)
            total_in_db = n_old + n_recent

            print(f"\n  DB: {total_in_db} sessions total, loaded {session_count}")
            print(f"  Recent (≤7d): {n_recent}, Old (>7d): {n_old}")

            # 只加载窗口内数据（不应全量加载）
            assert session_count > 0, "Should load at least some sessions"
            assert session_count < total_in_db, \
                f"Should filter old data: loaded {session_count} < total {total_in_db}"

        finally:
            try:
                os.unlink(db_path)
            except PermissionError:
                pass
