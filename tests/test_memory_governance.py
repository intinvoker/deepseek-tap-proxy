"""Tests for memory_governance module — pure function, 0 I/O, deterministic."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

import orjson
import pytest

from deepseek_tap_proxy.shared.memory_governance import (
    estimate_bytes,
    evict_stale_sessions,
    evict_lru,
    truncate_moments,
    EvictionStats,
)
from deepseek_tap_proxy.shared.store import MemoryStore, SessionRecord, ChainRecord


# ── 测试用轻量 dataclass ──

@dataclass
class FakeSessionRecord:
    session_id: str = ""
    last_seen_time: str = ""


@dataclass
class FakeChainRecord:
    chain_id: str = ""
    session_id: str = ""
    sys: str = "[]"
    tools: str = "[]"
    messages: str = "[]"
    sys_hashes: str = ""
    tools_hashes: str = ""
    messages_hashes: str = ""
    moments: str = "[]"
    last_seen: str = ""


class TestEstimateBytes:
    def test_empty_store_returns_zero(self):
        assert estimate_bytes({}, {}) == 0

    def test_sessions_only(self):
        sessions = {"s1": FakeSessionRecord(session_id="s1")}
        est = estimate_bytes(sessions, {})
        assert est == 512

    def test_grows_with_chains(self):
        chains = {
            "s1": {
                "fk1": [
                    FakeChainRecord(
                        chain_id="c1",
                        sys='[{"role":"system","content":"hello"}]',
                        tools='[{"type":"function"}]',
                        messages='[{"role":"user","content":"hi"}]',
                    )
                ]
            }
        }
        est = estimate_bytes({}, chains)
        assert est > 256

    def test_shrinks_after_removal(self):
        chains = {
            "s1": {
                "fk1": [FakeChainRecord(chain_id="c1", messages="x" * 1000)]
            }
        }
        est_before = estimate_bytes({}, chains)
        chains.clear()
        est_after = estimate_bytes({}, {})
        assert est_after < est_before


class TestTruncateMoments:
    def test_truncates_excess_moments(self):
        moments = [{"op": "update", "timestamp": f"2026-07-{i:02d}T00:00:00Z"}
                   for i in range(1, 151)]
        rec = FakeChainRecord(moments=orjson.dumps(moments).decode())
        n = truncate_moments(rec, max_moments=100)
        assert n == 50
        remaining = orjson.loads(rec.moments)
        assert len(remaining) == 100

    def test_keeps_newest(self):
        moments = [{"op": "update", "n": i} for i in range(200)]
        rec = FakeChainRecord(moments=orjson.dumps(moments).decode())
        truncate_moments(rec, max_moments=100)
        remaining = orjson.loads(rec.moments)
        assert remaining[0]["n"] == 100
        assert remaining[-1]["n"] == 199

    def test_no_truncation_within_limit(self):
        moments = [{"op": "update", "n": i} for i in range(50)]
        rec = FakeChainRecord(moments=orjson.dumps(moments).decode())
        n = truncate_moments(rec, max_moments=100)
        assert n == 0
        assert len(orjson.loads(rec.moments)) == 50

    def test_empty_moments(self):
        rec = FakeChainRecord(moments="[]")
        n = truncate_moments(rec, max_moments=100)
        assert n == 0

    def test_invalid_json_returns_zero(self):
        rec = FakeChainRecord(moments="not json")
        n = truncate_moments(rec, max_moments=100)
        assert n == 0


class TestEvictStaleSessions:
    def _frozen_now(self):
        return datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_stale_session_evicted(self):
        now = self._frozen_now()
        sessions = {
            "old": FakeSessionRecord(
                session_id="old",
                last_seen_time=(now - timedelta(days=8)).isoformat(),
            ),
            "recent": FakeSessionRecord(
                session_id="recent",
                last_seen_time=(now - timedelta(days=1)).isoformat(),
            ),
        }
        chains = {
            "old": {"fk1": [FakeChainRecord(chain_id="c1")]},
            "recent": {"fk2": [FakeChainRecord(chain_id="c2")]},
        }
        n = evict_stale_sessions(sessions, chains, now=now, ttl_hours=168)
        assert n == 1
        assert "old" not in sessions
        assert "old" not in chains
        assert "recent" in sessions
        assert "recent" in chains

    def test_active_session_kept(self):
        now = self._frozen_now()
        sessions = {
            "s1": FakeSessionRecord(
                session_id="s1",
                last_seen_time=(now - timedelta(hours=1)).isoformat(),
            ),
        }
        chains = {"s1": {"fk1": [FakeChainRecord(chain_id="c1")]}}
        n = evict_stale_sessions(sessions, chains, now=now, ttl_hours=168)
        assert n == 0
        assert "s1" in sessions

    def test_empty_dicts_no_error(self):
        n = evict_stale_sessions({}, {}, now=self._frozen_now(), ttl_hours=168)
        assert n == 0

    def test_missing_last_seen_kept(self):
        now = self._frozen_now()
        sessions = {"s1": FakeSessionRecord(session_id="s1", last_seen_time="")}
        chains = {}
        n = evict_stale_sessions(sessions, chains, now=now, ttl_hours=168)
        assert n == 0
        assert "s1" in sessions

    def test_invalid_last_seen_format_kept(self):
        """非法日期格式的 session 保留（不淘汰）。"""
        sessions = {
            "s1": FakeSessionRecord(session_id="s1", last_seen_time="not-a-date")
        }
        n = evict_stale_sessions(sessions, {}, now=self._frozen_now(), ttl_hours=168)
        assert n == 0
        assert "s1" in sessions


class TestEvictLRU:
    def _make_chain(self, cid, days_ago, messages_len=500):
        ts = (datetime(2026, 8, 1, tzinfo=timezone.utc)
              - timedelta(days=days_ago)).isoformat()
        return FakeChainRecord(
            chain_id=cid, last_seen=ts,
            messages="x" * messages_len,
        )

    def test_evicts_oldest_first(self):
        chains = {
            "s1": {"fk1": [
                self._make_chain("c1", 5),
                self._make_chain("c2", 3),
                self._make_chain("c3", 1),
            ]}
        }
        n = evict_lru({}, chains, max_memory=1024)
        assert n > 0
        remaining_ids = [
            c.chain_id
            for clist in chains.get("s1", {}).values()
            for c in clist
        ]
        assert "c1" not in remaining_ids

    def test_stops_at_threshold(self):
        chains = {
            "s1": {
                "fk1": [self._make_chain(f"c{i}", i, messages_len=2000)
                        for i in range(1, 6)]
            }
        }
        est_before = estimate_bytes({}, chains)
        n = evict_lru({}, chains, max_memory=est_before // 2)
        assert n > 0
        est_after = estimate_bytes({}, chains)
        assert est_after <= est_before // 2

    def test_no_eviction_below_threshold(self):
        chains = {
            "s1": {"fk1": [self._make_chain("c1", 1, messages_len=100)]}
        }
        n = evict_lru({}, chains, max_memory=10 * 1024 * 1024)
        assert n == 0
        assert "s1" in chains

    def test_empty_store_no_error(self):
        n = evict_lru({}, {}, max_memory=1024)
        assert n == 0

    def test_cleans_up_empty_family_keys(self):
        chains = {
            "s1": {"fk1": [self._make_chain("c1", 5, messages_len=10000)]}
        }
        n = evict_lru({}, chains, max_memory=1024)
        assert n > 0
        assert "fk1" not in chains.get("s1", {})
        assert "s1" not in chains


class TestGovernanceToggle:
    """验证 enable_governance 开关。"""

    def test_disabled_skips_eviction(self):
        """enable_governance=False → maybe_maintain 返回零统计，不淘汰。"""
        from datetime import datetime, timezone, timedelta

        store = MemoryStore(enable_governance=False)
        now = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
        store._now = lambda: now
        store._last_eviction_check = 0

        # 插入 8 天前的 session（本应被 TTL 淘汰）
        sid = "old_session"
        store._sessions[sid] = SessionRecord(
            session_id=sid,
            last_seen_time=(now - timedelta(days=8)).isoformat(),
        )
        store._chains[sid] = {"fk1": [
            ChainRecord(chain_id="c1", messages="x" * 100000, last_seen=now.isoformat())
        ]}

        stats = store.maybe_maintain()

        assert stats.stale_evicted == 0
        assert stats.lru_evicted == 0
        assert sid in store._sessions  # 未被淘汰

    def test_enabled_by_default(self):
        """默认 enable_governance=True，maybe_maintain 正常执行。"""
        store = MemoryStore()
        assert store.enable_governance is True
        stats = store.maybe_maintain()
        assert isinstance(stats, EvictionStats)
