"""Chain 状态事件日志 — 权威事实源（Garnet AOF 理念）。

事件 append 到跨重启稳定的 chain-events.jsonl，SQLite 仅作 checkpoint 投影。
"""
from __future__ import annotations

import glob
import json
import logging
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger("deepseek-tap")

# 单次写失败后的重试次数（不包含首次尝试）—— 简单指数退避（与 store.py 同构）
_MAX_RETRIES = 2
_RETRY_DELAYS = [0.1, 0.5]  # 100ms → 500ms


@dataclass
class ChainEvent:
    evt: str          # chain_upsert / chain_moment / canonical_set / session_meta
    seq: int
    data: dict[str, Any] = field(default_factory=dict)
    ts: str = ""


def serialize_event(evt: str, seq: int, data: dict[str, Any], ts: str = "") -> str:
    return json.dumps(
        {"evt": evt, "seq": seq, "ts": ts or datetime.now(timezone.utc).isoformat(), "data": data},
        ensure_ascii=False,
    ) + "\n"


def parse_event(line: str) -> ChainEvent | None:
    """解析一行事件。行损坏（JSON 解析失败）返回 None —— append-only 尾部容错。"""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
        return ChainEvent(
            evt=obj["evt"],
            seq=int(obj["seq"]),
            data=obj.get("data", {}),
            ts=obj.get("ts", ""),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _ordered_event_paths(base: str) -> list[str]:
    """返回 base 及全部轮转文件路径，按日志时序升序（base → base-2 → base-3 …）。

    轮转文件名形如 ``base-2.jsonl``、``base-3.jsonl`` —— 序号递增即日志时序递增
    （新段承接旧段之后的事件）。必须按**数字**升序排序：字典序会把 base-10 排在
    base-2 之前，导致重放时序错乱（canonical 后写胜出语义被破坏）。
    """
    base_no_suffix = base.removesuffix(".jsonl")
    pattern = f"{base_no_suffix}-*.jsonl"
    rotated: list[tuple[int, str]] = []
    for p in glob.glob(pattern):
        m = re.fullmatch(re.escape(base_no_suffix) + r"-(\d+)\.jsonl", p)
        if m:
            rotated.append((int(m.group(1)), p))
    rotated.sort()
    return [base] + [p for _, p in rotated]


class ChainEventWriter:
    """单写者事件写入器：同步 append 入队，后台线程批量 flush 到 JSONL。

    与 MemoryStore 的 SQLite 单写者模式同构 —— 同步方法可直接调用
    ``append()``（不阻塞请求热路径）。
    """

    _BATCH_MAX = 500

    def __init__(self, path: str, file_max_bytes: int = 50 * 1024 * 1024) -> None:
        self._base_path = path
        self._path = path
        self._file_max_bytes = file_max_bytes
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._seq = 0
        self._f = None
        self._thread: threading.Thread | None = None
        self._file_count = 1
        self.failures = 0

    @property
    def path(self) -> str:
        return self._path

    @property
    def seq(self) -> int:
        return self._seq

    def _open(self) -> None:
        try:
            # 自定义 --events-path 指向不存在目录时自动创建，避免静默丢权威日志
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            self._f = open(self._path, "a", encoding="utf-8", buffering=1)
        except OSError as exc:
            # 目录不可写 / 创建失败等 → 记录失败而非崩溃（故障统计路径）
            log.error("ChainEventWriter: open %s failed: %s", self._path, exc)
            self.failures += 1
            self._f = None

    def _rotated_paths(self) -> list[str]:
        """返回 base 及全部轮转文件路径（供 seq 恢复扫描 / 归档）。"""
        return _ordered_event_paths(self._base_path)

    def _restore_seq(self) -> None:
        """扫描 base + 全部轮转文件的最大 seq，保证跨重启单调（轮转后不冲突）。"""
        last_seq = None
        for p in self._rotated_paths():
            if not os.path.exists(p):
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        evt = parse_event(line)
                        if evt is not None:
                            last_seq = evt.seq if last_seq is None else max(last_seq, evt.seq)
            except OSError:
                continue
        if last_seq is not None:
            self._seq = last_seq + 1

    def start(self) -> None:
        self._restore_seq()
        self._open()
        self._thread = threading.Thread(target=self._thread_loop, daemon=True, name="chain-events-writer")
        self._thread.start()

    def append(self, evt: str, data: dict[str, Any]) -> None:
        """同步入队（不阻塞）。seq 在锁内分配，保证多生产者下单调不重复。"""
        with self._lock:
            seq = self._seq
            line = serialize_event(evt, seq, data)
            self._seq += 1
        self._q.put((line, seq))

    def _write_batch(self, batch: list[str]) -> None:
        if self._f is None:
            log.error("ChainEventWriter: file not open, dropping %d events", len(batch))
            self.failures += 1
            return
        data = "".join(batch)
        last_exc = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                self._f.write(data)
                self._f.flush()
                self.failures = 0  # 成功 → 重置连续失败计数（瞬态失败自愈）
                return
            except OSError as exc:
                last_exc = exc
                log.warning(
                    "ChainEventWriter: append failed (attempt %d/%d): %s",
                    attempt + 1, _MAX_RETRIES + 1, exc,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAYS[attempt])
        log.error(
            "ChainEventWriter: append failed after %d attempts: %s",
            _MAX_RETRIES + 1, last_exc,
        )
        self.failures += 1

    def _maybe_rotate(self) -> None:
        if self._f is None or self._f.tell() < self._file_max_bytes:
            return
        try:
            self._f.close()
        except OSError:
            pass
        self._file_count += 1
        # 从 base 计算轮转文件名，绝不基于 self._path 累加（避免 -2-3-4 累积）
        self._path = f"{self._base_path.removesuffix('.jsonl')}-{self._file_count}.jsonl"
        self._open()

    def _thread_loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:  # sentinel 关闭信号
                break
            line, _ = item
            batch = [line]
            while len(batch) < self._BATCH_MAX:
                try:
                    nxt = self._q.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:  # 批内遇到 sentinel → flush 后退出（先取后判再解包，避免崩溃）
                    self._write_batch(batch)
                    self._maybe_rotate()
                    return
                line, _ = nxt
                batch.append(line)
            self._write_batch(batch)
            self._maybe_rotate()

    def flush_sync(self) -> None:
        """同步排空队列写入（测试 / 关闭前用）。"""
        batch = []
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                # sentinel 放回队列，交由后台线程消化（否则与 close 并发时线程等不到 sentinel 挂起）
                self._q.put(None)
                break
            line, _ = item
            batch.append(line)
        if batch:
            self._write_batch(batch)
            self._maybe_rotate()

    def get_offset(self) -> int:
        """返回当前活跃文件的已写字节偏移。

        须配合 ``.path`` 属性定位文件 —— checkpoint 需要 (path, offset) 二元组。
        """
        try:
            return self._f.tell() if self._f else 0
        except OSError:
            return 0

    def close(self) -> None:
        if self._thread is not None:
            self._q.put(None)  # sentinel 唤醒线程，避免等 q.get(timeout) 超时
            self._thread.join(timeout=5)
            self._thread = None
        self.flush_sync()
        if self._f is not None:
            try:
                self._f.close()
            except OSError:
                pass
            self._f = None

    def archive_old_segments(self, keep_from_offset: int = 0) -> None:
        """归档旧日志段（checkpoint 后调用）——checkpoint 即 AOF 压缩点（先投影后截断）。

        ``keep_from_offset=0``（默认）：删除 base + 全部轮转文件，重建空 base（完全压缩）。
        ``keep_from_offset>0``：仅删除该偏移之前的日志段（已被 checkpoint 投影），保留
        该偏移之后的尾部 —— 快照点之后新 append 的事件不因归档而丢，写入新 base 供重启
        从 offset 0 重放。配合 K1 顺序（checkpoint 先提交投影再归档），崩溃任何时刻都
        安全：日志完整 + 投影 + 重放，或最新投影 + 空日志，至少一方能重建内存。

        seq 保留（同进程内跨归档单调），文件偏移从 0 重新开始（checkpoint_offset
        应配合重置为 0）。生命周期与 close() 对称。
        """
        # 1. 优雅停止后台线程（sentinel 唤醒 → flush → 退出）
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                # I2：线程超时未退出（如 _write_batch 卡 I/O）→ 不重启新线程，避免
                # 双消费者抢队列造成丢事件/线程泄漏。checkpoint 已提交，日志未截断
                # 也安全（replay 幂等），下次周期 checkpoint 重试。
                log.warning(
                    "archive_old_segments: writer thread failed to stop in 5s, "
                    "skip archive to avoid double consumer"
                )
                return
            self._thread = None
        self.flush_sync()
        # 2. 关闭当前文件句柄（可能指向轮转文件）
        if self._f is not None:
            try:
                self._f.close()
            except OSError:
                pass
            self._f = None
        # 3. 保留快照点之后的尾部（checkpoint 期间新 append 的事件），删除其余段
        tail = ""
        if keep_from_offset > 0:
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    f.seek(keep_from_offset)
                    tail = f.read()
            except OSError:
                tail = ""  # 读失败 → 回退完全压缩（checkpoint 已提交，安全）
        for p in self._rotated_paths():
            try:
                os.remove(p)
            except OSError:
                pass
        # 4. 重建 base：有尾部则写入，否则空 base（路径/计数重置，seq 保留单调）
        self._path = self._base_path
        self._file_count = 1
        if tail:
            try:
                with open(self._base_path, "w", encoding="utf-8") as f:
                    f.write(tail)
            except OSError:
                pass
        self._open()  # 追加模式打开（尾部之后继续 append）
        # 5. 重启后台线程
        self._thread = threading.Thread(target=self._thread_loop, daemon=True,
                                        name="chain-events-writer")
        self._thread.start()


# ═══════════════════════════════════════════════════════════════════════
#  重放器
# ═══════════════════════════════════════════════════════════════════════


def replay_events(path: str, from_offset: int, into) -> int:
    """从 from_offset 重放事件日志到 ``into``（MemoryStore）。

    跨轮转段重放：先读 base（``path``，从 ``from_offset`` 起），再按序号升序
    整段读全部轮转段（``base-2.jsonl``、``base-3.jsonl`` …）。archive 保证
    checkpoint 后 base 归 0（offset=0）；崩溃窗口内 base 完整从 offset 续读，
    轮转段全读 —— 轮转 → checkpoint 之间的状态变更不丢。

    返回最后成功应用的事件 seq（无事件则 -1）。损坏行跳过（append-only 容错：
    崩溃半行 / offset 落在行中间 / 语义损坏行均不中断重放）。重放时经
    ``_persist_event=False`` 避免回写事件日志（防循环）。
    """
    paths = [p for p in _ordered_event_paths(path) if os.path.exists(p)]
    if not paths:
        return -1
    last_seq = -1
    for i, p in enumerate(paths):
        offset = from_offset if i == 0 else 0  # from_offset 只对 base 生效
        with open(p, "r", encoding="utf-8") as f:
            f.seek(offset)
            for line in f:
                evt = parse_event(line)
                if evt is None:
                    log.warning("replay: 损坏行跳过（非 JSON / 残缺行）: %r", line[:80])
                    continue
                try:
                    _apply_event(evt, into)
                    last_seq = evt.seq
                except Exception as exc:
                    # 语义损坏（如 msg_count: null → int(None)）不中断整段重放：
                    # 记日志跳过该事件，继续后续行。
                    log.warning("replay: 事件 seq=%s 应用失败，跳过: %s", evt.seq, exc)
                    continue
    return last_seq


def _apply_event(evt: ChainEvent, store) -> None:
    """应用单个事件到 store。事件记录"已生效终态"（任务 4 修正后语义）。"""
    data = evt.data
    if evt.evt == "chain_upsert":
        # 事件里 safe_breakpoints 是 JSON 字符串；upsert_chain 接受 list[int]，
        # 需先反序列化，否则 upsert_chain 内 orjson.dumps 会二次编码。
        sb_raw = data.get("safe_breakpoints")
        if sb_raw:
            try:
                safe_breakpoints = json.loads(sb_raw)
            except (json.JSONDecodeError, TypeError):
                safe_breakpoints = None
        else:
            safe_breakpoints = None
        store.upsert_chain(
            data.get("session_id", ""), data.get("chain_id", ""),
            sys=data.get("sys", "[]"), tools=data.get("tools", "[]"),
            messages=data.get("messages", "[]"),
            sys_hashes=data.get("sys_hashes", ""),
            tools_hashes=data.get("tools_hashes", ""),
            messages_hashes=data.get("messages_hashes", ""),
            msg_count=int(data.get("msg_count", 0)),
            is_frozen=bool(data.get("is_frozen", False)),
            family_key=data.get("family_key", ""),
            safe_breakpoints=safe_breakpoints,
            moments=data.get("moments", "[]"),
            total_input_tokens=int(data.get("total_input_tokens", 0)),
            created_at=data.get("created_at", ""),
            last_seen=data.get("last_seen", ""),
            _persist_event=False,
        )
    elif evt.evt == "chain_moment":
        # 直接构造 moments，不走 add_moment（防重放循环）。append 后按
        # max_moments 截断，使重放态与 live 态一致。
        rec = store.get_chain_record(data.get("chain_id", ""))
        moment = data.get("moment", {})
        if rec is None:
            log.warning("replay: chain_moment 目标链 %s 不存在，跳过",
                        data.get("chain_id", ""))
            return
        if not moment:
            return
        import orjson
        cur = orjson.loads(rec.moments or "[]")
        # 全等去重：仅当已存在完全相同的 moment dict（含 total_tokens/timestamp
        # 全字段）才跳过；同 op 同 messages_count 但 total_tokens 不同的 moment
        # 是合法独立快照，保留。重放幂等靠 dict 全等而非 (op, messages_count) 指纹。
        if all(m != moment for m in cur):
            cur.append(moment)
            rec.moments = orjson.dumps(cur).decode()
            from deepseek_tap_proxy.shared.memory_governance import truncate_moments
            truncate_moments(rec, max_moments=store.max_moments)
    elif evt.evt == "canonical_set":
        # chain_id 由事件携带（旧日志无该字段 → 默认 "" → 归一为默认链，
        # 与迁移前 session 级语义等价）。缺了它子代理的冻结面会被重放进
        # 默认链，把主代理 canonical 覆写 —— 正是本机制要根治的 bug。
        store.set_canonical(
            data.get("session_id", ""),
            chain_id=data.get("chain_id", ""),
            sys=data.get("sys", ""), tools=data.get("tools", ""),
            _persist_event=False,
        )
