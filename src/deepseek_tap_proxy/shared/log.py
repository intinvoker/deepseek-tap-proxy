"""Log and trace output views — application-scenario abstractions over ctx.

Architecture
    ctx                  = data abstraction (pipeline output, single source of truth)
    ctx.persist          = application scenario (DB query / audit, strong-typed)
    LogView(ctx)         = application scenario (terminal + OTLP monitoring)
    TraceView(ctx)       = application scenario (debug / replay archive)

All views hold a reference to ctx and read fields directly — zero copying.
"""

from __future__ import annotations

import asyncio
import threading


def _coerce_keys(obj):
    """Recursively coerce dict keys to str for orjson compatibility."""
    if isinstance(obj, dict):
        return {str(k): _coerce_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_keys(item) for item in obj]
    return obj
import json
import orjson
import time
from datetime import datetime as dt, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from deepseek_tap_proxy.pipeline import Context


# ═══════════════════════════════════════════════════════════════════════
#  LogView — terminal + OTLP per-request summary
# ═══════════════════════════════════════════════════════════════════════


class LogView:
    """Per-request log summary for human (terminal) and AI (OTLP JSONL).

    Holds ctx reference, reads fields directly — zero copying.

    ctx fields read:
        req_id, status, duration_ms, api_format, resp_model, upstream_name,
        session_id, client_type, session_method, chat_type,
        cache_hit_tokens, cache_miss_tokens, lock_type, sys_tools_locked,
        pref_hit_count, orig_sys_len, orig_tools_len, orig_msgs_len,
        chain_action, anchor_moment (total_tokens, char_count), new_char_count,
        persist (lock_type, chain_tokens, timestamp, api_format)
    """

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx

    # ── Terminal (human-readable colored line) ─────────────────────

    def terminal(self) -> str:
        """Format as colored ANSI line for terminal output.

        Dispatches to topic-specific formatters:
          chat          → _terminal_chat (full format)
          count_tokens  → _terminal_count_tokens (lightweight)
          passthrough   → _terminal_passthrough (lightweight + path)
          rejected      → _terminal_rejected (minimal)
        """
        c = self._ctx
        topic = getattr(c, "topic", None) or "?"

        if topic == "count_tokens":
            return self._terminal_count_tokens()
        if topic == "passthrough":
            return self._terminal_passthrough()
        if topic == "rejected":
            return self._terminal_rejected()
        # chat / default: 现有完整格式
        return self._terminal_chat()

    def _terminal_chat(self) -> str:
        """Format full log line for chat requests.

        Structure:
          [{req_id}] ← {status} {duration}s  lock={type}  slock=...  cache=...  pref(stm)=...  est=...  [upstream, model, session, client, intent]
        """
        from deepseek_tap_proxy.shared.colors import colorize, lock_symbol

        c = self._ctx
        r = c.persist

        status = c.status or 0
        duration_s = round((c.duration_ms or 0) / 1000, 1)

        # ── lock ──
        lock_type = r.lock_type or c.meta.get("log_lock_type", "none")
        lock_display = lock_symbol(lock_type)
        lock_kv = f"lock={colorize(lock_display, 'lock', lock_type)}"

        # ── slock (separate k=v, only when action present) ──
        snap = c.sys_tools_snap
        if snap.action == "lock":
            s_b, s_m, s_a = snap.sys.before, snap.sys.match, snap.sys.after
            t_b, t_m, t_a = snap.tools.before, snap.tools.match, snap.tools.after
            slock_parts = ["slock=lock"]
            if s_b != s_a:
                slock_parts.append(f"sys:{s_m}/{s_b}→{s_a}")
            if t_b != t_a:
                slock_parts.append(f"tools:{t_m}/{t_b}→{t_a}")
            slock_kv = " ".join(slock_parts)
        elif snap.action:
            slock_kv = f"slock={snap.action}"
        else:
            slock_kv = ""

        # ── 身份前缀：仅非空时显示（主代理日志零变化）──
        # 追加在 slock 组装「之后」——子代理首次是 init、稳定期是 pass，
        # 只在 action=="lock" 分支里加会让最该被观测的场景没有输出。
        if c.chain_id:
            slock_kv = f"{slock_kv} chain={c.chain_id[:8]}".strip()

        # ── cache ──
        cache_hit = r.cache_hit_tokens or c.meta.get("cache_hit_tokens", 0)
        cache_miss = r.cache_miss_tokens or c.meta.get("cache_miss_tokens", 0)
        cache_total = cache_hit + cache_miss
        cache_pct = round(cache_hit / cache_total * 100, 2) if cache_total > 0 else 0
        cache_kv = f"cache={cache_hit}/{cache_total}({colorize(f'{cache_pct:.2f}%', 'cache_pct', cache_pct)})"

        # ── pref (three-segment prefix hit) ──
        s_hit = c.meta.get("pref_sys_hit", 0)
        t_hit = c.meta.get("pref_tools_hit", 0)
        m_hit = c.meta.get("pref_msgs_hit", c.meta.get("pref_hit_count", 0))

        def _seg_color(hit: int, total: int) -> str:
            if total == 0:
                return "0/0"
            pct = int(hit / total * 100)
            return colorize(f"{hit}/{total}", "pref", pct)

        s_str = _seg_color(s_hit, c.orig_sys_len)
        t_str = _seg_color(t_hit, c.orig_tools_len)
        m_str = _seg_color(m_hit, c.orig_msgs_len)
        pref_kv = f"pref(stm)={s_str} {t_str} {m_str}"

        # ── est (prefix-cache token ratio; ? = no token history yet) ──
        anchor = c.anchor_moment
        new_cc = c.new_char_count
        if anchor and anchor.total_tokens > 0 and new_cc > 0:
            avg_tpc = anchor.total_tokens / anchor.char_count
            new_tokens_est = int(new_cc * avg_tpc)
            est_denom = anchor.total_tokens + new_tokens_est
            est_pct = round(anchor.total_tokens / est_denom * 100, 2)
            est_str = f"{anchor.total_tokens}/{est_denom}({est_pct:.2f}%)"
        else:
            est_pct = 0.0
            est_str = "?"
        est_kv = f"est={colorize(est_str, 'cache_pct', est_pct)}"

        # ── sid (session id) ──
        sess_id = c.session_id or ""
        sid_kv = f"sid={sess_id}"

        # ── trailing bracket: metadata ──
        req_m = c.req_model or "?"
        resp_m = c.resp_model or "?"
        model = f"{req_m}({resp_m})" if req_m != resp_m else req_m
        upstream = c.meta.get("upstream_name", "default")
        intent = c.chat_type
        if c.meta.get("compaction_fallback_blocked"):
            intent = "compaction-blocked"

        # ── assemble: header  k=v  k=v  ...  [meta, meta, ...] ──
        header = (
            f"[{c.req_id}] ← "
            f"{colorize(str(status), 'status', status)} "
            f"{colorize(f'{duration_s}s', 'duration', duration_s)}"
        )

        kvs = [lock_kv]
        if slock_kv:
            kvs.append(slock_kv)
        kvs.append(cache_kv)
        kvs.append(pref_kv)
        kvs.append(est_kv)
        kvs.append(sid_kv)

        trailer = f"[{c.format}, {upstream}, {model}, {c.client_type}, {intent}]"

        return "  ".join([header] + kvs + [trailer])

    def _terminal_count_tokens(self) -> str:
        """count_tokens 的 terminal 输出：轻量，只显示 tokens 数和来源。"""
        c = self._ctx
        source = c.meta.get("tokens_source", "?")
        tokens = c.meta.get("estimated_tokens", 0) or 0
        model = c.req_model or "?"
        sess = c.session_id or ""
        status = c.status or 0
        dur = c.duration_ms or 0
        return (
            f"[{c.req_id}] ← {status} {dur}ms  "
            f"path=count_tokens  tokens={tokens}  sid={sess}  "
            f"[{c.format}, count_tokens, {source}, {model}]"
        )

    def _terminal_passthrough(self) -> str:
        """passthrough（FIM/GET 等）的 terminal 输出。"""
        c = self._ctx
        upstream = c.meta.get("upstream_name", "default")
        status = c.status or 0
        dur = c.duration_ms or 0
        return (
            f"[{c.req_id}] ← {status} {dur}ms  "
            f"path={c.request.path}  sid={c.session_id or ''}  "
            f"[{c.format}, passthrough, {upstream}]"
        )

    def _terminal_rejected(self) -> str:
        """rejected 请求的 terminal 输出：最简。"""
        c = self._ctx
        return (
            f"[{c.req_id}] ← {c.status or 404} 0ms  "
            f"path={c.request.path}  [{c.format}, rejected]"
        )

    # ── OTLP JSONL (AI consumption) ────────────────────────────────

    def otlp_json(self) -> str:
        """Deprecated: OTLP output now uses _build_otlp_record via OtlpLogWriter.

        Retained for backwards compatibility — old tests and data tools
        still reference this method. Prefer OtlpLogWriter.write(ctx, run_id).
        """
        import warnings
        warnings.warn(
            "otlp_json is deprecated, use OtlpLogWriter with _build_otlp_record",
            DeprecationWarning,
            stacklevel=2,
        )
        c = self._ctx
        r = c.persist
        ca = getattr(c, "chain_action", None)

        cache_hit = r.cache_hit_tokens or c.meta.get("cache_hit_tokens", 0)
        cache_miss = r.cache_miss_tokens or c.meta.get("cache_miss_tokens", 0)

        anchor = c.anchor_moment
        new_cc = c.new_char_count
        if anchor and anchor.total_tokens > 0 and anchor.char_count > 0 and new_cc > 0:
            avg_tpc = anchor.total_tokens / anchor.char_count
            new_tokens_est = int(new_cc * avg_tpc)
            est_denom = anchor.total_tokens + new_tokens_est
            est_pct = round(anchor.total_tokens / est_denom * 100, 2) if est_denom else 0.0
        else:
            est_pct = 0.0

        lock_type = r.lock_type or c.meta.get("log_lock_type", "none")
        upstream = c.meta.get("upstream_name", "default")

        # cache ratio
        cache_total = cache_hit + cache_miss
        cache_pct = round(cache_hit / cache_total * 100, 2) if cache_total > 0 else 0.0

        attrs = [
            _otlp_attr("req_id", c.req_id),
            _otlp_attr("status_code", c.status or 0),
            _otlp_attr("duration_ms", c.duration_ms or 0),
            _otlp_attr("api_format", c.format),
            _otlp_attr("model", c.resp_model or "?"),
            _otlp_attr("request_model", c.req_model or "?"),
            _otlp_attr("client_type", c.client_type),
            _otlp_attr("session_type", c.client_type),
            _otlp_attr("session_method", c.session_method),
            _otlp_attr("session_id", c.session_id or ""),
            _otlp_attr("upstream_name", upstream),
            _otlp_attr("cache_hit_tokens", cache_hit),
            _otlp_attr("cache_miss_tokens", cache_miss),
            _otlp_attr("cache_hit_ratio", cache_pct),
            _otlp_attr("lock_type", lock_type),
            _otlp_attr("pref_hit_count", c.meta.get("pref_hit_count", 0)),
            _otlp_attr("pref_sys_hit", c.meta.get("pref_sys_hit", 0)),
            _otlp_attr("pref_tools_hit", c.meta.get("pref_tools_hit", 0)),
            _otlp_attr("pref_msgs_hit", c.meta.get("pref_msgs_hit", 0)),
            _otlp_attr("orig_sys_len", c.orig_sys_len),
            _otlp_attr("orig_tools_len", c.orig_tools_len),
            _otlp_attr("orig_msgs_len", c.orig_msgs_len),
            _otlp_attr("est_pct", est_pct),
            _otlp_attr("chat_type", c.chat_type),
            _otlp_attr("intent", c.chat_type),
        ]

        sys_tools_locked = c.meta.get("sys_tools_locked", "")
        if sys_tools_locked:
            attrs.append(_otlp_attr("sys_tools_locked", sys_tools_locked))
        # SysToolsSnapshot details
        snap = c.sys_tools_snap
        if snap and snap.action == "lock":
            attrs.append(_otlp_attr("sys_lock_sys_before", snap.sys.before))
            attrs.append(_otlp_attr("sys_lock_sys_match", snap.sys.match))
            attrs.append(_otlp_attr("sys_lock_sys_after", snap.sys.after))
            attrs.append(_otlp_attr("sys_lock_tools_before", snap.tools.before))
            attrs.append(_otlp_attr("sys_lock_tools_match", snap.tools.match))
            attrs.append(_otlp_attr("sys_lock_tools_after", snap.tools.after))
        if ca and ca.action:
            attrs.append(_otlp_attr("chain_action", ca.action))
            attrs.append(_otlp_attr("chain_id", (ca.chain_id or "")[:16]))

        record = {
            "timeUnixNano": int(time.time_ns()),
            "observedTimeUnixNano": int(time.time_ns()),
            "body": {"stringValue": f"request {c.req_id} completed (status={c.status or 0})"},
            "attributes": attrs,
        }
        return orjson.dumps(record).decode()


# ═══════════════════════════════════════════════════════════════════════
#  TraceView — raw request/response archive
# ═══════════════════════════════════════════════════════════════════════


# 截断计数（/health 消费）— 证明写入侧有界的度量闭环
RESP_TRUNCATED_COUNT = 0


def get_truncated_count() -> int:
    return RESP_TRUNCATED_COUNT


def _truncate_strings(obj, limit: int):
    """递归截断 dict/list 中超长字符串（保留结构）。与 viewer 展示层同构。"""
    if isinstance(obj, str):
        if len(obj) > limit:
            return obj[:limit] + f"\n...[truncated, {len(obj)} chars total]"
        return obj
    if isinstance(obj, dict):
        return {k: _truncate_strings(v, limit) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_strings(v, limit) for v in obj]
    return obj


def truncate_resp_body(body, max_bytes: int) -> tuple:
    """结构内截断响应体，使 trace_record 字节有界（单请求内存峰值上限）。

    截断作用于 ctx.resp_body 聚合副本（转发流不受影响）。语义与 viewer
    展示层 _truncate_body 一致：外层结构与头部字段保留，长字符串截断。
    返回 (new_body, was_truncated)。
    """
    if body is None or max_bytes <= 0:
        return body, False
    size = len(orjson.dumps(body))
    if size <= max_bytes:
        return body, False
    # 保守：单字符串上限 = 阈值 1/4（结构开销 + 多字段冗余）
    limit = max(max_bytes // 4, 1024)
    body = _truncate_strings(body, limit)
    body["_truncated"] = True
    body["_original_bytes"] = size
    global RESP_TRUNCATED_COUNT
    RESP_TRUNCATED_COUNT += 1
    return body, True


class TraceView:
    """Raw request/response archive for debugging and replay.

    Holds ctx reference, reads fields directly — zero copying.

    ctx fields read:
        req_id, persist (request_id, timestamp, api_format, tokens, etc),
        target_url, body_readonly, resp_body, request (headers, path),
        resp_headers, status, duration_ms, elapsed_ms
    """

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx

    def as_dict(self) -> dict:
        """Full trace record as dict for JSONL serialization."""
        from deepseek_tap_proxy.shared.headers import redact
        from datetime import datetime as dt, timezone

        c = self._ctx
        r = c.persist

        return {
            "request_id": r.request_id or c.req_id,
            "timestamp": r.timestamp or dt.now(timezone.utc).isoformat(),
            "duration_ms": c.duration_ms or c.elapsed_ms or 0,
            "upstream_base_url": c.target_url,
            "upstream_name": c.meta.get("upstream_name", "default"),
            "api_format": r.api_format,
            "topic": c.topic,
            "mode": r.mode or "managed",
            "kind": r.kind or "",
            "intervention": 1 if r.is_modified else 0,
            "usage_source": r.usage_source or "",
            "request": {
                "method": c.request.method,
                "path": c.request.path_qs,
                "headers": redact(dict(c.request.headers)),
                "body": c.body_readonly,
            },
            "response": {
                "status": c.status,
                "headers": redact(c.resp_headers),
                "body": c.resp_body,
            },
            "usage": {
                "request_id": r.request_id,
                "total_input_tokens": r.total_input_tokens,
                "output_tokens": r.output_tokens,
                "total_tokens": r.total_input_tokens + r.output_tokens,
                "cache_hit_tokens": r.cache_hit_tokens,
                "cache_miss_tokens": r.cache_miss_tokens,
                "cache_hit_ratio": r.cache_hit_ratio,
                "has_reasoning": r.has_reasoning,
                "has_tool_calls": r.has_tool_calls,
                "prompt_messages_count": r.prompt_messages_count,
                "prompt_tools_count": r.prompt_tools_count,
            },
        }


# ═══════════════════════════════════════════════════════════════════════
#  TraceWriter — daily-rotated JSONL for raw trace archive
# ═══════════════════════════════════════════════════════════════════════


class TraceWriter:
    """Buffered, daily-rotated JSONL for raw trace archive.

    Write accumulates records in a buffer; flush happens when
    ``batch_max`` records accumulate OR ``flush_interval`` elapses OR
    ``close()`` is called.  Single-file size capped at ``file_max_bytes``
    (rotates to ``traces-{seq}.jsonl`` within the same day).
    """

    def __init__(self, run_dir: str, run_id: str,
                 flush_interval: float = 2.0, batch_max: int = 500,
                 file_max_bytes: int = 500 * 1024 * 1024) -> None:
        self.run_id = run_id
        self.run_dir = Path(run_dir)
        self._lock = asyncio.Lock()
        self.count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_create_tokens = 0
        self.total_cache_miss_tokens = 0
        self.models_used: dict[str, int] = {}
        self._current_date = ""
        self._file = None
        self._file_seq = 1
        self.path = Path(run_dir) / "traces.jsonl"  # 向后兼容引用
        self._flush_interval = flush_interval
        self._batch_max = batch_max
        self._file_max_bytes = file_max_bytes
        self._pending_count = 0
        self._last_flush = time.monotonic()
        self._flusher_task: asyncio.Task | None = None

    def _open_file(self, today: str) -> None:
        day_dir = self.run_dir / today
        day_dir.mkdir(parents=True, exist_ok=True)
        name = "traces.jsonl" if self._file_seq == 1 else f"traces-{self._file_seq}.jsonl"
        self._file = open(day_dir / name, "ab")
        self.path = day_dir / name

    def _flush_locked(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.flush()
        self._pending_count = 0
        self._last_flush = time.monotonic()

    async def write(self, record: dict) -> int:
        today = dt.now(timezone.utc).strftime("%Y-%m-%d")
        async with self._lock:
            if today != self._current_date:
                self._close_file()
                self._current_date = today
                self._file_seq = 1
                self._open_file(today)
            # 500MB 单文件大小轮转
            if self._file is not None and self._file.tell() >= self._file_max_bytes:
                self._close_file()
                self._file_seq += 1
                self._open_file(today)
            record["run_id"] = self.run_id
            offset = self._file.tell()
            self._file.write(orjson.dumps(_coerce_keys(record)) + b"\n")
            self._pending_count += 1
            self.count += 1
            self._update_stats(record)
            if (self._pending_count >= self._batch_max
                    or time.monotonic() - self._last_flush >= self._flush_interval):
                self._flush_locked()
            return offset

    def _close_file(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.flush()
            self._file.close()
            self._file = None

    async def flush(self) -> None:
        async with self._lock:
            self._flush_locked()

    async def start_flusher(self) -> None:
        if self._flusher_task is not None:
            return
        self._flusher_task = asyncio.create_task(self._flusher_loop())

    async def _flusher_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._flush_interval)
                async with self._lock:
                    if self._pending_count > 0:
                        self._flush_locked()
        except asyncio.CancelledError:
            pass

    async def close(self) -> None:
        if self._flusher_task is not None:
            self._flusher_task.cancel()
            try:
                await self._flusher_task
            except asyncio.CancelledError:
                pass
            self._flusher_task = None
        async with self._lock:
            self._flush_locked()
            self._close_file()

    def _update_stats(self, record: dict) -> None:
        req_body = record.get("request", {}).get("body", {})
        model = req_body.get("model", "unknown") if isinstance(req_body, dict) else "unknown"
        self.models_used[model] = self.models_used.get(model, 0) + 1

        usage = record.get("usage", {})
        self.total_input_tokens += usage.get("total_input_tokens", 0)
        self.total_output_tokens += usage.get("output_tokens", 0)
        self.total_cache_read_tokens += usage.get("cache_hit_tokens", 0)
        self.total_cache_miss_tokens += usage.get("cache_miss_tokens", 0)

    def get_summary(self) -> dict:
        return {
            "api_calls": self.count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cache_hit_tokens": self.total_cache_read_tokens,
            "cache_miss_tokens": self.total_cache_miss_tokens,
            "models_used": self.models_used,
            "run_id": self.run_id,
            "trace_file": str(self.path),
        }


# ═══════════════════════════════════════════════════════════════════════
#  OTLP helpers
# ═══════════════════════════════════════════════════════════════════════


def _otlp_severity(status: int) -> tuple[int, str]:
    """Map HTTP/proxy status to OTLP severity (RFC 7159 LogRecord)."""
    if status >= 500 or status == 0:
        return (17, "ERROR")
    elif status >= 400:
        return (13, "WARN")
    else:
        return (9, "INFO")


def _build_otlp_record(ctx, run_id: str) -> dict:
    """Build a standard OTLP LogRecord dict (not wrapped in ExportLogsServiceRequest).

    Attributes are selected based on ctx.topic:
      chat          → base + full chat-specific fields
      count_tokens  → base + tokens_source, estimated_tokens
      passthrough   → base + upstream_name
      rejected/*    → base only
    """
    status = ctx.status or 0
    severity_number, severity_text = _otlp_severity(status)
    now_ns = time.time_ns()

    attributes = []

    def _str_attr(key: str, value: str):
        attributes.append({"key": key, "value": {"stringValue": value}})

    def _int_attr(key: str, value: int):
        attributes.append({"key": key, "value": {"intValue": value}})

    # ── Base attributes for all topics ──
    _str_attr("run_id", run_id)
    _str_attr("req_id", ctx.req_id)
    _int_attr("status_code", status)
    _int_attr("duration_ms", ctx.duration_ms or 0)
    _str_attr("api_format", ctx.format or "")
    _str_attr("topic", getattr(ctx, "topic", None) or "")
    _str_attr("request_path", getattr(getattr(ctx, "request", None), "path", ""))
    model = getattr(ctx, "resp_model", None) or getattr(ctx, "req_model", None) or "?"
    _str_attr("model", model)
    _str_attr("session_id", ctx.session_id or "")
    _str_attr("kind", "request")
    # passthrough-policy：模式 / 端点类型 / 干预 / usage 可信度（所有 topic 都带）
    _persist = getattr(ctx, "persist", None)
    if _persist is not None:
        _str_attr("mode", getattr(_persist, "mode", "") or "managed")
        _str_attr("request_kind", getattr(_persist, "kind", "") or "")
        _int_attr("intervention", 1 if getattr(_persist, "is_modified", False) else 0)
        if getattr(_persist, "usage_source", ""):
            _str_attr("usage_source", _persist.usage_source)

    # ── Topic-specific attributes ──
    topic = getattr(ctx, "topic", None) or ""
    if topic == "count_tokens":
        _str_attr("tokens_source", ctx.meta.get("tokens_source", "?"))
        _int_attr("estimated_tokens", ctx.meta.get("estimated_tokens", 0))
    elif topic == "passthrough":
        _str_attr("upstream_name", ctx.meta.get("upstream_name", "default"))
    elif topic == "chat":
        attributes.extend(_chat_otlp_attrs(ctx))
    # rejected / unknown: only base attrs

    return {
        "timeUnixNano": str(now_ns),
        "observedTimeUnixNano": str(now_ns),
        "severityNumber": severity_number,
        "severityText": severity_text,
        "body": {"stringValue": f"request {ctx.req_id} completed (status={status})"},
        "attributes": attributes,
        "droppedAttributesCount": 0,
        "flags": 0,
    }


def _chat_otlp_attrs(ctx) -> list[dict]:
    """chat topic 的 OTLP 专属字段 —— OTLP 是持久化超集（除原始 body/headers）。

    字段对齐 PersistRecord。
    """
    from deepseek_tap_proxy.shared.cache_attribution import cache_attribution

    c = ctx
    p = ctx.persist
    ca = getattr(c, "chain_action", None)

    attrs = []

    def s(k, v):
        attrs.append({"key": k, "value": {"stringValue": str(v)}})

    def i(k, v):
        attrs.append({"key": k, "value": {"intValue": int(v)}})

    def f(k, v):
        attrs.append({"key": k, "value": {"doubleValue": float(v)}})

    # 分类/会话
    s("client_type", c.client_type or "")
    s("chat_type", c.chat_type or "")
    s("session_method", c.session_method or "")
    s("upstream_name", c.meta.get("upstream_name", "default"))
    s("session_id", c.session_id or "")
    # session key 来源观测（fingerprint 退役 V2）：plugin=检测器给出 session_id；none=无会话键
    s("session_key_source", c.meta.get("session_key_source", "?"))
    # 链谱系（缓存归因 / 超集分析用）：优先 ctx.chain_id，回退到 chain_action 的链（完整值，不截断）
    s("chain_id", getattr(c, "chain_id", "") or (ca.chain_id if ca and ca.chain_id else ""))
    s("parent_chain_id", getattr(c, "parent_chain_id", "") or "")
    s("trace_category", getattr(p, "trace_category", "") or "")
    s("session_uuid", c.meta.get("session_uuid", "") or "")
    # 请求
    i("stream_mode", 1 if getattr(c, "is_streaming", False) else 0)
    i("prompt_messages_count", p.prompt_messages_count or 0)
    i("prompt_tools_count", p.prompt_tools_count or 0)
    s("msg_hash_chain", getattr(p, "msg_hash_chain", "") or c.meta.get("msg_hash_chain", ""))
    # 锁/链
    s("lock_type", p.lock_type or "")
    s("sys_tools_locked", c.meta.get("sys_tools_locked", ""))
    s("prefix_chain_id", getattr(p, "prefix_chain_id", "") or "")
    # usage
    i("total_input_tokens", p.total_input_tokens or 0)
    i("output_tokens", p.output_tokens or 0)
    i("total_tokens", (p.total_input_tokens or 0) + (p.output_tokens or 0))
    i("cache_hit_tokens", p.cache_hit_tokens or 0)
    i("cache_miss_tokens", p.cache_miss_tokens or 0)
    i("cache_creation_tokens", p.cache_creation_tokens or 0)
    f("cache_hit_ratio", p.cache_hit_ratio or 0.0)
    i("has_reasoning", 1 if p.has_reasoning else 0)
    i("has_tool_calls", 1 if p.has_tool_calls else 0)
    # 修改留痕
    i("is_modified", 1 if p.is_modified else 0)
    s("modification_note", getattr(p, "modification_note", "") or "")
    i("system_role_converted", 1 if p.system_role_converted else 0)
    # 前缀命中
    i("pref_hit_count", c.meta.get("pref_hit_count", 0))
    i("pref_sys_hit", c.meta.get("pref_sys_hit", 0))
    i("pref_tools_hit", c.meta.get("pref_tools_hit", 0))
    i("pref_msgs_hit", c.meta.get("pref_msgs_hit", c.meta.get("pref_hit_count", 0)))
    # 缓存归因（客户端侧匹配原因 + 期望命中 vs 实际命中差距）
    attr = cache_attribution(c)
    s("cache_match_reason", attr["cache_match_reason"])
    i("cache_expected_hit_tokens", attr["cache_expected_hit_tokens"])
    i("cache_hit_gap", attr["cache_hit_gap"])
    # 链
    if ca and ca.action:
        s("chain_action", ca.action)

    return attrs


# ═══════════════════════════════════════════════════════════════════════
#  OtlpLogWriter — standard OTLP LogRecord JSONL with date rotation
# ═══════════════════════════════════════════════════════════════════════


class OtlpLogWriter:
    """Writes standard OTLP LogRecord JSONL to runs/{run_id}/{date}/otlp.jsonl.

    每行一个裸 LogRecord（不包装 ExportLogsServiceRequest）。
    按 UTC 日期轮转。
    """

    def __init__(self, run_dir: str):
        self._run_dir = Path(run_dir)
        self._lock = threading.Lock()
        self._current_date = ""
        self._file = None

    def write(self, ctx, run_id: str) -> None:
        """Write OTLP record synchronously. 线程安全（threading.Lock）。"""
        record = _build_otlp_record(ctx, run_id)
        with self._lock:
            today = dt.now(timezone.utc).strftime("%Y-%m-%d")
            if today != self._current_date:
                self._close_file()
                self._current_date = today
                day_dir = self._run_dir / today
                day_dir.mkdir(parents=True, exist_ok=True)
                self._file = open(day_dir / "otlp.jsonl", "a", encoding="utf-8")
            self._file.write(orjson.dumps(record).decode() + "\n")
            self._file.flush()

    def _close_file(self):
        if self._file is not None and not self._file.closed:
            self._file.flush()
            self._file.close()
            self._file = None

    def close(self):
        with self._lock:
            self._close_file()


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════


def _otlp_attr(key: str, value: Any) -> dict:
    """Build an OTLP-compatible attribute dict."""
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": value}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


# ═══════════════════════════════════════════════════════════════════════
#  Convenience function — emits to both terminal and OTLP
# ═══════════════════════════════════════════════════════════════════════


def emit_request_log(ctx: Context) -> None:
    """Emit per-request log: terminal (human) + OTLP JSONL (AI)."""
    view = LogView(ctx)

    # Terminal
    import logging
    log = logging.getLogger("deepseek-tap")
    log.info(view.terminal())

    # OTLP JSONL — 同步写入（OtlpLogWriter.write 使用 threading.Lock，不在 event loop 内阻塞）
    rt = ctx.request.get("runtime")
    proxy_ctx = {"otlp_log_writer": rt.otlp_writer, "run_id": rt.run_id} if rt else {}
    otlp_writer: OtlpLogWriter | None = proxy_ctx.get("otlp_log_writer")
    run_id = proxy_ctx.get("run_id", "")
    if otlp_writer is not None and run_id:
        otlp_writer.write(ctx, run_id)
