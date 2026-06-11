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
import json
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
        chain_action, chain_tokens_est, chain_char_count, new_char_count,
        persist (lock_type, chain_tokens, timestamp, api_format)
    """

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx

    # ── Terminal (human-readable colored line) ─────────────────────

    def terminal(self) -> str:
        """Format as colored ANSI line for terminal output.

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
        chain_tokens = c.meta.get("chain_tokens_est", 0) or r.chain_tokens
        chain_cc = c.meta.get("chain_char_count", 0)
        new_cc = c.meta.get("new_char_count", 0)
        if chain_tokens and chain_cc and chain_tokens > 0:
            avg_tpc = chain_tokens / chain_cc
            new_tokens_est = int(new_cc * avg_tpc)
            est_denom = chain_tokens + new_tokens_est
            est_pct = round(chain_tokens / est_denom * 100, 2) if est_denom else 0.0
            est_str = f"{chain_tokens}/{est_denom}({est_pct:.2f}%)"
        else:
            est_pct = 0.0
            est_str = "?"
        est_kv = f"est={colorize(est_str, 'cache_pct', est_pct)}"

        # ── sid (session id) ──
        sess_id = c.session_id or ""
        sid_kv = f"sid={sess_id}"

        # ── trailing bracket: metadata ──
        model = c.resp_model or "?"
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

        trailer = f"[{upstream}, {model}, {c.client_type}, {intent}]"

        return "  ".join([header] + kvs + [trailer])

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

        chain_tokens = c.meta.get("chain_tokens_est", 0) or r.chain_tokens
        chain_cc = c.meta.get("chain_char_count", 0)
        new_cc = c.meta.get("new_char_count", 0)
        if chain_tokens and chain_cc and chain_tokens > 0:
            avg_tpc = chain_tokens / chain_cc
            new_tokens_est = int(new_cc * avg_tpc)
            est_denom = chain_tokens + new_tokens_est
            est_pct = round(chain_tokens / est_denom * 100, 2) if est_denom else 0.0
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
        return json.dumps(record, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
#  TraceView — raw request/response archive
# ═══════════════════════════════════════════════════════════════════════


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
            "request": {
                "method": "POST",
                "path": c.request.path,
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
    """Writes full trace records to runs/{run_id}/{date}/traces.jsonl."""

    def __init__(self, run_dir: str, run_id: str):
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
        self.path = Path(run_dir) / "traces.jsonl"  # 向后兼容引用

    async def write(self, record: dict) -> int:
        today = dt.now(timezone.utc).strftime("%Y-%m-%d")
        async with self._lock:
            if today != self._current_date:
                self._close_file()
                self._current_date = today
                day_dir = self.run_dir / today
                day_dir.mkdir(parents=True, exist_ok=True)
                self._file = open(day_dir / "traces.jsonl", "a", encoding="utf-8")
                self.path = day_dir / "traces.jsonl"
            record["run_id"] = self.run_id
            offset = self._file.tell()
            self._file.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            self._file.flush()
            self.count += 1
            self._update_stats(record)
            return offset

    def _close_file(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.flush()
            self._file.close()
            self._file = None

    async def close(self) -> None:
        async with self._lock:
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

    Compatible with OpenTelemetry LogRecord protobuf JSON mapping and Aspire Dashboard.
    """
    status = ctx.status or 0
    severity_number, severity_text = _otlp_severity(status)
    now_ns = time.time_ns()
    p = ctx.persist

    attributes = []

    def _str_attr(key: str, value: str):
        attributes.append({"key": key, "value": {"stringValue": value}})

    def _int_attr(key: str, value: int):
        attributes.append({"key": key, "value": {"intValue": value}})

    _str_attr("run_id", run_id)
    _str_attr("req_id", ctx.req_id)
    _int_attr("status_code", status)
    _int_attr("duration_ms", ctx.duration_ms or 0)
    _str_attr("api_format", ctx.format or "")
    _str_attr("client_type", ctx.client_type or "")
    _str_attr("chat_type", ctx.chat_type or "")
    _str_attr("model", p.model or "")
    _str_attr("session_id", ctx.session_id or "")
    _str_attr("session_method", ctx.session_method or "")
    _str_attr("upstream_name", ctx.meta.get("upstream_name", "default"))
    _str_attr("lock_type", p.lock_type or "")
    _str_attr("sys_tools_locked", ctx.meta.get("sys_tools_locked", ""))
    _int_attr("cache_hit_tokens", p.cache_hit_tokens or 0)
    _int_attr("cache_miss_tokens", p.cache_miss_tokens or 0)
    _int_attr("pref_hit_count", ctx.meta.get("pref_hit_count", 0))

    chain_action = getattr(ctx, "chain_action", None)
    if chain_action and chain_action.action:
        _str_attr("chain_action", chain_action.action)
        if chain_action.chain_id:
            _str_attr("chain_id", chain_action.chain_id[:16])

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
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
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
    proxy_ctx = ctx.request.app.get("proxy_ctx", {})
    otlp_writer: OtlpLogWriter | None = proxy_ctx.get("otlp_log_writer")
    run_id = proxy_ctx.get("run_id", "")
    if otlp_writer is not None and run_id:
        otlp_writer.write(ctx, run_id)
