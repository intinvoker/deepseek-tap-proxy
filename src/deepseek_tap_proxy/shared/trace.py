from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

class TraceWriter:
    def __init__(self, trace_dir: str, session_id: str):
        self.session_id = session_id
        self.trace_dir = Path(trace_dir)
        self._lock = asyncio.Lock()
        self.count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_create_tokens = 0
        self.total_cache_miss_tokens = 0
        self.models_used: dict[str, int] = {}

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._session_dir = self.trace_dir / today
        self._session_dir.mkdir(parents=True, exist_ok=True)

        now_str = datetime.now(timezone.utc).strftime("%H%M%S")
        filename = f"trace_{now_str}.jsonl"
        self.path = self._session_dir / filename
        self._file = open(self.path, "a", encoding="utf-8")

    async def write(self, record: dict) -> int:
        async with self._lock:
            offset = self._file.tell()
            self._file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            self._file.flush()
            self.count += 1
            self._update_stats(record)
            return offset

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

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
            "session_id": self.session_id,
            "trace_file": str(self.path),
        }


def build_trace_record(ctx) -> dict:
    """Build a unified trace record for JSONL output.

    Raw request/response from Context directly (not from persist).
    Usage/meta from ctx.persist (same source as DB).
    """
    from deepseek_tap_proxy.shared.headers import redact
    from datetime import datetime, timezone

    r = ctx.persist
    return {
        "request_id": r.request_id or ctx.req_id,
        "timestamp": r.timestamp or datetime.now(timezone.utc).isoformat(),
        "duration_ms": ctx.duration_ms or ctx.elapsed_ms or 0,
        "upstream_base_url": ctx.target_url,
        "upstream_name": ctx.meta.get("upstream_name", "default"),
        "api_format": r.api_format,
        "request": {
            "method": "POST",
            "path": ctx.request.path,
            "headers": redact(dict(ctx.request.headers)),
            "body": ctx.body_readonly,
        },
        "response": {
            "status": ctx.status,
            "headers": redact(ctx.resp_headers),
            "body": ctx.resp_body,
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
