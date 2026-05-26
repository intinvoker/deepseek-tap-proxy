"""Built-in pipeline stages for the OpenAI proxy service.

Each stage is a self-contained processor that handles one aspect
of the proxy request lifecycle.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path

import aiohttp
from aiohttp import web

from deepseek_tap_proxy.pipeline import Context, Stage
from deepseek_tap_proxy.shared.decompress import decompress_body
from deepseek_tap_proxy.shared.headers import HOP_BY_HOP
from deepseek_tap_proxy.openai.db import (
    _resolve_session_fingerprint,
    get_previous_msg0,
    get_previous_msg0_by_uuid,
    get_previous_msg_chain_by_uuid,
    get_previous_msg_chain_by_fingerprint,
    get_previous_request_jsonl_path,
)
from deepseek_tap_proxy.openai.usage import normalize_usage
from deepseek_tap_proxy.openai.sse import OpenaiReconstructor

log = logging.getLogger("deepseek-tap")


def _t_db(db, func, *args):
    """Thread-safe wrapper: calls func(db.conn, *args) inside executor thread."""
    return asyncio.to_thread(lambda d=db, f=func, a=args: f(d.conn, *a))


# ── Session identifier patterns ──

# Copilot: VS Code debug log UUID embedded in msg[0]
_SESSION_UUID_RE = re.compile(
    r'GitHub\.copilot-chat[\\/]debug-logs[\\/]'
    r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})'
)

# Hermes: explicit Current Session ID in msg[0]
_HERMES_SESSION_ID_RE = re.compile(
    r'Current Session ID[:\s]+(\S+)', re.IGNORECASE
)


def _msg_hash(msg: dict) -> str:
    """Per-message hash: first 8 hex chars of md5(role + ':' + content)."""
    content = msg.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, sort_keys=True) if content else ""
    raw = f"{msg.get('role', '')}:{content}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def _compute_msg_hash_chain(messages: list) -> str:
    """Compute comma-separated hash chain for a list of messages."""
    return ",".join(_msg_hash(m) for m in messages)


def _read_messages_from_jsonl(jsonl_path: str, target_req_id: str, jsonl_offset: int = 0) -> list[dict] | None:
    """Read messages from a JSONL trace file, searching from end for efficiency."""
    path = Path(jsonl_path)
    if not path.exists():
        return None

    # Fast path: byte offset from DB
    if jsonl_offset > 0:
        with open(path, "rb") as f:
            f.seek(jsonl_offset)
            line = f.readline()
            try:
                rec = json.loads(line.decode("utf-8"))
                return rec.get("request", {}).get("body", {}).get("messages", [])
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass  # fall through to reverse scan

    file_size = path.stat().st_size
    chunk_size = 65536
    position = file_size

    with open(path, "rb") as f:
        while position > 0:
            read_size = min(chunk_size, position)
            position -= read_size
            f.seek(position)
            raw = f.read(read_size + (chunk_size if position > 0 else 0))
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue

            for line in reversed(text.split("\n")):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("request_id") == target_req_id:
                        return rec.get("request", {}).get("body", {}).get("messages", [])
                except json.JSONDecodeError:
                    continue

            if position == 0:
                break
    return None


def _compute_prefix_ratio(curr_hashes: list[str], prev_hashes: list[str]) -> float:
    """Compute what fraction of prev_hashes matches curr_hashes as prefix.

    Returns 0.0-1.0. Example:
      prev=[a,b,c,d], curr=[a,b,x,d,e] -> 2/4 = 0.5
    """
    if not prev_hashes or not curr_hashes:
        return 0.0
    match_count = 0
    for i in range(min(len(prev_hashes), len(curr_hashes))):
        if curr_hashes[i] == prev_hashes[i]:
            match_count += 1
        else:
            break
    return match_count / len(prev_hashes)


class ParseStage(Stage):
    """Parse the incoming request: read body, validate path, extract metadata."""

    async def process(self, ctx: Context) -> Context:
        body = await ctx.request.read()
        ctx.raw_body = body

        fwd_headers = dict(ctx.request.headers)
        fwd_headers.pop("Host", None)

        req_content_encoding = ctx.request.headers.get("Content-Encoding", "").lower()
        if req_content_encoding in ("zstd", "gzip", "deflate", "br"):
            try:
                body = decompress_body(body, req_content_encoding)
            except (ImportError, ValueError, OSError):
                log.warning(f"[{ctx.req_id}] unsupported Content-Encoding: {req_content_encoding}, forwarding as-is")
        fwd_headers["Accept-Encoding"] = "identity"
        ctx.fwd_headers = fwd_headers

        try:
            req_body = json.loads(body) if body else None
        except (json.JSONDecodeError, ValueError):
            req_body = None
        ctx.req_body = req_body

        if isinstance(req_body, dict):
            ctx.is_streaming = req_body.get("stream", False)

        model = req_body.get("model", "") if isinstance(req_body, dict) else ""

        normalized_path = "/" + ctx.request.path.lstrip("/").rstrip("/")
        allowed_suffixes = ("/chat/completions",)
        if not normalized_path.endswith(allowed_suffixes):
            log.warning(f"[{ctx.req_id}] blocked path: {ctx.request.path}")
            ctx.should_abort = True
            ctx.response = web.Response(status=404, text="Not Found")
            return ctx

        if isinstance(req_body, dict):
            msgs = req_body.get("messages", [])
            if msgs and isinstance(msgs, list):
                ctx.meta["prompt_messages_count"] = len(msgs)
                tools = req_body.get("tools", [])
                if isinstance(tools, list):
                    ctx.meta["prompt_tools_count"] = len(tools)

                fake_record = {"request": {"body": req_body}}
                fp, stype = _resolve_session_fingerprint(fake_record)
                ctx.meta["session_fingerprint"] = fp
                ctx.meta["session_type"] = stype

                if len(msgs) > 0:
                    c0 = msgs[0].get("content", "")
                    if isinstance(c0, str) and c0:
                        ctx.meta["msg0_hash"] = hashlib.md5(c0.encode("utf-8")).hexdigest()
                        ctx.meta["msg0_content"] = c0

                ctx.meta["msg_hash_chain"] = _compute_msg_hash_chain(msgs)

                if len(msgs) > 0:
                    c0 = msgs[0].get("content", "")
                    if isinstance(c0, str):
                        if stype == "copilot":
                            uuid_match = _SESSION_UUID_RE.search(c0)
                            if uuid_match:
                                ctx.meta["session_uuid"] = uuid_match.group(1)
                        elif stype == "hermes":
                            hermes_id = _HERMES_SESSION_ID_RE.search(c0)
                            if hermes_id:
                                ctx.meta["session_uuid"] = hermes_id.group(1)

        proxy_ctx = ctx.request.app.get("proxy_ctx", {})
        ctx.target_url = proxy_ctx.get("target_url", "")

        db = ctx.request.app.get("db")
        msg_hash_chain = ctx.meta.get("msg_hash_chain")
        if msg_hash_chain and db:
            session_uuid = ctx.meta.get("session_uuid")
            session_fp = ctx.meta.get("session_fingerprint")
            if session_uuid:
                prev = await _t_db(
                    db, get_previous_msg_chain_by_uuid, session_uuid, ctx.req_id
                )
            elif session_fp:
                prev = await _t_db(
                    db, get_previous_msg_chain_by_fingerprint, session_fp, ctx.req_id
                )
            else:
                prev = None
            if prev:
                prev_chain_str, prev_count = prev
                if prev_chain_str:
                    curr_hashes = msg_hash_chain.split(",")
                    prev_hashes = prev_chain_str.split(",")
                    ratio = _compute_prefix_ratio(curr_hashes, prev_hashes)
                    log.info(
                        f"[{ctx.req_id}] prefix match: {int(ratio * len(prev_hashes))}/"
                        f"{len(prev_hashes)} ({ratio:.1%}) "
                        f"prev={prev_count} curr={len(curr_hashes)} "
                        f"{session_uuid or session_fp or 'unknown'}"
                    )

        log.info(f"[{ctx.req_id}] -> POST {ctx.request.path} (model={model}, stream={ctx.is_streaming})")
        return ctx


class Msg0LockStage(Stage):
    """System prompt lock: sync msg[0] with previous request in session."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    async def process(self, ctx: Context) -> Context:
        if not self.enabled:
            return ctx
        if not ctx.req_body or not ctx.meta.get("session_fingerprint") or not ctx.meta.get("msg0_hash"):
            return ctx
        if not ctx.meta.get("msg0_content"):
            return ctx

        db = ctx.request.app.get("db")
        if not db:
            return ctx

        if ctx.meta.get("session_uuid"):
            prev = await _t_db(db, get_previous_msg0_by_uuid, ctx.meta.get("session_uuid"))
        else:
            prev = await _t_db(db, get_previous_msg0, ctx.meta.get("session_fingerprint"))
        if not prev:
            return ctx

        prev_hash, prev_content = prev
        if ctx.meta.get("msg0_hash") == prev_hash:
            return ctx

        msgs = ctx.req_body.get("messages", [])
        if not msgs:
            return ctx
        locked_msgs = list(msgs)
        locked_msgs[0] = {**locked_msgs[0], "content": prev_content}
        locked_body = {**ctx.req_body, "messages": locked_msgs}

        body_bytes = json.dumps(locked_body, separators=(",", ":")).encode("utf-8")
        ctx.raw_body = body_bytes
        if "Content-Length" in ctx.fwd_headers:
            ctx.fwd_headers["Content-Length"] = str(len(body_bytes))

        ctx.meta["modified"] = True
        ctx.meta["lock_type"] = "msg0"
        ctx.meta["modification_note"] = "msg[0] synced from previous request in session"
        log.info(f"[{ctx.req_id}] msg[0] locked -- synced from previous session request")
        return ctx


class PrefixLockStage(Stage):
    """Prefix lock: ensure messages prefix matches previous request in session."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    async def process(self, ctx: Context) -> Context:
        if not self.enabled:
            return ctx
        if not ctx.req_body or not ctx.meta.get("msg_hash_chain"):
            return ctx

        session_key = ctx.meta.get("session_uuid") or ctx.meta.get("session_fingerprint")
        if not session_key:
            return ctx

        msgs = ctx.req_body.get("messages", [])
        if not msgs:
            return ctx

        db = ctx.request.app.get("db")
        if not db:
            return ctx

        session_uuid = ctx.meta.get("session_uuid")
        session_fp = ctx.meta.get("session_fingerprint")
        if session_uuid:
            prev = await _t_db(
                db, get_previous_msg_chain_by_uuid, session_uuid, ctx.req_id
            )
        else:
            prev = await _t_db(
                db, get_previous_msg_chain_by_fingerprint, session_fp, ctx.req_id
            )

        if not prev:
            return ctx

        prev_chain_str, prev_count = prev
        if not prev_chain_str:
            return ctx

        prev_hashes = prev_chain_str.split(",")
        curr_hashes = ctx.meta.get("msg_hash_chain", "").split(",")

        if len(curr_hashes) <= len(prev_hashes):
            return ctx

        mismatch_idx = -1
        for i in range(len(prev_hashes)):
            if curr_hashes[i] != prev_hashes[i]:
                mismatch_idx = i
                break

        if mismatch_idx == -1:
            return ctx

        info = await _t_db(
            db, get_previous_request_jsonl_path, session_key,
            session_uuid is not None, ctx.req_id
        )
        if not info:
            log.warning(f"[{ctx.req_id}] prefix lock: cannot find previous request JSONL")
            return ctx

        prev_jsonl_path, prev_req_id, prev_jsonl_offset = info
        prev_msgs = _read_messages_from_jsonl(prev_jsonl_path, prev_req_id, prev_jsonl_offset)
        if not prev_msgs:
            log.warning(f"[{ctx.req_id}] prefix lock: cannot read previous messages from {prev_jsonl_path}")
            return ctx

        merged_msgs = list(prev_msgs) + list(msgs[len(prev_msgs):])
        merged_body = {**ctx.req_body, "messages": merged_msgs}
        body_bytes = json.dumps(merged_body, separators=(",", ":")).encode("utf-8")
        ctx.raw_body = body_bytes
        if "Content-Length" in ctx.fwd_headers:
            ctx.fwd_headers["Content-Length"] = str(len(body_bytes))

        ctx.meta["modified"] = True
        ctx.meta["lock_type"] = "prefix"
        ctx.meta["modification_note"] = (
            f"prefix override: {mismatch_idx}/{len(prev_hashes)} matched, "
            f"merged prev({len(prev_msgs)}) + curr tail({len(msgs) - len(prev_msgs)})"
        )
        log.info(
            f"[{ctx.req_id}] *** PREFIX OVERRIDE *** {mismatch_idx}/{len(prev_hashes)} "
            f"matched, merged prev({len(prev_msgs)}) + "
            f"curr tail({len(msgs) - len(prev_msgs)}) -> "
            f"total({len(merged_msgs)})"
        )
        return ctx


class ForwardStage(Stage):
    """Forward the request to the upstream API and capture the response."""

    async def process(self, ctx: Context) -> Context:
        proxy_ctx = ctx.request.app.get("proxy_ctx", {})
        session: aiohttp.ClientSession = proxy_ctx.get("session")
        target: str = proxy_ctx.get("target_url", "")
        timeout_sec: int = proxy_ctx.get("timeout", 600)

        if not session or not target:
            ctx.should_abort = True
            ctx.response = web.Response(status=502, text="Upstream not configured")
            return ctx

        upstream_url = target.rstrip("/") + "/chat/completions"

        try:
            upstream_resp = await session.request(
                method="POST",
                url=upstream_url,
                headers=ctx.fwd_headers,
                data=ctx.raw_body,
                timeout=aiohttp.ClientTimeout(total=timeout_sec, sock_read=timeout_sec),
            )
        except Exception as exc:
            log.error(f"[{ctx.req_id}] upstream error: {exc}")
            ctx.should_abort = True
            ctx.response = web.Response(status=502, text=str(exc))
            return ctx

        ctx.status = upstream_resp.status
        ctx.resp_headers = {k: v for k, v in upstream_resp.headers.items()
                            if k.lower() not in HOP_BY_HOP}

        if ctx.is_streaming and upstream_resp.status == 200:
            await self._handle_streaming(ctx, upstream_resp)
        else:
            await self._handle_non_streaming(ctx, upstream_resp)

        return ctx

    async def _handle_streaming(self, ctx: Context,
                                 upstream_resp: aiohttp.ClientResponse) -> None:
        resp = web.StreamResponse(
            status=upstream_resp.status,
            headers=ctx.resp_headers,
        )
        await resp.prepare(ctx.request)
        ctx.response = resp

        reassembler = OpenaiReconstructor()
        try:
            async for chunk in upstream_resp.content.iter_any():
                await resp.write(chunk)
                reassembler.feed_bytes(chunk)
        except (ConnectionError, asyncio.CancelledError):
            pass

        try:
            await resp.write_eof()
        except (ConnectionError, ConnectionResetError, Exception):
            pass

        ctx.duration_ms = ctx.elapsed_ms
        reconstructed = reassembler.reconstruct()
        ctx.resp_body = reconstructed

        if reconstructed:
            usage = normalize_usage(reconstructed.get("usage", {}))
            ca = compute_cache_analysis(ctx.req_body, reconstructed, reassembler.events)
            self._apply_cache_analysis(ctx, usage, ca)

    async def _handle_non_streaming(self, ctx: Context,
                                     upstream_resp: aiohttp.ClientResponse) -> None:
        resp_bytes = await upstream_resp.read()
        ctx.duration_ms = ctx.elapsed_ms
        ctx.resp_bytes = resp_bytes

        try:
            resp_body = json.loads(resp_bytes) if resp_bytes else None
        except (json.JSONDecodeError, ValueError):
            resp_body = None
        ctx.resp_body = resp_body

        if resp_body:
            usage = normalize_usage(resp_body.get("usage", {}))
            ca = compute_cache_analysis(ctx.req_body, resp_body)
            self._apply_cache_analysis(ctx, usage, ca)

        ctx.response = web.Response(
            status=upstream_resp.status,
            headers=ctx.resp_headers,
            body=resp_bytes,
        )

    @staticmethod
    def _apply_cache_analysis(ctx: Context, usage: dict,
                               cache_analysis: dict) -> None:
        ctx.meta["prompt_tokens"] = cache_analysis.get("prompt_total_tokens", 0)
        ctx.meta["completion_tokens"] = cache_analysis.get("output_tokens", 0)
        ctx.meta["cache_hit_tokens"] = cache_analysis.get("cache_hit_tokens", 0)
        ctx.meta["cache_miss_tokens"] = cache_analysis.get("cache_miss_tokens", 0)
        ctx.meta["cache_hit_ratio"] = cache_analysis.get("cache_hit_ratio", 0.0)
        ctx.meta["has_reasoning"] = cache_analysis.get("has_reasoning_content", False)
        ctx.meta["has_tool_calls"] = cache_analysis.get("has_tool_calls", False)


class ReturnStage(Stage):
    """Final stage: the response is already set by ForwardStage."""

    async def process(self, ctx: Context) -> Context:
        ctx.should_abort = True
        return ctx


def compute_cache_analysis(
    req_body: dict | None, resp_body: dict | None, sse_events: list[dict] | None = None,
) -> dict:
    """Compute cache analysis from request and response bodies."""
    cache_hit = 0
    cache_miss = 0
    prompt_tokens = 0
    output_tokens = 0
    has_reasoning = False
    has_tool_calls = False

    if resp_body and isinstance(resp_body, dict):
        usage = resp_body.get("usage", {})
        if not usage:
            usage = normalize_usage(resp_body)
        else:
            usage = normalize_usage(usage)

        prompt_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        cache_hit = usage.get("cache_read_input_tokens", 0)
        cache_miss = usage.get("cache_miss_input_tokens", 0)

        choices = resp_body.get("choices", [])
        if choices and isinstance(choices, list):
            msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            if isinstance(msg, dict):
                has_reasoning = bool(msg.get("reasoning_content"))
                has_tool_calls = bool(msg.get("tool_calls"))

    prompt_messages_count = 0
    prompt_tools_count = 0
    if req_body and isinstance(req_body, dict):
        messages = req_body.get("messages", [])
        if isinstance(messages, list):
            prompt_messages_count = len(messages)
        tools = req_body.get("tools", [])
        if isinstance(tools, list):
            prompt_tools_count = len(tools)

    cache_hit_ratio = (cache_hit / (cache_hit + cache_miss)) if (cache_hit + cache_miss) > 0 else 0.0

    return {
        "cache_hit_tokens": cache_hit,
        "cache_miss_tokens": cache_miss,
        "cache_hit_ratio": round(cache_hit_ratio, 4),
        "prompt_total_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "has_reasoning_content": has_reasoning,
        "has_tool_calls": has_tool_calls,
        "prompt_messages_count": prompt_messages_count,
        "prompt_tools_count": prompt_tools_count,
    }
