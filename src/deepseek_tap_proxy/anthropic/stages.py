"""Anthropic API pipeline stages."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re
import uuid

import aiohttp
from aiohttp import web

from deepseek_tap_proxy.pipeline import Context, Stage
from deepseek_tap_proxy.shared.decompress import decompress_body
from deepseek_tap_proxy.shared.headers import filter_hop_by_hop
from deepseek_tap_proxy.anthropic.sse import AnthropicReconstructor
from deepseek_tap_proxy.anthropic.usage import normalize_anthropic_usage
from deepseek_tap_proxy.anthropic.db import (
    find_baseline_by_hashes, get_baseline, create_baseline, insert_audit,
)

log = logging.getLogger("deepseek-tap")

_ANTHROPIC_USER_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

_ALLOWED_PATH_PREFIXES = ("/v1/messages", "/v1/complete", "/v1/models")


def _is_allowed_path(path: str) -> bool:
    clean = path.split("?", 1)[0].rstrip("/")
    return any(
        clean == p or clean.startswith(p + "/") or clean.startswith(p + ":")
        for p in _ALLOWED_PATH_PREFIXES
    )


class AnthropicParseStage(Stage):
    async def process(self, ctx: Context) -> Context:
        path_suffix = ctx.request.path
        if path_suffix.startswith("/anthropic"):
            path_suffix = path_suffix[len("/anthropic"):]
        if not _is_allowed_path(path_suffix):
            log.warning(f"[{ctx.req_id}] blocked path: {ctx.request.path}")
            ctx.should_abort = True
            ctx.response = web.Response(status=404, text="Not Found")
            return ctx

        raw_body = await ctx.request.read() if ctx.request.method in ("POST", "PUT", "PATCH") else b""
        ctx.raw_body = raw_body

        encoding = ctx.request.headers.get("Content-Encoding", "").lower()
        if encoding in ("zstd", "gzip", "deflate", "br"):
            try:
                raw_body = decompress_body(raw_body, encoding)
            except (ImportError, ValueError, OSError):
                log.warning(f"[{ctx.req_id}] unsupported Content-Encoding: {encoding}")

        if raw_body:
            try:
                parsed = json.loads(raw_body)
                ctx.req_body = parsed
                ctx.is_streaming = parsed.get("stream", False)
            except (json.JSONDecodeError, ValueError):
                pass

        fwd_headers = dict(ctx.request.headers)
        fwd_headers.pop("Host", None)
        if ctx.raw_body and ctx.req_body is not None:
            fwd_headers["Content-Length"] = str(len(ctx.raw_body))
        ctx.fwd_headers = fwd_headers

        proxy_ctx = ctx.request.app.get("proxy_ctx", {})
        ctx.target_url = proxy_ctx.get("target_url", "")
        ctx.timeout = proxy_ctx.get("timeout", 600)

        # Extract Claude-Code session ID for persist and baseline
        sid = ctx.request.headers.get("X-Claude-Code-Session-Id", "") or \
              ctx.request.headers.get("x-claude-code-session-id", "")
        ctx.meta["anthropic_sid"] = sid

        # Upstream base URL for persist
        ctx.meta["upstream_base_url"] = proxy_ctx.get("target_url", "")

        model = (ctx.req_body or {}).get("model", "?")
        log.info(
            f"[{ctx.req_id}] → {ctx.request.method} {ctx.request.path} "
            f"(model={model}, stream={ctx.is_streaming})"
        )
        return ctx


class AnthropicNormalizeStage(Stage):
    """Hash non-standard user_id for DeepSeek compatibility.

    Reads original metadata from req_body (immutable), writes normalized
    user_id to fwd_body (mutable). Archives original metadata in meta.
    """

    async def process(self, ctx: Context) -> Context:
        if not isinstance(ctx.req_body, dict):
            return ctx
        metadata = ctx.req_body.get("metadata")
        if not isinstance(metadata, dict):
            return ctx
        user_id = metadata.get("user_id")
        if not isinstance(user_id, str) or _ANTHROPIC_USER_ID_RE.fullmatch(user_id):
            return ctx

        digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]
        ctx.meta["original_metadata"] = dict(metadata)
        if ctx.fwd_body is None:
            ctx.fwd_body = copy.deepcopy(ctx.req_body)
        new_meta = dict(ctx.fwd_body.get("metadata", {}))
        new_meta["user_id"] = f"tap_{digest}"
        ctx.fwd_body["metadata"] = new_meta
        return ctx


class AnthropicBlock0ClearStage(Stage):
    """Clear first system block text for cache stability.

    Reads original system from req_body, empties block[0].text in fwd_body.
    """

    async def process(self, ctx: Context) -> Context:
        if not isinstance(ctx.req_body, dict):
            return ctx
        system = ctx.req_body.get("system")
        if not isinstance(system, list) or not system:
            return ctx
        block0 = system[0]
        if not isinstance(block0, dict) or not isinstance(block0.get("text"), str):
            return ctx
        if not block0["text"]:
            return ctx

        if ctx.fwd_body is None:
            ctx.fwd_body = copy.deepcopy(ctx.req_body)
        fwd_system = list(ctx.fwd_body.get("system", system))
        fwd_system[0] = dict(fwd_system[0])
        fwd_system[0]["text"] = ""
        ctx.fwd_body["system"] = fwd_system
        ctx.meta["modified"] = True
        return ctx


class AnthropicCanonicalUserStage(Stage):
    """Look up existing sys+tools hash baseline to reuse user_id for cross-session APC.

    Reads system/tools from req_body (immutable), computes hashes, checks
    DB for existing baseline with same hashes. If found and has a user_id,
    overrides fwd_body metadata with the canonical user_id.
    """

    async def process(self, ctx: Context) -> Context:
        if not isinstance(ctx.req_body, dict):
            return ctx
        sid = ctx.meta.get("anthropic_sid", "")
        if not sid:
            return ctx

        system = ctx.req_body.get("system")
        tools = ctx.req_body.get("tools", [])
        has_sys = isinstance(system, list) and len(system) >= 1
        has_tools = isinstance(tools, list) and len(tools) > 0
        if not has_sys or not has_tools:
            return ctx

        sys_hash = hashlib.sha256(
            json.dumps(system, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        tools_hash = hashlib.sha256(
            json.dumps(tools, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        ctx.meta["sys_hash"] = sys_hash
        ctx.meta["tools_hash"] = tools_hash

        conn = ctx.request.app.get("db_conn")
        if conn is None:
            return ctx

        # Determine current user_id (from original_metadata or req_body)
        raw_meta = ctx.meta.get("original_metadata") or ctx.req_body.get("metadata") or {}
        raw_uid = raw_meta.get("user_id", "") or ""
        if raw_uid:
            if _ANTHROPIC_USER_ID_RE.fullmatch(raw_uid):
                current_uid = raw_uid
            else:
                digest = hashlib.sha256(raw_uid.encode("utf-8")).hexdigest()[:24]
                current_uid = f"tap_{digest}"
        else:
            current_uid = ""

        # Look for existing baseline with same sys+tools hash
        existing = find_baseline_by_hashes(conn, sys_hash, tools_hash)

        if existing:
            canonical = existing["user_id"]
            if canonical:
                if canonical != current_uid:
                    if ctx.fwd_body is None:
                        ctx.fwd_body = copy.deepcopy(ctx.req_body)
                    new_meta = dict(ctx.fwd_body.get("metadata", {}))
                    new_meta["user_id"] = canonical
                    ctx.fwd_body["metadata"] = new_meta
                    ctx.meta["user_id"] = canonical
                    ctx.meta["user_id_was_overridden"] = True
                else:
                    ctx.meta["user_id"] = canonical or current_uid
            else:
                ctx.meta["user_id"] = current_uid
        else:
            ctx.meta["user_id"] = current_uid

        return ctx


class AnthropicBaselineStage(Stage):
    """Sys/tools baseline: capture on first request, compare+replace on subsequent.

    On first request in session: saves system+tools JSON + hashes to DB.
    On subsequent requests: compares current hashes vs baseline.
    If sys or tools changed, replaces fwd_body with baseline values.
    """

    async def process(self, ctx: Context) -> Context:
        if not isinstance(ctx.req_body, dict):
            return ctx
        sid = ctx.meta.get("anthropic_sid", "")
        sys_hash = ctx.meta.get("sys_hash", "")
        if not sid or not sys_hash:
            return ctx

        conn = ctx.request.app.get("db_conn")
        if conn is None:
            return ctx

        system = ctx.req_body.get("system")
        tools = ctx.req_body.get("tools", [])
        tools_hash = ctx.meta.get("tools_hash", "")
        baseline = get_baseline(conn, sid)

        if baseline is None:
            # First request: create baseline with canonical user_id
            sys_ser = json.dumps(system, separators=(",", ":"), ensure_ascii=False)
            tools_ser = json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
            canonical_uid = ctx.meta.get("user_id", "")
            create_baseline(conn,
                            session_id=sid, system_json=sys_ser, tools_json=tools_ser,
                            request_id=ctx.req_id, system_hash=sys_hash,
                            tools_hash=tools_hash, user_id=canonical_uid)
            ctx.meta["is_baseline_set"] = True
            insert_audit(conn,
                         request_id=ctx.req_id, session_id=sid,
                         system_hash=sys_hash, tools_hash=tools_hash,
                         is_baseline=True, sys_changed=False, tools_changed=False,
                         was_replaced=False,
                         user_id_was_overridden=ctx.meta.get("user_id_was_overridden", False),
                         metadata=ctx.meta.get("original_metadata"))
        else:
            # Subsequent request: compare hashes
            ctx.meta["is_baseline_set"] = False
            sys_changed = sys_hash != baseline["system_hash"]
            tools_changed = tools_hash != baseline["tools_hash"]
            replaced = sys_changed or tools_changed

            if replaced:
                if ctx.fwd_body is None:
                    ctx.fwd_body = copy.deepcopy(ctx.req_body)
                if sys_changed:
                    ctx.fwd_body["system"] = json.loads(baseline["system_json"])
                if tools_changed:
                    ctx.fwd_body["tools"] = json.loads(baseline["tools_json"])
                ctx.meta["modified"] = True

            insert_audit(conn,
                         request_id=ctx.req_id, session_id=sid,
                         system_hash=sys_hash, tools_hash=tools_hash,
                         is_baseline=False,
                         sys_changed=sys_changed, tools_changed=tools_changed,
                         was_replaced=replaced,
                         user_id_was_overridden=ctx.meta.get("user_id_was_overridden", False),
                         metadata=ctx.meta.get("original_metadata"))

        return ctx


class AnthropicForwardStage(Stage):
    async def process(self, ctx: Context) -> Context:
        proxy_ctx = ctx.request.app.get("proxy_ctx", {})
        target = proxy_ctx.get("target_url", "")
        timeout = proxy_ctx.get("timeout", 600)

        if not target:
            ctx.should_abort = True
            ctx.response = web.Response(status=502, text="Upstream not configured")
            return ctx

        req_path = ctx.request.path_qs
        if req_path.startswith("/anthropic"):
            req_path = req_path[len("/anthropic"):]
        upstream_url = target.rstrip("/") + "/anthropic" + req_path

        max_attempts = 2
        upstream_resp = None
        fresh_session = None
        last_exc = None

        for attempt in range(1, max_attempts + 1):
            try:
                fresh_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=timeout, sock_read=timeout),
                )
                # fwd_body carries stage modifications (normalized user_id, block0 clear, etc.)
                if ctx.fwd_body is not None:
                    body_bytes = json.dumps(
                        ctx.fwd_body, ensure_ascii=False
                    ).encode("utf-8")
                else:
                    body_bytes = ctx.raw_body or None
                if body_bytes is not None:
                    ctx.fwd_headers["Content-Length"] = str(len(body_bytes))
                upstream_resp = await fresh_session.request(
                    method=ctx.request.method, url=upstream_url,
                    headers=ctx.fwd_headers, data=body_bytes,
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if fresh_session:
                    await fresh_session.close()
                    fresh_session = None
                if attempt < max_attempts:
                    await asyncio.sleep(0.5)

        if last_exc is not None:
            log.error(f"[{ctx.req_id}] ← 502 upstream error after {max_attempts} attempts: {last_exc}")
            ctx.should_abort = True
            ctx.response = web.Response(status=502, text=str(last_exc))
            return ctx

        ctx.status = upstream_resp.status
        ctx.resp_headers = filter_hop_by_hop(dict(upstream_resp.headers))

        if ctx.is_streaming and upstream_resp.status == 200:
            await self._relay_stream(ctx, upstream_resp)
        else:
            await self._relay_non_stream(ctx, upstream_resp)

        if fresh_session:
            await fresh_session.close()

        ctx.duration_ms = ctx.elapsed_ms
        usage = ctx.meta.get("usage", {})
        log.info(
            f"[{ctx.req_id}] ← {ctx.status} {ctx.duration_ms}ms"
            + (f" in={usage.get('input_tokens', 0)}"
               f" out={usage.get('output_tokens', 0)}"
               f" cache_r={usage.get('cache_read_input_tokens', 0)}"
               if usage else "")
        )
        return ctx

    async def _relay_stream(self, ctx: Context, upstream_resp) -> None:
        resp = web.StreamResponse(status=upstream_resp.status, headers=ctx.resp_headers)
        await resp.prepare(ctx.request)
        ctx.response = resp
        reconstructor = AnthropicReconstructor()
        try:
            async for chunk in upstream_resp.content.iter_any():
                await resp.write(chunk)
                reconstructor.feed_bytes(chunk)
        except (ConnectionError, asyncio.CancelledError):
            pass
        try:
            await resp.write_eof()
        except Exception:
            pass
        ctx.resp_body = reconstructor.reconstruct()
        ctx.meta["sse_events"] = reconstructor.events
        if ctx.resp_body:
            ctx.meta["usage"] = normalize_anthropic_usage(ctx.resp_body.get("usage", {}))

    async def _relay_non_stream(self, ctx: Context, upstream_resp) -> None:
        resp_bytes = await upstream_resp.read()
        ctx.resp_bytes = resp_bytes
        decode_bytes = resp_bytes
        encoding = upstream_resp.headers.get("Content-Encoding", "").lower()
        if resp_bytes and encoding in ("gzip", "deflate"):
            try:
                if encoding == "gzip":
                    import gzip
                    decode_bytes = gzip.decompress(resp_bytes)
                else:
                    import zlib
                    decode_bytes = zlib.decompress(resp_bytes)
            except Exception:
                pass
        try:
            ctx.resp_body = json.loads(decode_bytes) if decode_bytes else None
        except (json.JSONDecodeError, ValueError):
            ctx.resp_body = None
        if ctx.resp_body:
            ctx.meta["usage"] = normalize_anthropic_usage(ctx.resp_body.get("usage", {}))
        ctx.response = web.Response(
            status=upstream_resp.status, headers=ctx.resp_headers, body=resp_bytes,
        )


class ReturnStage(Stage):
    async def process(self, ctx: Context) -> Context:
        ctx.should_abort = True
        return ctx
