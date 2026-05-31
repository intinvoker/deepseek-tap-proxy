"""Anthropic API pipeline stages."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re

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
        fwd_headers = filter_hop_by_hop(fwd_headers)
        if ctx.raw_body and ctx.req_body is not None:
            fwd_headers["Content-Length"] = str(len(ctx.raw_body))
        ctx.fwd_headers = fwd_headers

        proxy_ctx = ctx.request.app.get("proxy_ctx", {})
        if not ctx.target_url:
            ctx.target_url = proxy_ctx.get("target_url", "")
        ctx.timeout = proxy_ctx.get("timeout", 600)

        # Extract Claude-Code session ID for persist and baseline
        sid = ctx.request.headers.get("X-Claude-Code-Session-Id", "") or \
              ctx.request.headers.get("x-claude-code-session-id", "")
        ctx.meta["anthropic_sid"] = sid

        # Upstream base URL for persist (only set if handler didn't already)
        if "upstream_base_url" not in ctx.meta:
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

    Uses fwd_body.system (post-Block0Clear, billing header already cleared)
    for hash computation, so sys_hash is consistent across sessions with the
    same core system prompt, enabling cross-session APC cache hits.
    """

    async def process(self, ctx: Context) -> Context:
        if not isinstance(ctx.req_body, dict):
            return ctx
        sid = ctx.meta.get("anthropic_sid", "")
        if not sid:
            return ctx

        # Use fwd_body.system (post-Block0Clear) for consistent cross-session hash
        system = ctx.fwd_body.get("system") if ctx.fwd_body is not None else ctx.req_body.get("system")
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

        # Use fwd_body.system if available (post-Block0Clear, system[0].text already emptied)
        # to avoid storing and later restoring the billing-header that breaks APC prefix.
        system = ctx.fwd_body.get("system") if ctx.fwd_body is not None else ctx.req_body.get("system")
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


class AnthropicSystemRoleConvertStage(Stage):
    """Convert role=system messages to role=user for upstream compatibility.

    Claude Code MCP mode may inject MCP server instructions as a message with
    role='system' in the messages array. Some upstreams (e.g. Xiaomi Mimo)
    reject role='system' in messages. This stage converts them to role='user'
    and sets a flag for DB tracking.
    """

    async def process(self, ctx: Context) -> Context:
        if not isinstance(ctx.req_body, dict):
            return ctx
        messages = ctx.req_body.get("messages", [])
        if not isinstance(messages, list):
            return ctx

        converted = False
        has_system_role = any(
            isinstance(m, dict) and m.get("role") == "system"
            for m in messages
        )
        if not has_system_role:
            return ctx

        # Build fwd_body if not already modified
        body = ctx.fwd_body if ctx.fwd_body is not None else ctx.req_body
        new_messages = []
        for m in body.get("messages", []):
            if isinstance(m, dict) and m.get("role") == "system":
                m = dict(m)
                m["role"] = "user"
                converted = True
            new_messages.append(m)

        if converted:
            fwd = copy.deepcopy(ctx.req_body)
            fwd["messages"] = new_messages
            ctx.fwd_body = fwd
            ctx.meta["system_role_converted"] = True
            log.info(
                f"[{ctx.req_id}] converted role=system → role=user "
                f"({sum(1 for m in new_messages if m.get('role')=='user')} user messages)"
            )

        return ctx


class AnthropicUpstreamPathStage(Stage):
    """Compute upstream URL based on upstream's path_prefix.

    If the upstream has a `path_prefix` configured and the request is for
    the Messages API (/v1/messages), the upstream URL is built as:

        target + path_prefix

    Otherwise falls back to the default behavior:

        target + /anthropic + path

    The result is stored in ctx.upstream_url for ForwardStage to consume.
    """

    _MESSAGES_PATH = "/v1/messages"

    async def process(self, ctx: Context) -> Context:
        upstream = ctx.request.get("upstream")
        target = ctx.target_url
        if not target:
            ctx.should_abort = True
            ctx.response = web.Response(status=502, text="Upstream not configured")
            return ctx

        req_path = ctx.request.path_qs
        if req_path.startswith("/anthropic"):
            req_path = req_path[len("/anthropic"):]
        # Strip query string for path matching
        clean_path = req_path.split("?", 1)[0].rstrip("/")

        path_prefix = upstream.path_prefix if upstream else None

        if path_prefix and clean_path == self._MESSAGES_PATH:
            ctx.upstream_url = target.rstrip("/") + path_prefix
            log.debug(
                f"[{ctx.req_id}] upstream path override: "
                f"{target} + {path_prefix} → {ctx.upstream_url}"
            )
        else:
            ctx.upstream_url = target.rstrip("/") + "/anthropic" + req_path

        return ctx


class AnthropicForwardStage(Stage):
    async def process(self, ctx: Context) -> Context:
        proxy_ctx = ctx.request.app.get("proxy_ctx", {})
        session: aiohttp.ClientSession = proxy_ctx.get("session")
        timeout = proxy_ctx.get("timeout", 600)

        target = ctx.target_url
        if not target:
            ctx.should_abort = True
            ctx.response = web.Response(status=502, text="Upstream not configured")
            return ctx

        upstream_url = ctx.upstream_url
        if not upstream_url:
            # Fallback: build URL the old way (shouldn't happen with UpstreamPathStage in place)
            req_path = ctx.request.path_qs
            if req_path.startswith("/anthropic"):
                req_path = req_path[len("/anthropic"):]
            upstream_url = target.rstrip("/") + "/anthropic" + req_path

        upstream = ctx.request.get("upstream")
        if upstream is not None:
            ctx.meta["upstream_name"] = upstream.upstream_name

        # fwd_body carries stage modifications (normalized user_id, block0 clear, etc.)
        if ctx.fwd_body is not None:
            body_bytes = json.dumps(
                ctx.fwd_body, ensure_ascii=False
            ).encode("utf-8")
        else:
            body_bytes = ctx.raw_body or None
        if body_bytes is not None:
            ctx.fwd_headers["Content-Length"] = str(len(body_bytes))

        try:
            upstream_resp = await session.request(
                method=ctx.request.method, url=upstream_url,
                headers=ctx.fwd_headers, data=body_bytes,
                timeout=aiohttp.ClientTimeout(total=timeout, sock_read=timeout),
            )
        except Exception as exc:
            log.error(f"[{ctx.req_id}] upstream error: {exc}")
            ctx.should_abort = True
            ctx.response = web.Response(status=502, text=str(exc))
            return ctx

        ctx.status = upstream_resp.status
        ctx.resp_headers = filter_hop_by_hop(dict(upstream_resp.headers))

        if ctx.is_streaming and upstream_resp.status == 200:
            await self._relay_stream(ctx, upstream_resp)
        else:
            await self._relay_non_stream(ctx, upstream_resp)

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
