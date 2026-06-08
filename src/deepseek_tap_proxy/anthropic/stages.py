"""Anthropic API pipeline stages."""

from __future__ import annotations

import asyncio
import json
import logging

from datetime import datetime, timezone

import aiohttp
from aiohttp import web

from deepseek_tap_proxy.pipeline import Context, Stage
from deepseek_tap_proxy.shared.decompress import decompress_body
from deepseek_tap_proxy.shared.headers import filter_hop_by_hop
from deepseek_tap_proxy.anthropic.sse import AnthropicReconstructor
from deepseek_tap_proxy.anthropic.usage import normalize_anthropic_usage

log = logging.getLogger("deepseek-tap")

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

        encoding = ctx.request.headers.get("Content-Encoding", "").lower()
        if encoding in ("zstd", "gzip", "deflate", "br"):
            try:
                raw_body = decompress_body(raw_body, encoding)
            except (ImportError, ValueError, OSError):
                log.warning(f"[{ctx.req_id}] unsupported Content-Encoding: {encoding}")

        if raw_body:
            try:
                parsed = json.loads(raw_body)
                ctx.body_readonly = parsed
                if isinstance(parsed, dict):
                    ctx.fwd_body = dict(parsed)  # 浅拷贝比 deepcopy 快 N 倍
                else:
                    ctx.fwd_body = parsed
                # ── 规范形式 view 就绪 ──
                ctx.format = "anthropic"
                ctx.is_streaming = parsed.get("stream", False) if isinstance(parsed, dict) else False
            except (json.JSONDecodeError, ValueError):
                pass

        fwd_headers = dict(ctx.request.headers)
        fwd_headers.pop("Host", None)
        fwd_headers = filter_hop_by_hop(fwd_headers)
        ctx.fwd_headers = fwd_headers

        proxy_ctx = ctx.request.app.get("proxy_ctx", {})
        if not ctx.target_url:
            ctx.target_url = proxy_ctx.get("target_url", "")
        ctx.timeout = proxy_ctx.get("timeout", 600)

        # Extract Claude-Code session ID for persist
        sid = ctx.request.headers.get("X-Claude-Code-Session-Id", "") or \
              ctx.request.headers.get("x-claude-code-session-id", "")
        ctx.meta["anthropic_sid"] = sid

        # Session detection via shared/analysis.py
        from deepseek_tap_proxy.shared.analysis import resolve_session_anthropic
        system = (ctx.body_readonly or {}).get("system", [])
        db = ctx.request.app.get("db")
        session = resolve_session_anthropic(system, dict(ctx.request.headers), db=db)
        ctx.meta["session"] = session

        # Upstream base URL for persist (only set if handler didn't already)
        if "upstream_base_url" not in ctx.meta:
            ctx.meta["upstream_base_url"] = proxy_ctx.get("target_url", "")

        # strip metadata.user_id if configured
        config = ctx.request.app.get("config", {})
        if getattr(config, "strip_user_id", True):
            meta = ctx.fwd_body.get("metadata") if isinstance(ctx.fwd_body, dict) else None
            if isinstance(meta, dict):
                meta.pop("user_id", None)

        # ── persist fields ──
        from deepseek_tap_proxy.anthropic.persist import _detect_title_gen
        ctx.persist.trace_category = _detect_title_gen(ctx.body_readonly) or ""
        ctx.persist.api_format = "anthropic"
        ctx.persist.model = (ctx.body_readonly or {}).get("model", "")
        ctx.persist.stream_mode = ctx.is_streaming
        ctx.persist.request_id = ctx.req_id
        ctx.persist.timestamp = datetime.now(timezone.utc).isoformat()

        return ctx



class AnthropicBlock0ClearStage(Stage):
    """Clear first system block text for cache stability.

    Reads original system from body_readonly, empties block[0].text in fwd_body.
    """

    async def process(self, ctx: Context) -> Context:
        if not isinstance(ctx.body_readonly, dict):
            return ctx
        system = ctx.body_readonly.get("system")
        if not isinstance(system, list) or not system:
            return ctx
        block0 = system[0]
        if not isinstance(block0, dict) or not isinstance(block0.get("text"), str):
            return ctx
        if not block0["text"]:
            return ctx

        if ctx.fwd_body is None and ctx.body_readonly is not None:
            ctx.fwd_body = dict(ctx.body_readonly)  # 浅拷贝
        fwd_system = list(ctx.fwd_body.get("system", system))
        fwd_system[0] = dict(fwd_system[0])
        fwd_system[0]["text"] = ""
        ctx.fwd_body["system"] = fwd_system
        ctx.persist.is_modified = True
        return ctx




class AnthropicSystemRoleConvertStage(Stage):
    """Convert role=system messages to role=user for upstream compatibility.

    Operates on ctx.messages view — no direct fwd_body access.
    """

    async def process(self, ctx: Context) -> Context:
        if not ctx.messages:
            return ctx

        converted = False
        new_msgs = []
        for m in ctx.messages:  # 通过 view 读取
            if m.get("role") == "system":
                m = dict(m)
                m["role"] = "user"
                converted = True
            new_msgs.append(m)

        if converted:
            ctx.messages = new_msgs  # 通过 view 写回
            ctx.persist.system_role_converted = True
            ctx.persist.is_modified = True
            log.info(
                f"[{ctx.req_id}] converted role=system → role=user "
                f"({sum(1 for m in new_msgs if m.get('role')=='user')} user messages)"
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
            body_bytes = None
        if body_bytes is not None:
            ctx.fwd_headers["Content-Length"] = str(len(body_bytes))
        else:
            # 防止残留原始 Content-Length 导致上游空等 body
            ctx.fwd_headers.pop("Content-Length", None)

        try:
            upstream_resp = await session.request(
                method=ctx.request.method, url=upstream_url,
                headers=ctx.fwd_headers, data=body_bytes,
                timeout=aiohttp.ClientTimeout(total=timeout, sock_read=timeout),
            )
        except Exception as exc:
            ctx.status = 502
            ctx.duration_ms = ctx.elapsed_ms
            from deepseek_tap_proxy.shared.stages import emit_request_log
            emit_request_log(ctx)
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
        from deepseek_tap_proxy.shared.stages import apply_cache_analysis, emit_request_log
        apply_cache_analysis(ctx)
        emit_request_log(ctx)
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
