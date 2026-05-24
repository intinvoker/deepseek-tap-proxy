"""Anthropic API pipeline stages."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

import aiohttp
from aiohttp import web

from deepseek_tap_proxy.pipeline import Context, Stage
from deepseek_tap_proxy.shared.decompress import decompress_body
from deepseek_tap_proxy.shared.headers import filter_hop_by_hop
from deepseek_tap_proxy.anthropic.sse import AnthropicReconstructor
from deepseek_tap_proxy.anthropic.usage import normalize_anthropic_usage
from deepseek_tap_proxy.anthropic.preprocess import normalize_metadata

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

        if ctx.req_body:
            normalized = normalize_metadata(ctx.req_body)
            if normalized is not ctx.req_body:
                ctx.req_body = normalized
                ctx.raw_body = json.dumps(normalized, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                log.info(f"[{ctx.req_id}] metadata user_id normalized")

        fwd_headers = dict(ctx.request.headers)
        fwd_headers.pop("Host", None)
        if ctx.raw_body and ctx.req_body is not None:
            fwd_headers["Content-Length"] = str(len(ctx.raw_body))
        ctx.fwd_headers = fwd_headers

        proxy_ctx = ctx.request.app.get("proxy_ctx", {})
        ctx.target_url = proxy_ctx.get("target_url", "")
        ctx.timeout = proxy_ctx.get("timeout", 600)

        model = (ctx.req_body or {}).get("model", "?")
        log.info(
            f"[{ctx.req_id}] → {ctx.request.method} {ctx.request.path} "
            f"(model={model}, stream={ctx.is_streaming})"
        )
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
                upstream_resp = await fresh_session.request(
                    method=ctx.request.method, url=upstream_url,
                    headers=ctx.fwd_headers, data=ctx.raw_body or None,
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
