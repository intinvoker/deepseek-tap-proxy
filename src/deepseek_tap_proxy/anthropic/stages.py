"""Anthropic API pipeline stages."""

from __future__ import annotations

import asyncio
import collections.abc
import json
import logging
import orjson

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
                parsed = orjson.loads(raw_body)
                ctx.body_readonly = parsed
                if isinstance(parsed, dict):
                    ctx.fwd_body = dict(parsed)  # 浅拷贝比 deepcopy 快 N 倍
                else:
                    ctx.fwd_body = parsed
                # ── 规范形式 view 就绪 ──
                ctx.format = "anthropic"
                ctx.is_streaming = parsed.get("stream", False) if isinstance(parsed, dict) else False
            except (orjson.JSONDecodeError, json.JSONDecodeError, ValueError):
                ctx.should_abort = True
                ctx.response = web.Response(status=400, text="Invalid JSON")
                return ctx

        fwd_headers = dict(ctx.request.headers)
        fwd_headers.pop("Host", None)
        fwd_headers = filter_hop_by_hop(fwd_headers)
        ctx.fwd_headers = fwd_headers

        rt = ctx.request.get("runtime")
        if rt is not None:
            if not ctx.target_url and rt.config:
                ctx.target_url = getattr(rt.config, "target", "")
            ctx.timeout = getattr(rt.config, "timeout", 600) if rt.config else 600

        # Session detection is now handled by RequestAnalysisStage
        # after CanonicalStage.  AnthropicParseStage only parses the body.

        # Upstream base URL for persist (only set if handler didn't already)
        if "upstream_base_url" not in ctx.meta:
            ctx.meta["upstream_base_url"] = getattr(rt.config, "target", "") if rt else ""

        # strip metadata.user_id if configured using Runtime config
        rt = ctx.request.get("runtime")
        strip_user_id = getattr(rt.config, "strip_user_id", True) if rt is not None and rt.config else True
        if strip_user_id:
            meta = ctx.fwd_body.get("metadata") if isinstance(ctx.fwd_body, dict) else None
            if isinstance(meta, dict):
                meta.pop("user_id", None)

        # ── persist fields ──
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

        if path_prefix and not clean_path.startswith(path_prefix.rstrip("/")):
            # path_prefix 存在且 clean_path 尚未包含前缀 → 加上
            ctx.upstream_url = target.rstrip("/") + path_prefix + clean_path
            log.debug(
                f"[{ctx.req_id}] upstream path: "
                f"{target} + {path_prefix} + ... → {ctx.upstream_url}"
            )
        elif path_prefix and clean_path.startswith(path_prefix.rstrip("/")):
            # path_prefix 存在且 clean_path 已包含前缀 → 直接用 clean_path
            ctx.upstream_url = target.rstrip("/") + clean_path
        else:
            # 无 path_prefix → 保留 /anthropic 路径
            ctx.upstream_url = target.rstrip("/") + "/anthropic" + req_path
            log.debug(
                f"[{ctx.req_id}] upstream path (default): "
                f"{target}/anthropic{req_path}"
            )

        return ctx


class AnthropicForwardStage(Stage):
    async def process(self, ctx: Context) -> Context:
        rt = ctx.request.get("runtime")
        if rt is None or rt.transport is None:
            ctx.should_abort = True
            ctx.response = web.Response(status=502, text="Transport not configured")
            return ctx

        # 记录 upstream_name（从 Runtime 读取）
        upstream_config = rt.upstream_config
        if upstream_config is not None:
            ctx.meta["upstream_name"] = getattr(upstream_config, 'upstream_name', 'default')

        # 通过 Transport 获取上游响应
        resp = await rt.transport.fetch(ctx)

        ctx.status = resp.status
        ctx.resp_headers = resp.headers

        # ── 流式: 从 resp.stream 转发 chunk 并重建 ──
        if resp.stream is not None and resp.status == 200:
            await self._relay_stream(ctx, resp.stream)
        else:
            # ── 非流式: 直接用 transport 返回的 body ──
            ctx.resp_body = resp.body
            ctx.resp_bytes = resp.raw_bytes
            ctx.duration_ms = ctx.elapsed_ms
            ctx.response = web.Response(
                status=resp.status,
                headers=resp.headers,
                body=resp.raw_bytes,
            )
            if ctx.resp_body:
                ctx.meta["usage"] = normalize_anthropic_usage(ctx.resp_body.get("usage", {}))

        from deepseek_tap_proxy.shared.stages import apply_cache_analysis
        apply_cache_analysis(ctx)
        return ctx

    async def _relay_stream(self, ctx: Context, stream: collections.abc.AsyncIterable[bytes]) -> None:
        resp = web.StreamResponse(status=200, headers=ctx.resp_headers)
        await resp.prepare(ctx.request)
        ctx.response = resp
        reconstructor = AnthropicReconstructor()
        try:
            async for chunk in stream:
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
            ctx.resp_body = orjson.loads(decode_bytes) if decode_bytes else None
        except (orjson.JSONDecodeError, json.JSONDecodeError, ValueError):
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
