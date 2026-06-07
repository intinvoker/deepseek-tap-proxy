"""OpenAI API pipeline stages — format-aware entry and exit points.

Format-agnostic stages (Msg0Lock, PrefixLock) live in shared/stages.py.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from aiohttp import web

from deepseek_tap_proxy.pipeline import Context, Stage
from deepseek_tap_proxy.shared.decompress import decompress_body
from deepseek_tap_proxy.shared.headers import HOP_BY_HOP, filter_hop_by_hop
from deepseek_tap_proxy.shared.stages import apply_cache_analysis, emit_request_log
from deepseek_tap_proxy.openai.sse import OpenaiReconstructor

log = logging.getLogger("deepseek-tap")


_SUMMARY_TAG_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)


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
    """Parse the incoming request: read body, validate path, extract metadata.

    Sets ``ctx.format = "openai"`` so the canonical view properties
    (``sys`` / ``tools`` / ``messages``) map to the OpenAI message layout.
    """

    async def process(self, ctx: Context) -> Context:
        body = await ctx.request.read()

        fwd_headers = dict(ctx.request.headers)
        fwd_headers.pop("Host", None)
        fwd_headers = filter_hop_by_hop(fwd_headers)

        req_content_encoding = ctx.request.headers.get("Content-Encoding", "").lower()
        if req_content_encoding in ("zstd", "gzip", "deflate", "br"):
            try:
                body = decompress_body(body, req_content_encoding)
            except (ImportError, ValueError, OSError):
                log.warning(f"[{ctx.req_id}] unsupported Content-Encoding: {req_content_encoding}, forwarding as-is")
        fwd_headers["Accept-Encoding"] = "identity"
        ctx.fwd_headers = fwd_headers

        try:
            parsed = json.loads(body) if body else None
        except (json.JSONDecodeError, ValueError):
            parsed = None

        # body_readonly: set once, never modified (used for trace/debug)
        ctx.body_readonly = parsed
        if isinstance(parsed, dict):
            ctx.fwd_body = copy.deepcopy(parsed)
        else:
            ctx.fwd_body = parsed

        # ── 规范形式 view 就绪 ──
        ctx.format = "openai"

        ctx.is_streaming = parsed.get("stream", False) if isinstance(parsed, dict) else False

        model = parsed.get("model", "") if isinstance(parsed, dict) else ""

        normalized_path = "/" + ctx.request.path.lstrip("/").rstrip("/")
        allowed_suffixes = ("/chat/completions",)
        if not normalized_path.endswith(allowed_suffixes):
            log.warning(f"[{ctx.req_id}] blocked path: {ctx.request.path}")
            ctx.should_abort = True
            ctx.response = web.Response(status=404, text="Not Found")
            return ctx

        if isinstance(parsed, dict):
            all_msgs = ctx.sys + ctx.messages  # canonical: sys at [0], then conversation
            if all_msgs:
                # 规范形式 meta 已迁移到 CanonicalStage
                pass

        proxy_ctx = ctx.request.app.get("proxy_ctx", {})
        if not ctx.target_url:
            ctx.target_url = proxy_ctx.get("target_url", "")

        # ── persist fields ──
        ctx.persist.api_format = "openai"
        if isinstance(ctx.body_readonly, dict):
            ctx.persist.model = ctx.body_readonly.get("model", "")
        else:
            ctx.persist.model = ""
        ctx.persist.stream_mode = ctx.is_streaming
        ctx.persist.request_id = ctx.req_id
        ctx.persist.timestamp = datetime.now(timezone.utc).isoformat()

        return ctx


class ForwardStage(Stage):
    """Forward the request to the upstream API and capture the response."""

    async def process(self, ctx: Context) -> Context:
        proxy_ctx = ctx.request.app.get("proxy_ctx", {})
        session: aiohttp.ClientSession = proxy_ctx.get("session")
        upstream = ctx.request.get("upstream")
        if upstream is None:
            target = proxy_ctx.get("target_url", "")
        else:
            target = upstream.target_url
        timeout_sec: int = proxy_ctx.get("timeout", 600)

        if not session or not target:
            ctx.should_abort = True
            ctx.response = web.Response(status=502, text="Upstream not configured")
            return ctx

        upstream_url = target.rstrip("/") + ctx.request.path
        if upstream is not None:
            ctx.meta["upstream_name"] = upstream.upstream_name

        body_bytes = json.dumps(
            ctx.fwd_body, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8") if ctx.fwd_body else None

        if body_bytes is not None:
            ctx.fwd_headers["Content-Length"] = str(len(body_bytes))

        try:
            upstream_resp = await session.request(
                method="POST",
                url=upstream_url,
                headers=ctx.fwd_headers,
                data=body_bytes,
                timeout=aiohttp.ClientTimeout(total=timeout_sec, sock_read=timeout_sec),
            )
        except Exception as exc:
            ctx.status = 502
            ctx.duration_ms = ctx.elapsed_ms
            emit_request_log(ctx)
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

        emit_request_log(ctx)
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
            apply_cache_analysis(ctx)

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
            apply_cache_analysis(ctx)

        ctx.response = web.Response(
            status=upstream_resp.status,
            headers=ctx.resp_headers,
            body=resp_bytes,
        )



class ReturnStage(Stage):
    """Final stage: the response is already set by ForwardStage."""

    async def process(self, ctx: Context) -> Context:
        ctx.should_abort = True
        # Inject proxy request_id into response headers for correlation
        if ctx.response and not ctx.is_streaming:
            ctx.response.headers["X-Request-Id"] = ctx.req_id
        # Strip <summary> tags for compaction-rewritten responses
        if ctx.meta.get("is_compaction_rewritten") and ctx.resp_body is not None:
            try:
                content = ctx.resp_body.get("choices", [{}])[0].get("message", {}).get("content", "")
                m = _SUMMARY_TAG_RE.search(content)
                if m:
                    ctx.resp_body["choices"][0]["message"]["content"] = m.group(1).strip()
                elif "<summary>" in content:
                    idx = content.index("<summary>") + len("<summary>")
                    ctx.resp_body["choices"][0]["message"]["content"] = content[idx:].strip()
                # Rebuild response bytes
                new_body = json.dumps(ctx.resp_body, ensure_ascii=False).encode("utf-8")
                ctx.response = web.Response(
                    status=ctx.response.status,
                    headers=dict(ctx.response.headers),
                    body=new_body,
                )
                ctx.response.headers["Content-Length"] = str(len(new_body))
            except Exception:
                pass  # 剥离失败不影响主流程
        return ctx
