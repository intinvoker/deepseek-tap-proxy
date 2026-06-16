"""Transport — 响应提供者接口。

ForwardStage 不关心响应来源，只调 transport.fetch(ctx)。

运行时: AiohttpTransport(session) → 真实上游 HTTP 调用
Replay:  TraceTransport(trace_response) → trace JSON (Inspector 侧)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

log = logging.getLogger("deepseek-tap")


from collections.abc import AsyncIterable


@dataclass
class TransportResponse:
    """上游响应的统一表示。

    - 非流式: body / raw_bytes 直接填充
    - 流式:   body=None, stream 提供原始 chunk 迭代器
             ForwardStage 通过 async for chunk in resp.stream 转发
             并自行重建 resp_body

    stream 是 AsyncIterable[bytes]，任何 Transport 实现（包括
    TraceTransport）都可以提供，不泄漏底层 HTTP 库类型。
    """
    status: int
    headers: dict = field(default_factory=dict)
    body: dict | None = None
    raw_bytes: bytes = b""
    # 流式响应: 原始字节 chunk 迭代器（格式无关）
    stream: AsyncIterable[bytes] | None = None


class Transport:
    """响应提供者接口。ForwardStage 通过此接口获取响应。"""

    async def fetch(self, ctx) -> TransportResponse:
        raise NotImplementedError


class AiohttpTransport(Transport):
    """通过 aiohttp 发起真实 HTTP 请求（运行时）。"""

    def __init__(self, session, timeout: int = 600):
        self._session = session
        self._timeout = timeout

    async def fetch(self, ctx) -> TransportResponse:
        target = ctx.target_url
        if not target:
            return TransportResponse(status=502)

        upstream_url = ctx.upstream_url
        if not upstream_url:
            req_path = ctx.request.path_qs
            if getattr(ctx, 'format', '') == "anthropic" and req_path.startswith("/anthropic"):
                req_path = req_path[len("/anthropic"):]
                upstream_url = target.rstrip("/") + "/anthropic" + req_path
            else:
                upstream_url = target.rstrip("/") + req_path

        body_bytes = None
        if ctx.fwd_body is not None:
            body_bytes = json.dumps(ctx.fwd_body, ensure_ascii=False).encode("utf-8")
            ctx.fwd_headers["Content-Length"] = str(len(body_bytes))
        else:
            ctx.fwd_headers.pop("Content-Length", None)

        try:
            import aiohttp
            upstream_resp = await self._session.request(
                method=ctx.request.method, url=upstream_url,
                headers=ctx.fwd_headers, data=body_bytes,
                timeout=aiohttp.ClientTimeout(total=self._timeout, sock_read=self._timeout),
            )
        except Exception as exc:
            log.error(f"[{ctx.req_id}] upstream error: {exc}")
            return TransportResponse(status=502)

        from deepseek_tap_proxy.shared.headers import filter_hop_by_hop
        headers = filter_hop_by_hop(dict(upstream_resp.headers))

        # ── 流式: 不读 body，返回 chunk 流让 ForwardStage 边转发边重建 ──
        if getattr(ctx, 'is_streaming', False) and upstream_resp.status == 200:
            async def _iter_chunks():
                async for chunk in upstream_resp.content.iter_any():
                    yield chunk
            return TransportResponse(
                status=upstream_resp.status,
                headers=headers,
                body=None,
                raw_bytes=b"",
                stream=_iter_chunks(),
            )

        # ── 非流式: 读完整 body，解析 JSON ──
        resp_bytes = await upstream_resp.read()
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

        body = None
        if decode_bytes:
            try:
                body = json.loads(decode_bytes)
            except (json.JSONDecodeError, ValueError):
                pass

        return TransportResponse(
            status=upstream_resp.status,
            headers=headers,
            body=body,
            raw_bytes=resp_bytes,
        )
