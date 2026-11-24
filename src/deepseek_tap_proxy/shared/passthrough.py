"""Passthrough 运行时 — 字节级透明转发 + 尽力持久化。

设计目标：

* 请求体：直接使用客户端原始字节（``ctx.raw_body``），**不经过 orjson 重排**；
* 响应体（含 SSE）：逐字节转发，不做缓冲/重排/内容修改；
* 持久化：trace(JSONL) / requests 表 / OTLP 全部保留；usage 解析尽力而为，
  失败标 ``usage_source=unparsed`` 且绝不阻断转发；
* 不写 ``prefix_chains`` / 缓存链，不参与缓存归因。

只有 ``adapter`` 模式下白名单 stage 显式修改 fwd_body 时才会重新序列化请求体
（此时 ``persist.is_modified`` 为真，DB 标 ``intervention=1``）。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import orjson
from aiohttp import web

from deepseek_tap_proxy.pipeline import Context
from deepseek_tap_proxy.policies import (
    ADAPTER, PASSTHROUGH, PolicyDecision, PolicyResolver, KIND_OTHER,
)
from deepseek_tap_proxy.shared.decompress import decompress_body
from deepseek_tap_proxy.shared.headers import prepare_fwd_headers
from deepseek_tap_proxy.shared.usage import REGISTRY, detect_vendor

log = logging.getLogger("deepseek-tap")

_BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_DEFAULT_RESOLVER: PolicyResolver | None = None


def default_policy_resolver() -> PolicyResolver:
    """无 Runtime.policies 时的进程级兜底（单元测试 / 旧调用方）。"""
    global _DEFAULT_RESOLVER
    if _DEFAULT_RESOLVER is None:
        _DEFAULT_RESOLVER = PolicyResolver(None)
    return _DEFAULT_RESOLVER


# ── 预解析（不带任何改写）────────────────────────────────────────────


def guess_api_format(path: str, body: dict | None) -> str:
    """推断 API 格式（用于 usage 解析 / canonical view，仅只读用途）。"""
    clean = (path or "").split("?", 1)[0].rstrip("/")
    if clean.startswith("/anthropic"):
        clean = clean[len("/anthropic"):]
    if clean.endswith("/messages") or clean.endswith("/messages/count_tokens"):
        return "anthropic"
    if (
        clean.endswith("/chat/completions")
        or clean.endswith("/completions")
        or clean.endswith("/responses")
    ):
        return "openai"
    # 未知端点：按 body 特征尽力识别（不影响转发字节）
    if isinstance(body, dict):
        if isinstance(body.get("messages"), list) and "system" in body:
            return "anthropic"
        if isinstance(body.get("messages"), list):
            return "openai"
    return ""


def _guess_stream(body: dict | None) -> bool:
    if isinstance(body, dict):
        return bool(body.get("stream"))
    return False


async def read_and_parse_request(ctx: Context) -> None:
    """读取原始 body（保留字节）+ 尽力解析元数据；不修改请求内容。"""
    request = ctx.request
    raw_body = b""
    if request.method in _BODY_METHODS:
        try:
            raw_body = await request.read()
        except Exception as exc:  # noqa: BLE001 — 读取失败按空 body 转发，不阻断
            log.warning("[%s] passthrough read body failed: %s", ctx.req_id, exc)
    ctx.raw_body = raw_body

    encoding = (request.headers.get("Content-Encoding") or "").lower()
    parse_bytes = raw_body
    if raw_body and encoding in ("zstd", "gzip", "deflate", "br"):
        try:
            parse_bytes = decompress_body(raw_body, encoding)
        except (ImportError, ValueError, OSError):
            log.warning("[%s] passthrough unsupported Content-Encoding: %s", ctx.req_id, encoding)
            parse_bytes = b""

    body = None
    if parse_bytes:
        try:
            body = orjson.loads(parse_bytes)
        except (orjson.JSONDecodeError, ValueError):
            body = None
    ctx.body_readonly = body
    ctx.fwd_body = orjson.loads(orjson.dumps(body)) if body is not None else None
    ctx.fwd_headers = prepare_fwd_headers(request.headers)
    ctx.format = guess_api_format(request.path, body)
    ctx.is_streaming = _guess_stream(body)

    rt = request.get("runtime")
    if rt is not None:
        if not ctx.target_url and getattr(rt, "config", None) is not None:
            ctx.target_url = getattr(rt.config, "target", "")
        ctx.timeout = getattr(rt.config, "timeout", 600) if getattr(rt, "config", None) else 600
    if "upstream_base_url" not in ctx.meta:
        ctx.meta["upstream_base_url"] = ctx.target_url

    now = datetime.now(timezone.utc).isoformat()
    ctx.persist.api_format = ctx.format
    ctx.persist.model = body.get("model", "") if isinstance(body, dict) else ""
    ctx.persist.stream_mode = ctx.is_streaming
    ctx.persist.request_id = ctx.req_id
    ctx.persist.timestamp = now
    # 轻量计数（只读，供 trace/OTLP 观测；不触发 canonical hash 计算）
    if isinstance(body, dict):
        messages = body.get("messages")
        tools = body.get("tools")
        ctx.persist.prompt_messages_count = len(messages) if isinstance(messages, list) else 0
        ctx.persist.prompt_tools_count = len(tools) if isinstance(tools, list) else 0


# ── 客户端检测 + 模式决策 ────────────────────────────────────────────


def detect_client(ctx: Context):
    """复用既有 plugin 检测链，返回 (plugin, result)（可能为 (None, None)）。"""
    from deepseek_tap_proxy.plugins.registry import ALL_PLUGINS, detect_plugin

    try:
        return detect_plugin(ctx, ALL_PLUGINS)
    except Exception:  # noqa: BLE001 — 检测失败一律按未知客户端处理
        log.exception("[%s] client detection failed", ctx.req_id)
        return None, None


def resolve_request_policy(ctx: Context) -> tuple[PolicyDecision, object, object]:
    """检测客户端并解析模式；同时把身份写回 ctx（只读阶段，不改 body）。"""
    rt = ctx.request.get("runtime")
    resolver = getattr(rt, "policies", None) if rt is not None else None
    if not isinstance(resolver, PolicyResolver):
        resolver = default_policy_resolver()

    plugin, result = detect_client(ctx)
    client_type = result.client_type if result is not None else ""
    decision = resolver.resolve(
        client_type=client_type,
        path=ctx.request.path,
        body=ctx.body_readonly,
        headers=ctx.request.headers,
    )

    if result is not None:
        ctx.client_type = result.client_type
        ctx.chat_type = result.chat_type
        ctx.session_id = result.session_id
        ctx.chain_id = result.chain_id
        ctx.parent_chain_id = result.parent_chain_id
        ctx.meta["session_key_source"] = "plugin" if result.session_id else "none"
        ctx.persist.session_id = result.session_id
        ctx.persist.client_type = result.client_type
        ctx.persist.chat_type = result.chat_type
    else:
        ctx.client_type = "generic"
        ctx.chat_type = "normal"
        ctx.session_id = ""
        ctx.chain_id = ""
        ctx.parent_chain_id = ""
        ctx.meta["session_key_source"] = "none"
        ctx.persist.client_type = "generic"
        ctx.persist.chat_type = "normal"

    ctx.meta["client_type"] = ctx.client_type
    ctx.meta["mode"] = decision.mode
    ctx.meta["kind"] = decision.kind
    ctx.meta["policy_source"] = decision.source
    ctx.meta["policy_reason"] = decision.reason
    return decision, plugin, result


def adapter_stages_for(ctx: Context, plugin) -> list:
    """adapter 模式的显式白名单 stage（未实现该 hook 的 plugin → 空列表）。"""
    if plugin is None:
        return []
    builder = getattr(plugin, "build_adapter_stages", None)
    if builder is None:
        return []
    try:
        return list(builder(ctx))
    except Exception:  # noqa: BLE001 — 适配构建失败 → 退化为纯 passthrough
        log.exception("[%s] build_adapter_stages failed", ctx.req_id)
        return []


def force_managed_for_enforcement(ctx: Context, decision: PolicyDecision) -> bool:
    """``--enforce-client-detection`` 开启时保留既有 422 门禁。

    门禁是「身份认证」而非「模式」：未识别客户端的默认 passthrough 会让匿名
    请求绕过 422。此处把「default → passthrough」的决策降级回 managed 管线，
    只有 orchestrate 的检测门禁会返回 422（显式 override / client 映射不受影响）。
    """
    if decision.source != "default" or decision.mode != PASSTHROUGH:
        return False
    rt = ctx.request.get("runtime")
    cfg = getattr(rt, "config", None) if rt is not None else None
    return bool(getattr(cfg, "enforce_client_detection", False))


# ── 转发 ────────────────────────────────────────────────────────────


def _split_qs(path_qs: str) -> tuple[str, str]:
    if "?" in path_qs:
        clean, query = path_qs.split("?", 1)
        return clean, "?" + query
    return path_qs, ""


def build_upstream_url(ctx: Context) -> str:
    """按既有路径映射规则构造上游 URL（不读取/不改写 body）。"""
    target = (ctx.target_url or "").rstrip("/")
    path_qs = ctx.request.path_qs
    upstream = ctx.request.get("upstream")
    path_prefix = getattr(upstream, "path_prefix", None) if upstream is not None else None

    is_anthropic = path_qs.startswith("/anthropic") or ctx.format == "anthropic"
    if not is_anthropic:
        # OpenAI / 未知端点：与既有 openai handler 一致，不做 /anthropic 前缀。
        return target + path_qs

    if path_qs.startswith("/anthropic"):
        req_path = path_qs[len("/anthropic"):]
    else:
        req_path = path_qs
    clean_path, query = _split_qs(req_path)
    clean_path = clean_path.rstrip("/")

    if path_prefix:
        prefix = path_prefix.rstrip("/")
        if clean_path.startswith(prefix):
            return target + clean_path + query
        return target + path_prefix + clean_path + query

    if path_qs.startswith("/anthropic"):
        return target + "/anthropic" + req_path
    # /v1/messages 等无前缀 Anthropic 路由：与 managed 管线保持一致，补 /anthropic
    return target + "/anthropic" + path_qs


def _with_request_id(ctx: Context, headers: dict) -> dict:
    """响应头处理：去上游同名 x-request-id，注入 proxy 的关联 id。

    与 managed 管线一致：响应**体**字节原样，但 ``X-Request-Id``
    是 proxy 全局关联不变量（trace/日志/客户端可跨流式关联）。上游若自带同名头
    则替换，避免重复。
    """
    out = {k: v for k, v in headers.items() if k.lower() != "x-request-id"}
    out["X-Request-Id"] = ctx.req_id
    return out


async def _forward(ctx: Context) -> None:
    rt = ctx.request.get("runtime")
    transport = getattr(rt, "transport", None) if rt is not None else None
    if transport is None:
        ctx.status = 502
        ctx.response = web.Response(status=502, text="Transport not configured")
        return

    upstream = ctx.request.get("upstream")
    if upstream is not None:
        ctx.meta["upstream_name"] = getattr(upstream, "upstream_name", "default")

    try:
        resp = await transport.fetch(ctx)
    except Exception as exc:  # noqa: BLE001 — 上游异常不吞掉，转为 502 留痕
        log.error("[%s] passthrough upstream error: %s", ctx.req_id, exc)
        ctx.status = 502
        ctx.response = web.Response(status=502, text=str(exc))
        return

    ctx.status = resp.status
    ctx.resp_headers = dict(resp.headers or {})
    headers = _with_request_id(ctx, ctx.resp_headers)

    if resp.stream is not None:
        out = web.StreamResponse(status=resp.status, headers=headers)
        await out.prepare(ctx.request)
        ctx.response = out
        chunks: list[bytes] = []
        try:
            async for chunk in resp.stream:
                await out.write(chunk)
                chunks.append(chunk)
        except (ConnectionError, asyncio.CancelledError):
            pass
        try:
            await out.write_eof()
        except Exception:  # noqa: BLE001 — 客户端已断开
            pass
        ctx.resp_bytes = b"".join(chunks)
        ctx.duration_ms = ctx.elapsed_ms
        ctx.resp_body = _reconstruct_stream(ctx, ctx.resp_bytes)
    else:
        raw = resp.raw_bytes or b""
        ctx.resp_bytes = raw
        ctx.duration_ms = ctx.elapsed_ms
        ctx.resp_body = resp.body if isinstance(resp.body, (dict, list)) else None
        ctx.response = web.Response(status=resp.status, headers=headers, body=raw)


def _reconstruct_stream(ctx: Context, raw: bytes):
    """SSE 尽力重建（仅用于 trace/usage，不影响转发字节）。"""
    if not raw:
        return None
    try:
        if ctx.format == "anthropic":
            from deepseek_tap_proxy.anthropic.sse import AnthropicReconstructor

            recon = AnthropicReconstructor()
            recon.feed_bytes(raw)
            return recon.reconstruct()
        if ctx.format == "openai":
            from deepseek_tap_proxy.openai.sse import OpenaiReconstructor

            recon = OpenaiReconstructor()
            recon.feed_bytes(raw)
            return recon.reconstruct()
    except Exception:  # noqa: BLE001 — 解析失败只影响留痕，不影响转发
        log.debug("[%s] passthrough SSE reconstruct failed", ctx.req_id, exc_info=True)

    # 通用兜底：取最后一个可解析的 data: JSON object（新 API 的 SSE）
    last_obj = None
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[len(b"data:"):].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            obj = orjson.loads(payload)
        except (orjson.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            last_obj = obj
    return last_obj


# ── usage / 持久化 ──────────────────────────────────────────────────


def parse_usage_best_effort(ctx: Context) -> None:
    """尽力从响应中提取 usage；失败标 ``usage_source=unparsed``，绝不抛错。"""
    persist = ctx.persist
    persist.usage_source = "unparsed"
    try:
        raw = None
        if isinstance(ctx.resp_body, dict):
            raw = ctx.resp_body.get("usage")
        if not isinstance(raw, dict) or not raw:
            return
        vendor = ctx.meta.get("upstream_name", "") or detect_vendor(raw)
        normalized = REGISTRY.extract(raw, ctx.format or "", vendor)
        if not isinstance(normalized, dict):
            return
        total_in = int(normalized.get("total_input_tokens") or normalized.get("input_tokens") or 0)
        output = int(normalized.get("output_tokens") or 0)
        if total_in == 0 and output == 0:
            return
        persist.total_input_tokens = total_in
        persist.output_tokens = output
        persist.cache_hit_tokens = int(normalized.get("cache_read_input_tokens") or 0)
        persist.cache_miss_tokens = int(normalized.get("cache_miss_input_tokens") or 0)
        persist.cache_creation_tokens = int(normalized.get("cache_creation_input_tokens") or 0)
        denom = persist.cache_hit_tokens + persist.cache_miss_tokens
        persist.cache_hit_ratio = (
            round(persist.cache_hit_tokens / denom, 4) if denom > 0 else 0.0
        )
        persist.has_reasoning = bool(ctx.resp_has_reasoning)
        persist.has_tool_calls = bool(ctx.resp_has_tool_calls)
        persist.usage_source = "parsed"
    except Exception:  # noqa: BLE001 — usage 解析绝不阻断
        log.debug("[%s] passthrough usage parse failed", ctx.req_id, exc_info=True)


async def persist_passthrough(ctx: Context) -> None:
    """保留全部持久化（requests 表 / trace JSONL / OTLP），不写缓存链。"""
    from deepseek_tap_proxy.shared.consumers import run_consumers, store_phase1

    request = ctx.request
    app = getattr(request, "app", None)
    db = None
    proxy_ctx: dict = {}
    if app is not None:
        try:
            db = app.get("db")
            proxy_ctx = app.get("proxy_ctx", {}) or {}
        except (AttributeError, TypeError):
            db = None
            proxy_ctx = {}
    rt = request.get("runtime")
    store = getattr(rt, "store", None) if rt is not None else None
    writer = proxy_ctx.get("writer")
    otlp_writer = proxy_ctx.get("otlp_log_writer")
    run_id = proxy_ctx.get("run_id", "")

    # 非 managed 模式不建 sessions 行（不做缓存归因）：
    # 清空 DB 侧的 session_id，避免 PG 外键引用不存在的 sessions 行
    # （client_type 仍写入 session_type 列，身份可观测）。
    if getattr(ctx.persist, "mode", "") != "managed" and ctx.persist.session_id:
        ctx.meta.setdefault("detected_session_id", ctx.persist.session_id)
        ctx.persist.session_id = ""

    try:
        if store is not None and ctx.body_readonly is not None:
            await store_phase1(ctx, store)
        await run_consumers(
            ctx,
            writer=writer,
            store=store,
            otlp_writer=otlp_writer,
            run_id=run_id,
        )
    except Exception:  # noqa: BLE001 — 持久化失败不阻断响应
        log.exception("[%s] passthrough persist failed", ctx.req_id)


async def execute_passthrough(
    ctx: Context,
    *,
    mode: str = PASSTHROUGH,
    plugin=None,
    topic: str | None = None,
) -> Context:
    """passthrough / adapter 统一执行入口：可选适配 → 原始字节转发 → 持久化。"""
    ctx.persist.mode = mode
    if topic is not None:
        ctx.topic = topic

    if mode == ADAPTER:
        for stage in adapter_stages_for(ctx, plugin):
            try:
                ctx = await stage.process(ctx)
            except Exception:  # noqa: BLE001 — 适配失败退化为透明转发
                log.exception("[%s] adapter stage failed, falling back to raw", ctx.req_id)
                break

    if ctx.persist.is_modified and ctx.fwd_body is not None:
        try:
            ctx.fwd_raw_bytes = orjson.dumps(ctx.fwd_body)
        except (TypeError, ValueError):
            ctx.fwd_raw_bytes = ctx.raw_body
    else:
        ctx.fwd_raw_bytes = ctx.raw_body

    ctx.upstream_url = build_upstream_url(ctx)
    await _forward(ctx)
    parse_usage_best_effort(ctx)
    await persist_passthrough(ctx)
    return ctx
