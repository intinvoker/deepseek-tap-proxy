"""Anthropic proxy handler -- pipeline orchestration only."""
from __future__ import annotations

import asyncio
import logging
from aiohttp import web

from deepseek_tap_proxy.pipeline import Context, Pipeline
from deepseek_tap_proxy.router import ResolvedTarget
from deepseek_tap_proxy.shared.stages import (
    CanonicalStage, SysToolsLockStage, RequestClassifierStage,
)
from deepseek_tap_proxy.anthropic.stages import (
    AnthropicParseStage, AnthropicBlock0ClearStage,
    AnthropicSystemRoleConvertStage,
    AnthropicUpstreamPathStage, AnthropicForwardStage, ReturnStage,
)

log = logging.getLogger("deepseek-tap")


def _persist_chain_nowait(ctx, db) -> None:
    """Write prefix chain synchronously right after pipeline."""
    ca = getattr(ctx, "chain_action", None)
    session = ctx.meta.get("session")
    if ca and ca.action in ("create", "update") and session and ctx.meta.get("intent") != "compaction":
        from deepseek_tap_proxy.shared.db import upsert_prefix_chain, insert_chain_token
        from deepseek_tap_proxy.shared.colors import green
        import json as _json
        chain_tokens = ctx.persist.total_input_tokens or 0
        conn = db.conn
        upsert_prefix_chain(
            conn, session.key, ca.chain_id,
            sys=_json.dumps(ca.sys or ctx.sys),
            tools=_json.dumps(ca.tools or ctx.tools),
            messages=_json.dumps(ca.messages or ctx.messages),
            hash_chain=ca.hash_chain or ctx.meta.get("msg_hash_chain", ""),
            msg_count=ca.msg_count or len(ctx.sys + ctx.tools + ctx.messages),
        )
        insert_chain_token(conn, ca.chain_id,
                           msg_count=ca.msg_count,
                           tokens=chain_tokens,
                           char_count=ca.char_count)
        tag = "+ new" if ca.action == "create" else "~ update"
        print(green(f"  [chain] {tag}: {ca.chain_id[:16]}... ({ca.msg_count} msgs)"))


async def anthropic_proxy_handler(request: web.Request) -> web.StreamResponse:
    # Resolve upstream target via router (config-driven) or fallback to --target
    proxy_ctx = request.app.get("proxy_ctx", {})
    router = proxy_ctx.get("router")
    resolved = None
    if router is not None and router.is_available:
        resolved = router.resolve(
            request.headers.get("Authorization") or request.headers.get("x-api-key")
        )
    if resolved is None:
        resolved = ResolvedTarget(
            target_url=proxy_ctx.get("target_url", ""),
            upstream_name="default",
            matched_key_prefix=None,
        )
    request["upstream"] = resolved
    log.debug(
        f"[upstream] POST /v1/messages → "
        f"{resolved.upstream_name} "
        f"({'prefix ' + resolved.matched_key_prefix if resolved.matched_key_prefix else 'default'})"
    )

    ctx = Context(request=request, meta={})
    ctx.target_url = resolved.target_url
    ctx.meta["upstream_base_url"] = resolved.target_url

    config = request.app.get("config", {})
    lock_sys_tools = getattr(config, "lock_sys_tools", True)

    pipeline = Pipeline([
        AnthropicParseStage(),
        AnthropicBlock0ClearStage(),
        CanonicalStage(),
        RequestClassifierStage(),
        SysToolsLockStage(enabled=lock_sys_tools),
        AnthropicSystemRoleConvertStage(),
        AnthropicUpstreamPathStage(),
        AnthropicForwardStage(),
        ReturnStage(),
    ])
    ctx = await pipeline.execute(ctx)

    # ── 前缀链优先持久化：管道结束后立即写入，确保下一条请求能命中 ──
    db = request.app.get("db")
    proxy_ctx = request.app.get("proxy_ctx", {})
    writer = proxy_ctx.get("writer")
    if db:
        _persist_chain_nowait(ctx, db)

    # Fire-and-forget persist (tracked for completion during shutdown)
    persists: set[asyncio.Task] = proxy_ctx.setdefault("_pending_persists", set())
    if ctx.body_readonly is not None:
        from deepseek_tap_proxy.anthropic.persist import persist_anthropic
        t = asyncio.create_task(persist_anthropic(ctx, db, writer))
        persists.add(t)
        t.add_done_callback(persists.discard)

    return ctx.response or web.Response(status=502, text="No response")
