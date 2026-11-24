"""统一编排函数 — OpenAI / Anthropic 共用。"""
from __future__ import annotations

from aiohttp import web

from deepseek_tap_proxy.pipeline import Pipeline, Context
from deepseek_tap_proxy.plugins.registry import detect_plugin, ALL_PLUGINS


async def _persist_detected_session(ctx: Context, store) -> None:
    """session 解析后持久化到 MemoryStore + SQLite（根治 sessions 表冻结）。

    背景：7/13 重构（5a2e848 多客户端 pipeline 架构）删除 RequestAnalysisStage 的
    ``upsert_session`` 调用后，sessions 表 6/24 起不再落库（last_seen 冻结）→
    SysToolsLock 的 canonical 经 ``set_session_canonical`` UPDATE 0 行静默失败 →
    跨重启 canonical 100% 丢失。

    此处为恢复点：detect 解析出 session_id 后立即 upsert，使 sessions 表重新维护、
    canonical 获得落盘目标行。无 session_id（generic 无 UUID）时不执行。
    """
    if not store or not ctx.session_id:
        return
    await store.upsert_session(
        ctx.session_id, ctx.client_type,
        curr_hashes=ctx.messages_hashes,
    )


async def orchestrate(ctx, format_pipeline, plugins, format_tail):
    """管道1（解析+context[]化）→ 检测 → 管道2（客户端处理）。

    Args:
        ctx: Context 对象（body_readonly 已设，format 已设）
        format_pipeline: 管道1 Pipeline 实例
        plugins: 检测链 Plugin 列表（默认 ALL_PLUGINS）
        format_tail: 管道2 格式收尾 Stage 列表（含 Forward/Return）

    Returns:
        ctx（管道2 执行完毕，ctx.response 就绪）
    """
    # ── 内存维护（每次请求入口）──
    rt = ctx.request.get("runtime")
    if rt and rt.store:
        rt.store.maybe_maintain()

    # ── resp_body 截断阈值注入（写入侧字节有界）──
    cfg = ctx.request.app.get("config") if ctx.request is not None else None
    if cfg is not None:
        max_bytes = getattr(cfg, "trace_record_max_bytes", 0)
        # isinstance 防御：测试场景 request.app 为 MagicMock 时 getattr 返回 mock
        if isinstance(max_bytes, int):
            ctx.resp_body_max_bytes = max_bytes

    # ── 管道1：解析 + context[]化 ──
    ctx = await format_pipeline.execute(ctx)
    if ctx.should_abort:
        return ctx

    # ── 检测 ──
    plugin, result = detect_plugin(ctx, plugins or ALL_PLUGINS)
    if plugin is None or result is None:
        # 门禁可配置（--enforce-client-detection，默认 False）：
        # 未识别请求降级 generic 直通（无会话键 → SysToolsLock/prefix lock
        # 天然 pass，不参与缓存锁定）。原 fail-closed 行为（422 拒绝）
        # 在配置开启后保留——探活 ping 等无身份请求默认放行。
        _rt = ctx.request.get("runtime")
        _cfg = _rt.config if _rt else None
        if getattr(_cfg, "enforce_client_detection", False):
            ctx.should_abort = True
            ctx.response = web.Response(
                status=422,
                text="Unrecognized client: no session identifier found",
            )
            return ctx
        ctx.client_type = "generic"
        ctx.chat_type = "normal"
        ctx.session_id = ""
        ctx.chain_id = ""
        ctx.parent_chain_id = ""
        ctx.meta["session_key_source"] = "none"
        ctx.persist.client_type = "generic"
        ctx.persist.chat_type = "normal"
        ctx.persist.session_id = ""
    else:
        ctx.client_type = result.client_type
        ctx.chat_type = result.chat_type
        ctx.session_id = result.session_id
        ctx.chain_id = result.chain_id
        ctx.parent_chain_id = result.parent_chain_id
        ctx.meta["session_key_source"] = "plugin" if result.session_id else "none"
        ctx.persist.session_id = result.session_id
        ctx.persist.client_type = result.client_type
        ctx.persist.chat_type = result.chat_type

    # ── 根治：session 持久化（7/13 重构丢失的 upsert_session 调用）──
    # 在管道2（SysToolsLock 等）之前落盘 session 行，保证 canonical 有 UPDATE 目标
    _rt = ctx.request.get("runtime")
    await _persist_detected_session(ctx, _rt.store if _rt else None)

    # ── 管道2：客户端处理 ──
    middle = plugin.build_stages(ctx) if plugin else []
    pipeline = Pipeline(middle + list(format_tail))
    return await pipeline.execute(ctx)
