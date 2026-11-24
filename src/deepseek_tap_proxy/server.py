from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from datetime import datetime, timezone

import aiohttp
import orjson
from aiohttp import web

from deepseek_tap_proxy import analysis_api
from deepseek_tap_proxy.config import ProxyConfig
from deepseek_tap_proxy.db import Database
from deepseek_tap_proxy.shared.db import set_meta
from deepseek_tap_proxy.shared.events import ChainEventWriter
from deepseek_tap_proxy.shared.headers import HOP_BY_HOP, filter_hop_by_hop
from deepseek_tap_proxy.shared.log import OtlpLogWriter, TraceWriter
from deepseek_tap_proxy.shared.store import MemoryStore
from deepseek_tap_proxy.router import UpstreamRouter
from deepseek_tap_proxy.policies import PolicyResolver
from deepseek_tap_proxy.catchall import catchall_handler
from deepseek_tap_proxy.openai.routes import setup_openai_routes
from deepseek_tap_proxy.anthropic.routes import setup_anthropic_routes
from deepseek_tap_proxy.shared.runtime import Runtime
from deepseek_tap_proxy.shared.transport import AiohttpTransport

log = logging.getLogger("deepseek-tap")

# 事件日志（权威事实源）连续失败批数阈值：达此值 /health 才 degraded。
# failures 成功归零（瞬态自愈），连续 5 批失败 = 持续故障（对齐 store 侧保守风格）。
_EVENTS_WRITER_FAILURE_THRESHOLD = 5

# 只读查询 API 接受的 request_id 形态：与 Context 生成的 ``req_<12hex>`` +
# 客户端外传的 ``32 hex`` 同构（大小写不敏感）。查询前归一为小写——写侧
# request_id 恒为小写（外传 id 在 Context 中已 lower）。
_REQUEST_ID_RE = re.compile(r"^(?:[0-9a-f]{32}|req_[0-9a-f]{12})$", re.IGNORECASE)


def _body_bytes(body) -> int:
    """摘要用 body 字节数（口径与写侧一致：``len(orjson.dumps(body))``）。"""
    if body is None:
        return 0
    try:
        return len(orjson.dumps(body))
    except (TypeError, ValueError):
        return 0


def _summarize_trace(trace: dict, run_id: str) -> dict:
    """trace 记录 → 只读摘要。

    刻意**不返回** ``request.body`` / ``response.body`` / headers（prompt、
    cookie、Authorization 等敏感内容不得经 HTTP API 暴露）。
    """
    req = trace.get("request") or {}
    resp = trace.get("response") or {}
    usage = trace.get("usage") or {}
    record_id = trace.get("request_id") or ""
    return {
        "request_id": record_id,
        "timestamp": trace.get("timestamp", ""),
        "duration_ms": trace.get("duration_ms", 0),
        "api_format": trace.get("api_format", ""),
        "upstream_name": trace.get("upstream_name", ""),
        "upstream_base_url": trace.get("upstream_base_url", ""),
        "request": {
            "method": req.get("method", ""),
            "path": req.get("path", ""),
            "bytes": _body_bytes(req.get("body")),
        },
        "response": {
            "status": resp.get("status", 0),
            "bytes": _body_bytes(resp.get("body")),
        },
        "usage": {
            "total_input_tokens": usage.get("total_input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "cache_hit_tokens": usage.get("cache_hit_tokens", 0),
            "cache_miss_tokens": usage.get("cache_miss_tokens", 0),
            "cache_hit_ratio": usage.get("cache_hit_ratio", 0.0),
            "prompt_messages_count": usage.get("prompt_messages_count", 0),
            "prompt_tools_count": usage.get("prompt_tools_count", 0),
            "has_reasoning": usage.get("has_reasoning", False),
            "has_tool_calls": usage.get("has_tool_calls", False),
        },
        "trace_ref": {"run_id": run_id, "request_id": record_id},
    }


def _read_trace_summary(run_dir: str, request_id: str, run_id: str) -> dict | None:
    """worker 线程内完成「读 trace + 构造摘要」。

    摘要含 ``len(orjson.dumps(body))``（大 body 可达 MB 级）——文件读取与摘要
    计算放在同一 worker 线程，避免在事件循环线程做序列化阻塞其他请求。
    """
    trace = analysis_api.get_trace(run_dir, request_id)
    if trace is None:
        return None
    return _summarize_trace(trace, run_id)


async def _requests_read_handler(request: web.Request) -> web.Response:
    """``GET /requests/{request_id}`` — 只读查询单条请求摘要。

    先 flush TraceWriter（写侧缓冲，不 flush 会读不到最新记录），再在 worker
    线程完成「读 trace + 构造摘要」（含 body 序列化），事件循环只 await 结果。
    """
    raw_id = request.match_info["request_id"].strip()
    if not _REQUEST_ID_RE.match(raw_id):
        return web.json_response({"error": "invalid request id"}, status=400)
    request_id = raw_id.lower()

    proxy_ctx = request.app["proxy_ctx"]
    await request.app["writer"].flush()
    summary = await asyncio.to_thread(
        _read_trace_summary,
        proxy_ctx["run_dir"], request_id, proxy_ctx.get("run_id", ""),
    )
    if summary is None:
        return web.json_response({"error": "not found"}, status=404)
    return web.json_response(summary)


def _create_run_dir(trace_dir: str) -> tuple[str, str]:
    """创建 run 目录，返回 (run_id, run_dir_path).

    结构: {trace_dir}/{run_id}/
    run_id = HHMMSS_random8

    注：内层日期目录在写入第一条记录时动态创建，不在启动时预创建。
    """
    now = datetime.now(timezone.utc)
    time_str = now.strftime("%H%M%S")
    random_str = uuid.uuid4().hex[:8]
    run_id = f"{time_str}_{random_str}"
    run_dir = os.path.join(trace_dir, run_id)
    os.makedirs(run_dir, exist_ok=False)
    return run_id, run_dir


async def create_proxy_app(config: ProxyConfig, *, add_health: bool = True) -> web.Application:
    app = web.Application(client_max_size=100 * 1024**2)

    run_id, run_dir = _create_run_dir(config.trace_dir)
    writer = TraceWriter(
        run_dir, run_id,
        flush_interval=config.trace_flush_interval,
        batch_max=config.trace_batch_max,
        file_max_bytes=config.trace_file_max_bytes,
    )
    await writer.start_flusher()  # 启动后台定时 flush（低流量落盘）

    # 事件日志权威化（Garnet AOF 理念）：ChainEventWriter 接线 —— 写路径状态
    # 变更 append 到跨重启稳定的 chain-events.jsonl，SQLite 仅作 checkpoint 投影。
    events_writer = ChainEventWriter(
        config.events_path, file_max_bytes=config.event_file_max_bytes,
    )
    events_writer.start()

    db = Database(config.db_type, config.db_path, config.db_dsn)
    db.init_db()
    # Close the main-thread connection so the fire-and-forget writer thread
    # (started by MemoryStore.__init__) does not hit "database is locked".
    db.close()

    # MemoryStore: load existing data from SQLite into in-memory store (0 I/O runtime)
    # 事件权威化：传 events_path 触发增量重放（checkpoint_offset 起），
    # 传 events_writer 使启动后写路径继续 append（重放本身 _persist_event=False 防循环）。
    store = await MemoryStore.load_from(
        db, _db=db,
        enable_governance=config.enable_governance,
        db_commit_interval=config.db_commit_interval,
        db_batch_max=config.db_batch_max,
        db_queue_max=config.db_queue_max,
        events_path=config.events_path,
        events_writer=events_writer,
        max_memory_bytes=config.max_memory_bytes,
        ttl_hours=config.ttl_hours,
        max_moments=config.max_moments,
    )

    client_session = aiohttp.ClientSession(
        headers={"User-Agent": "DeepSeekTap/0.1"},
    )

    # Load upstream router (config file driven, disabled if no config)
    router = UpstreamRouter(config.upstreams_config or None)
    # 模式策略（passthrough/adapter/managed）：配置驱动 + 热加载；无配置走内置默认
    policies = PolicyResolver(config.policies_config or None)

    # OTLP log writer — 默认写入 run_dir/{date}/otlp.jsonl
    # 如果 config.otlp_log_dir 已设置（测试用），使用该路径
    _otlp_base = config.otlp_log_dir or run_dir
    otlp_writer = OtlpLogWriter(_otlp_base)

    app["proxy_ctx"] = {
        "writer": writer,
        "session": client_session,
        "target_url": config.target,
        "timeout": config.timeout,
        "session_id": run_id,
        "run_id": run_id,
        "run_dir": run_dir,
        "otlp_log_writer": otlp_writer,
    }
    app["db"] = db
    app["config"] = config
    app["writer"] = writer
    app["events_writer"] = events_writer
    app["store"] = store
    app["policies"] = policies

    # 🆕 构造 Runtime（强类型依赖注入）
    runtime = Runtime(
        store=store,
        config=config,
        transport=AiohttpTransport(client_session, config.timeout),
        session=client_session,
        router=router,
        policies=policies,
        run_id=run_id,
        otlp_writer=otlp_writer,
    )

    # 🆕 Middleware: 对每个请求注入 Runtime + upstream
    @web.middleware
    async def runtime_middleware(request: web.Request, handler):
        request["runtime"] = runtime
        # Resolve upstream for this request
        resolved = await _resolve_upstream_for_request(request)
        request["upstream"] = resolved
        return await handler(request)

    app.middlewares.append(runtime_middleware)

    setup_openai_routes(app)
    setup_anthropic_routes(app)

    # Wrap GET handlers with upstream resolution
    async def _proxy_get_with_upstream(request: web.Request) -> web.Response:
        upstream = await _resolve_upstream_for_request(request)
        request["upstream"] = upstream
        return await _proxy_get_handler(request)

    app.router.add_get("/v1/models", _proxy_get_with_upstream)
    app.router.add_get("/models", _proxy_get_with_upstream)
    app.router.add_get("/v1", _proxy_get_with_upstream)
    if add_health:
        app.router.add_get("/health", _health_handler)

    # 只读请求查询 API（与 /health 同级）——不写 DB / 不转发上游
    app.router.add_get("/requests/{request_id}", _requests_read_handler)

    # Internal endpoint: drain MemoryStore write queue + flush TraceWriter (used by tests)
    async def _flush_handler(request: web.Request) -> web.Response:
        if store:
            await store.flush()
        await writer.flush()
        return web.Response(text="ok")

    app.router.add_post("/__flush", _flush_handler)

    # 运行时重载 upstreams 配置（读同一配置文件路径，原子替换，失败保持旧配置）
    # 部署形态：改挂载的 upstreams.yaml → POST /api/reload-upstreams → 立即生效
    async def _reload_upstreams_handler(request: web.Request) -> web.Response:
        ok, message = router.reload()
        return web.json_response(
            {"ok": ok, "message": message},
            status=200 if ok else 400,
        )

    app.router.add_post("/api/reload-upstreams", _reload_upstreams_handler)

    # 运行时重载模式策略配置（policies.yaml）：原子替换，失败保持旧策略继续服务。
    async def _reload_policies_handler(request: web.Request) -> web.Response:
        ok, message = policies.reload()
        return web.json_response(
            {"ok": ok, "message": message},
            status=200 if ok else 400,
        )

    app.router.add_post("/api/reload-policies", _reload_policies_handler)

    # 通用 catch-all：未知端点（Search/Fetch 等新 API）透明转发 + 留痕。
    # 必须最后注册 —— 已注册路由（含 /anthropic/* 路径保护）优先匹配。
    app.router.add_route("*", "/{tail:.*}", catchall_handler)

    # 周期 checkpoint 任务：checkpoint 即 AOF 压缩点。
    # K1：先提交新投影（含 offset_before）再截断日志（archive）—— 崩溃任意时刻，
    # 日志完整 + 投影 + 重放 或 最新投影 + 空日志，至少一方能重建内存，不静默丢状态。
    async def _checkpoint_loop():
        while True:
            await asyncio.sleep(config.checkpoint_interval)
            try:
                # 1. flush 事件 → 记录快照点偏移
                events_writer.flush_sync()
                offset_before = events_writer.get_offset()
                # 满足 create_checkpoint 并发契约：先排空 SQLite 写队列，降 ghost row 风险
                await store.flush()
                # I1：事件循环线程内取内存快照（同步无 await → 与 handler 无迭代竞态）
                snap = store.snapshot()
                # 2. 投影 + 记录 offset（worker 线程内取 thread-local 连接）
                await asyncio.to_thread(_checkpoint_worker, snap, offset_before)
                # 3. 新 base 偏移归 0（先于 archive：闭合"archive 后 reset 前崩溃→meta 指向
                # 已删旧段→tail 丢失+损坏态"窗口。meta=0 时崩溃 → 从 0 幂等重放，安全）
                await asyncio.to_thread(_reset_offset_worker)
                # 4. 截断旧日志（投影已提交；keep_from_offset 保留快照后新 append 的尾部）
                events_writer.archive_old_segments(offset_before)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("periodic checkpoint failed: %s", exc)

    app["_checkpoint_task"] = asyncio.create_task(_checkpoint_loop())

    async def _shutdown(app: web.Application) -> None:
        """Ordered shutdown: 事件日志 finalize → flush MemoryStore → close resources.

        K2：flush_pending（排空 + 停 sqlite-writer）在 checkpoint 前 → checkpoint 与
        写者无竞争，无 ghost row（残留 session_persist 不会在投影后二次累加）。
        K1：checkpoint 在 archive 前 → 投影先提交，日志后截断，崩溃安全。
        """
        # 取消周期 checkpoint 任务（避免与最终 checkpoint 竞争 / 泄漏后台任务）
        ctask = app.get("_checkpoint_task")
        if ctask is not None:
            ctask.cancel()
            try:
                await ctask
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception:
                pass
        if store:
            # 事件日志 finalize + 最终 checkpoint（K1/K2 顺序）
            if events_writer:
                try:
                    # 1. flush 事件 → 记录快照点偏移
                    events_writer.flush_sync()
                    offset_before = events_writer.get_offset()
                    # 2. K2：先排空 + 停 sqlite-writer（checkpoint 不与写者竞争）
                    await store.flush_pending()
                    # I1：事件循环线程内取内存快照
                    snap = store.snapshot()
                    # 3. 投影 + 记录 offset
                    await asyncio.to_thread(_checkpoint_worker, snap, offset_before)
                    # 4. 新 base 偏移归 0（先于 archive：闭合损坏态窗口）
                    await asyncio.to_thread(_reset_offset_worker)
                    # 5. 截断旧日志（keep_from_offset 保留尾部）
                    events_writer.archive_old_segments(offset_before)
                except Exception as exc:
                    log.warning("shutdown checkpoint failed: %s", exc)
                finally:
                    events_writer.close()
            else:
                # 无事件日志路径：仅排空 SQLite 写队列
                await store.flush_pending()
        await _close_writer(writer)
        await _close_otlp_log(otlp_writer)
        await _close_session(client_session)
        await _close_db(db)

    def _checkpoint_worker(snap, offset):
        """worker 线程内取 thread-local SQLite 连接并投影（避免跨线程用 sqlite3 连接）。"""
        conn = db.conn
        store.create_checkpoint(conn, checkpoint_offset=offset, snapshot=snap)

    def _reset_offset_worker():
        """checkpoint_offset 归 0（先于 archive：meta=0 崩溃 → 从 0 幂等重放或读 tail 恢复）。"""
        conn = db.conn
        set_meta(conn, "checkpoint_offset", "0")

    app.on_shutdown.append(_shutdown)

    log.info(f"Trace file: {writer.path}")
    log.info(f"Database: {config.db_path}")
    log.info(f"Database backend: {config.db_type} ({config.db_path or config.db_dsn})")
    log.info(f"Listening on {config.host}:{config.port}")
    log.info(f"Upstream: {config.target}")
    if router.is_available:
        log.info(f"Upstream router: {config.upstreams_config}")
    else:
        log.info("Upstream router: disabled (no config)")

    return app


async def _resolve_upstream_for_request(request: web.Request):
    """Resolve upstream for GET passthrough requests (no pipeline)."""
    from deepseek_tap_proxy.router import ResolvedTarget
    rt = request.get("runtime")
    router = rt.router if rt else None
    if router is not None and router.is_available:
        resolved = router.resolve(request.headers.get("Authorization"))
        if resolved is not None:
            return resolved
    return ResolvedTarget(
        target_url=getattr(rt.config, "target", "") if rt and rt.config else "",
        upstream_name="default",
        matched_key_prefix=None,
    )


async def _proxy_get_handler(request: web.Request) -> web.Response:
    rt = request.get("runtime")
    session: aiohttp.ClientSession = rt.session if rt else request.app.get("proxy_ctx", {}).get("session")

    # Read resolved upstream from request (set by wrapper)
    upstream = request.get("upstream")
    if upstream is not None:
        target = upstream.target_url
    else:
        target = getattr(rt.config, "target", "") if rt and rt.config else ""

    log.debug(f"[upstream] GET {request.path} → {upstream.upstream_name if upstream else 'default'}")

    fwd_headers = dict(request.headers)
    fwd_headers.pop("Host", None)
    fwd_headers = filter_hop_by_hop(fwd_headers)
    path_prefix = upstream.path_prefix if upstream is not None and upstream.path_prefix else ""
    if path_prefix and not request.path.startswith(path_prefix):
        upstream_url = target.rstrip("/") + path_prefix + request.path
    else:
        upstream_url = target.rstrip("/") + request.path

    try:
        upstream_resp = await session.request(
            method="GET", url=upstream_url, headers=fwd_headers,
            timeout=aiohttp.ClientTimeout(total=30),
        )
    except Exception as exc:
        log.error(f"[proxy-get] upstream error for {request.path}: {exc}")
        return web.Response(status=502, text=str(exc))

    body = await upstream_resp.read()
    return web.Response(
        status=upstream_resp.status,
        headers={
            k: v for k, v in upstream_resp.headers.items()
            if k.lower() not in HOP_BY_HOP
        },
        body=body,
    )


def _read_rss_bytes() -> int | None:
    """进程 RSS（Linux: /proc/self/status VmRSS；非 Linux 平台 None——观测不跨平台报错）。"""
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


async def _health_handler(request: web.Request) -> web.Response:
    rt = request.get("runtime")
    store = rt.store if rt else request.app.get("store")
    result = {"status": "ok", "service": "proxy"}
    # 事件日志（权威事实源）健康：写失败计数接入健康状态 —— 权威日志丢事件
    # 比 SQLite 投影落后更严重。failures 为"连续失败批数"（成功归零，瞬态
    # 自愈），达阈值才 degraded，避免单次磁盘抖动永久误报。
    events_writer = request.app.get("events_writer")
    if events_writer is not None:
        result["events_writer"] = {
            "failures": events_writer.failures,
            "path": events_writer.path,
        }
        if events_writer.failures >= _EVENTS_WRITER_FAILURE_THRESHOLD:
            result["status"] = "degraded"
    if store:
        stats = store.get_write_stats()
        result["write_stats"] = stats
        if stats["writes_failed"] > 10:
            result["status"] = "degraded"
        if stats["writes_attempted"] > 100 and stats["failure_rate"] > 0.05:
            result["status"] = "degraded"

        # 内存统计
        mem_stats = store.get_memory_stats()
        result["memory"] = {
            "estimated_bytes": mem_stats["estimated_bytes"],
            # N4：进程真实 RSS（/proc/self/status）——估算口径再准也只是 store 面，
            # 长期运行的总水位必须以 RSS 为准（估算值可能显著低于真实值）
            "rss_bytes": _read_rss_bytes(),
            "max_bytes": mem_stats["max_bytes"],
            "usage_pct": mem_stats["usage_pct"],
            "sessions": mem_stats["sessions"],
            "chains": mem_stats["chains"],
        }
        result["eviction"] = {
            "stale_evictions_total": mem_stats["stale_evictions_total"],
            "lru_evictions_total": mem_stats["lru_evictions_total"],
            "moments_truncated_total": mem_stats["moments_truncated_total"],
            "last_eviction_check": mem_stats["last_eviction_check"],
        }
        from deepseek_tap_proxy.shared.log import get_truncated_count
        result["trace"] = {"resp_truncated_total": get_truncated_count()}
    return web.json_response(result)


async def _close_writer(writer: TraceWriter) -> None:
    await writer.close()


async def _close_otlp_log(otlp_writer: OtlpLogWriter | None) -> None:
    if otlp_writer is not None:
        otlp_writer.close()


async def _close_session(session: aiohttp.ClientSession) -> None:
    await session.close()


async def _close_db(db) -> None:
    db.close()
