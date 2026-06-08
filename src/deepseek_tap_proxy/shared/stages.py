"""Format-agnostic pipeline stages.

All stages here operate on the canonical view (``ctx.sys`` / ``ctx.tools`` /
``ctx.messages``) — they work identically for OpenAI and Anthropic requests.
"""

from __future__ import annotations

import asyncio
import json
import logging

from deepseek_tap_proxy.pipeline import Stage, Context
from deepseek_tap_proxy.shared.db import get_session_chains

log = logging.getLogger("deepseek-tap")


# ═══════════════════════════════════════════════════════════════════
#  PrefixLockStage
# ═══════════════════════════════════════════════════════════════════

class PrefixLockStage(Stage):
    """Prefix lock: match session's prefix chains against canonical form.

    Operates on the complete canonical view ``ctx.sys + ctx.tools +
    ctx.messages`` — the token sequence the model actually sees.

    When a chain matches, ``sys`` and ``tools`` are locked to the
    prefix values (they are part of the KV-cached prefix), and
    ``messages`` is extended with any new conversation messages
    beyond the prefix boundary.

    Works identically for OpenAI and Anthropic — the canonical
    view properties handle the format mapping transparently.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    async def process(self, ctx: Context) -> Context:
        if not self.enabled:
            return ctx

        # 确保 hash chain 已计算（可能被 CanonicalStage 延迟了）
        _ensure_msg_hash_chain(ctx)
        if not ctx.persist.msg_hash_chain:
            return ctx

        session = ctx.meta.get("session")
        if not session:
            return ctx

        # Full canonical form: sys + tools + messages
        all_msgs_c = ctx.sys + ctx.tools + ctx.messages
        if not all_msgs_c:
            return ctx

        db = ctx.request.app.get("db")
        if not db:
            return ctx

        chains = await asyncio.to_thread(
            lambda: get_session_chains(db.conn, session.key)
        )
        curr_hashes = ctx.persist.msg_hash_chain.split(",")
        if not chains:
            # No existing chains → create a new one
            from deepseek_tap_proxy.shared.prefix import make_chain_id, compute_safe_breakpoints
            from deepseek_tap_proxy.pipeline import ChainAction

            bp = compute_safe_breakpoints(all_msgs_c)
            cc = _compute_char_count(all_msgs_c)
            ctx.chain_action = ChainAction(
                action="create",
                chain_id=make_chain_id(curr_hashes),
                msg_count=len(all_msgs_c),
                hash_chain=ctx.persist.msg_hash_chain,
                sys=ctx.sys,
                tools=ctx.tools,
                messages=ctx.messages,
                safe_breakpoints=bp,
                hit_count=0,
                chain_tokens=0,
                char_count=cc,
            )
            ctx.persist.chain_id = ctx.chain_action.chain_id
            ctx.persist.chain_action = ctx.chain_action.action
            ctx.persist.chain_msg_count = ctx.chain_action.msg_count
            ctx.persist.chain_tokens = 0
            return ctx

        from deepseek_tap_proxy.shared.match import match_prefix

        # 计算 curr 中所有 role=user 的位置索引
        _user_positions = [
            i for i, m in enumerate(ctx.messages)
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        # user 的索引需要加上 sys+tools 的偏移（context = sys + tools + msg）
        n_sys = len(ctx.sys)
        n_tools = len(ctx.tools)
        user_positions = [i + n_sys + n_tools for i in _user_positions]

        # 匹配：compaction 禁用 user 锚点（旧对话历史，无新用户输入）
        _disable_ua = ctx.meta.get("intent") == "compaction"
        match_result = match_prefix(chains, curr_hashes, user_positions,
                                    disable_user_anchor=_disable_ua)
        if match_result is None:
            # 无匹配 → CREATE 新链
            from deepseek_tap_proxy.shared.prefix import make_chain_id, compute_safe_breakpoints
            from deepseek_tap_proxy.pipeline import ChainAction

            bp = compute_safe_breakpoints(all_msgs_c)
            ctx.chain_action = ChainAction(
                action="create",
                chain_id=make_chain_id(curr_hashes),
                msg_count=len(all_msgs_c),
                hash_chain=ctx.persist.msg_hash_chain,
                sys=ctx.sys,
                tools=ctx.tools,
                messages=ctx.messages,
                safe_breakpoints=bp,
                hit_count=0,
                chain_tokens=0,
            )
            ctx.persist.chain_id = ctx.chain_action.chain_id
            ctx.persist.chain_action = ctx.chain_action.action
            ctx.persist.chain_msg_count = ctx.chain_action.msg_count
            ctx.persist.chain_tokens = 0
            return ctx

        best_chain_id = match_result.chain_id
        n = match_result.n
        lock_type = match_result.lock_type

        # 找到完整的链记录
        best_chain = next(c for c in chains if c["chain_id"] == best_chain_id)
        ctx.persist.prefix_chain_id = best_chain["chain_id"]
        ctx.meta["prefix_chain_msg_count"] = best_chain["msg_count"]

        # Load prefix from canonical three-column format
        sys_p    = _json_list(best_chain.get("sys"))
        tools_p  = _json_list(best_chain.get("tools"))
        msg_p    = _json_list(best_chain.get("messages"))
        all_msgs_p = sys_p + tools_p + msg_p

        # 机械构造锁定结果
        # n 是链索引，锁定 = chain[:n+1] + curr[n+1:]
        locked_msgs = all_msgs_p[:n+1] + all_msgs_c[n+1:]

        # 防御性验证 + n 补正
        from deepseek_tap_proxy.shared.prefix import (
            compute_safe_breakpoints, find_safe_truncation,
            validate_tool_call_chain,
        )
        n_original = n
        if not validate_tool_call_chain(locked_msgs):
            # 罕见路径：n 意外打断了 tool_call 对，往回调到安全断点
            safe_bps = None
            if best_chain.get("safe_breakpoints"):
                try:
                    safe_bps = json.loads(best_chain["safe_breakpoints"])
                except (json.JSONDecodeError, TypeError):
                    safe_bps = None
            if safe_bps is None:
                safe_bps = compute_safe_breakpoints(all_msgs_p)
            n = find_safe_truncation(safe_bps, n)
            locked_msgs = all_msgs_p[:n+1] + all_msgs_c[n+1:]
            log.warning(
                f"[{ctx.req_id}] n corrected for tool_call boundary: "
                f"{n_original}→{n}"
            )

        # preserve upstream modifications (e.g. SysToolsLock already set is_modified)
        if lock_type != "layer1_perfect":
            ctx.persist.is_modified = True
        ctx.meta["log_msgs_after"] = len(locked_msgs)
        ctx.meta["pref_hit_count"] = n + 1  # n 是 0-based 索引，pref_hit 是数量

        # est 数据：查询匹配长度对应的 token 记录
        pref_hit = ctx.meta["pref_hit_count"]
        from deepseek_tap_proxy.shared.db import get_chain_tokens_and_chars, get_chain_tokens_ge
        chain_tc = await asyncio.to_thread(
            lambda: get_chain_tokens_and_chars(db.conn, best_chain["chain_id"], pref_hit)
        )
        if chain_tc is None:
            # pref_hit 不在已有 token_history 中 → 创建公共前缀链
            chain_tc_ge = await asyncio.to_thread(
                lambda: get_chain_tokens_ge(db.conn, best_chain["chain_id"], pref_hit)
            )
            if chain_tc_ge is not None:
                ge_tokens, ge_chars, ge_msg_count = chain_tc_ge
                # 估算：pref_hit 元素的总 token
                # = 最近 ≥pref_hit 的记录 token - 差值元素的估算 token
                overflow = ge_msg_count - pref_hit
                if overflow > 0 and ge_chars > 0:
                    overflow_chars = sum(
                        len(json.dumps(all_msgs_p[pref_hit + i], ensure_ascii=False))
                        for i in range(min(overflow, len(all_msgs_p) - pref_hit))
                    )
                    tpc = ge_tokens / ge_chars
                    prefix_tokens_est = int(ge_tokens - overflow_chars * tpc)
                else:
                    prefix_tokens_est = ge_tokens

                # 创建公共前缀链（冻结）
                from deepseek_tap_proxy.shared.hash import compute_canonical_hash_chain
                from deepseek_tap_proxy.shared.prefix import make_chain_id
                from deepseek_tap_proxy.shared.db import upsert_prefix_chain, insert_chain_token
                from deepseek_tap_proxy.shared.colors import green

                _prefix_context = all_msgs_p[:pref_hit]
                _prefix_hashes = best_chain["hash_chain"].split(",")[:pref_hit]
                _prefix_hash_str = ",".join(_prefix_hashes)
                _prefix_chain_id = make_chain_id(_prefix_hashes)

                _prefix_sys = _prefix_context[:n_sys]
                _prefix_tools = _prefix_context[n_sys:n_sys + n_tools]
                _prefix_msgs = _prefix_context[n_sys + n_tools:]
                _prefix_cc = _compute_char_count(_prefix_context)

                await asyncio.to_thread(
                    lambda: upsert_prefix_chain(
                        db.conn, session.key, _prefix_chain_id,
                        sys=json.dumps(_prefix_sys),
                        tools=json.dumps(_prefix_tools),
                        messages=json.dumps(_prefix_msgs),
                        hash_chain=_prefix_hash_str,
                        msg_count=pref_hit,
                        safe_breakpoints=compute_safe_breakpoints(_prefix_context),
                        is_frozen=True,
                    ))
                await asyncio.to_thread(
                    lambda: insert_chain_token(
                        db.conn, _prefix_chain_id,
                        msg_count=pref_hit,
                        tokens=prefix_tokens_est,
                        char_count=_prefix_cc,
                    ))
                print(green(f"  [chain] + prefix: {_prefix_chain_id[:16]}... "
                            f"({pref_hit} msgs, frozen, est={prefix_tokens_est}t)"))

                chain_tokens = prefix_tokens_est
                chain_char_count = _prefix_cc
            else:
                chain_tokens = 0
                chain_char_count = 0
        else:
            chain_tokens = chain_tc[0]
            chain_char_count = chain_tc[1]
        ctx.meta["chain_tokens_est"] = chain_tokens
        ctx.meta["chain_char_count"] = chain_char_count
        new_msgs = all_msgs_c[pref_hit:]
        ctx.meta["new_char_count"] = _compute_char_count(new_msgs)

        # 拆分回规范形式三元组并写回 fwd_body
        ctx.sys      = locked_msgs[:n_sys]
        ctx.tools    = locked_msgs[n_sys:n_sys + n_tools]
        ctx.messages = locked_msgs[n_sys + n_tools:]

        ctx.persist.lock_type = lock_type
        correction_note = ""
        if n != n_original:
            correction_note = f" n_corrected: {n_original}→{n}"
        ctx.persist.modification_note = (
            f"{lock_type}: "
            f"n={n}, prefix={len(all_msgs_p)}msgs, curr={len(all_msgs_c)}msgs, "
            f"locked_to={len(locked_msgs)}msgs"
            + correction_note
        )

        from deepseek_tap_proxy.shared.hash import compute_canonical_hash_chain
        from deepseek_tap_proxy.shared.prefix import make_chain_id
        from deepseek_tap_proxy.pipeline import ChainAction

        full_hash = compute_canonical_hash_chain(ctx.sys, ctx.tools, ctx.messages)
        cc = _compute_char_count(locked_msgs)

        safe_bps = None
        if best_chain.get("safe_breakpoints"):
            try:
                safe_bps = json.loads(best_chain["safe_breakpoints"])
            except (json.JSONDecodeError, TypeError):
                safe_bps = None
        if safe_bps is None:
            safe_bps = compute_safe_breakpoints(all_msgs_p)

        # 链管理决策：只有严格比链长 且 链非冻结 才 UPDATE，否则 CREATE 新链
        _matched_is_frozen = bool(best_chain.get("is_frozen"))
        if len(all_msgs_c) > len(all_msgs_p) and not _matched_is_frozen:
            ctx.chain_action = ChainAction(
                action="update",
                chain_id=best_chain["chain_id"],
                msg_count=len(locked_msgs),
                hash_chain=full_hash,
                sys=ctx.sys,
                tools=ctx.tools,
                messages=ctx.messages,
                safe_breakpoints=safe_bps,
                hit_count=len(all_msgs_p),
                chain_tokens=chain_tokens,
                char_count=cc,
            )
        else:
            ctx.chain_action = ChainAction(
                action="create",
                chain_id=make_chain_id(curr_hashes),
                msg_count=len(locked_msgs),
                hash_chain=ctx.persist.msg_hash_chain,
                sys=ctx.sys,
                tools=ctx.tools,
                messages=ctx.messages,
                safe_breakpoints=compute_safe_breakpoints(all_msgs_c),
                hit_count=0,
                chain_tokens=0,
                char_count=_compute_char_count(all_msgs_c),
            )
        ctx.persist.chain_id = ctx.chain_action.chain_id
        ctx.persist.chain_action = ctx.chain_action.action
        ctx.persist.chain_msg_count = ctx.chain_action.msg_count
        if ctx.chain_action.action == "update":
            ctx.persist.chain_tokens = chain_tokens or 0
        return ctx


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _ensure_msg_hash_chain(ctx) -> str:
    """Ensure msg_hash_chain is computed (lazy). 返回 hash_chain 字符串。"""
    if not ctx.persist.msg_hash_chain:
        from deepseek_tap_proxy.shared.hash import compute_canonical_hash_chain
        ctx.persist.msg_hash_chain = compute_canonical_hash_chain(
            ctx.sys, ctx.tools, ctx.messages,
        )
    return ctx.persist.msg_hash_chain


def _compute_char_count(messages: list[dict]) -> int:
    """Compute total serialized character count for token estimation.

    Uses ``json.dumps`` (full message dict serialization) to capture all
    text — including tool definitions (``function.description``,
    ``parameters``) that lack a ``content`` field — so the
    ``tokens / chars`` ratio  is consistent with the inline
    ``overflow_chars`` calculation in ``PrefixLockStage``.
    """
    return sum(
        len(json.dumps(m, ensure_ascii=False))
        for m in messages
    )


def _json_list(raw) -> list[dict]:
    """Parse a JSON column into list[dict], returning [] on failure."""
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


# ═══════════════════════════════════════════════════════════════════
#  SysToolsLockStage
# ═══════════════════════════════════════════════════════════════════

class SysToolsLockStage(Stage):
    """Lock sys+tools to canonical values within a session.

    Ensures that the system prompt and tool definitions remain
    consistent across all requests in a session, preventing
    cache pollution from injected/modified content.

    Operates on the canonical ``ctx.sys`` and ``ctx.tools`` views.

    - ``init``: first time canonical is captured
    - ``lock``: locked (restored from canonical) because current != canonical
    - ``pass``: already matches canonical, no action

    ``lock_sys_tools`` 配置从 ``ctx.request.app["config"]`` 读取，
    不依赖构造参数（使得 PipelineFactory 预构造不会丢失配置）。
    """

    def __init__(self, enabled: bool = True) -> None:
        # enabled 参数保留向后兼容，但实际值从运行时配置读取
        self._default_enabled = enabled

    async def process(self, ctx: Context) -> Context:
        # 运行时读取配置，优先级高于构造参数
        config = ctx.request.app.get("config", {})
        lock_enabled = getattr(config, "lock_sys_tools", self._default_enabled)
        if not lock_enabled:
            return ctx

        # RequestClassifierStage 设此标记时跳过（如 Copilot 子代理请求）
        if ctx.meta.get("sys_tools_skip"):
            ctx.meta["sys_tools_locked"] = "skip"
            return ctx

        session = ctx.meta.get("session")
        if not session:
            return ctx

        db = ctx.request.app.get("db")
        if not db:
            return ctx

        def _sync_canonical(database, sk, cur_sys, cur_tools):
            """单次 to_thread 内完成查+写。

            接收 Database 实例而非 conn，因为 SQLite connection 绑创建线程。
            在 thread pool worker 内访问 db.conn 可自动创建新连接。

            返回 (action, restored_sys, restored_tools)：
            - "none": 无内容，什么都不做
            - "init": 首次写入 canonical
            - "lock": 需要恢复（cur_sys != canon_sys 等），restored_* 为基线值
            - "pass": 已匹配，无需操作
            """
            conn = database.conn  # 在 thread pool 内获取连接
            from deepseek_tap_proxy.shared.db import (
                get_session_canonical, set_session_canonical,
            )
            cur_sys_json = json.dumps(cur_sys, separators=(",", ":"), ensure_ascii=False)
            cur_tools_json = json.dumps(cur_tools, separators=(",", ":"), ensure_ascii=False)

            canon = get_session_canonical(conn, sk)
            has_canon_sys = bool(canon.get("sys"))
            has_canon_tools = bool(canon.get("tools"))

            # 无基线 → 首次写入
            if not has_canon_sys and not has_canon_tools:
                if not cur_sys and not cur_tools:
                    return ("none", [], [])
                new_sys = cur_sys_json if cur_sys else ""
                new_tools = cur_tools_json if cur_tools else ""
                set_session_canonical(conn, sk, sys=new_sys, tools=new_tools)
                return ("init", [], [])

            # 有基线 → 比较
            restored_sys = list(cur_sys)
            restored_tools = list(cur_tools)
            needs_lock = False

            if has_canon_sys:
                canon_sys = json.loads(canon["sys"])
                if cur_sys != canon_sys:
                    restored_sys = canon_sys
                    needs_lock = True

            if has_canon_tools:
                canon_tools = json.loads(canon["tools"])
                if cur_tools != canon_tools:
                    restored_tools = canon_tools
                    needs_lock = True

            if needs_lock:
                return ("lock", restored_sys, restored_tools)
            return ("pass", [], [])

        # 传 db 而非 db.conn：SQLite connection 绑创建线程，
        # _sync_canonical 内部在 thread pool 中访问 db.conn 自动创建新连接
        action, restored_sys, restored_tools = await asyncio.to_thread(
            _sync_canonical, db, session.key,
            ctx.sys, ctx.tools,
        )

        if action == "init":
            ctx.meta["sys_tools_locked"] = "init"
            return ctx
        if action == "none":
            return ctx
        if action == "pass":
            ctx.meta["sys_tools_locked"] = "pass"
            return ctx

        # action == "lock" — 恢复 sys/tools
        if restored_sys:
            ctx.sys = restored_sys
        if restored_tools:
            ctx.tools = restored_tools

        ctx.meta["sys_tools_locked"] = "lock"
        ctx.persist.is_modified = True
        ctx.persist.modification_note = "sys_tools_locked"

        # 重新计算 hash_chain（sys/tools 已变）
        from deepseek_tap_proxy.shared.hash import compute_canonical_hash_chain
        ctx.persist.msg_hash_chain = compute_canonical_hash_chain(
            ctx.sys, ctx.tools, ctx.messages,
        )
        # 修正三段长度
        ctx.meta["orig_sys_len"] = len(ctx.sys)
        ctx.meta["orig_tools_len"] = len(ctx.tools)
        ctx.meta["orig_msgs_len"] = len(ctx.messages)

        return ctx


# ═══════════════════════════════════════════════════════════════════
#  RequestClassifierStage — 通用请求分类器
#
#  在 CanonicalStage 之后、SysToolsLockStage 之前执行。
#  分析请求上下文，设置 ctx.meta 标记供后续 stage 决策使用。
#
#  当前分类：
#    Copilot type 下区分 normal / compaction / subagent
#    → 子代理请求设 sys_tools_skip = True，让 SysToolsLock 跳过
#
#  未来扩展：
#    ctx.meta["msg0_class"]         — msg[0] 类型分类
#    ctx.meta["tools_changed"]      — 工具集是否变化
#    ctx.meta["is_locked_agent"]    — 特定 agent 锁定规则
# ═══════════════════════════════════════════════════════════════════


class RequestClassifierStage(Stage):
    """分析请求上下文，分类请求类型，设置决策标记。

    当前检测的标记：
    - ``sys_tools_skip`` — 后续 stage 应跳过 sys/tools 相关处理
    """

    async def process(self, ctx: Context) -> Context:
        # 执行标记：用于测试验证 pipeline 是否包含此 stage
        ctx.meta["classifier_executed"] = True

        session = ctx.meta.get("session")
        if not session:
            return ctx

        client_type = getattr(session, "client_type", None)
        if client_type != "copilot":
            return ctx

        msgs = ctx.body_readonly.get("messages", []) if ctx.body_readonly else []
        if not msgs:
            return ctx

        c0 = msgs[0].get("content", "")
        if not isinstance(c0, str):
            return ctx

        # 已知的 Copilot 正常/压缩前缀（走 SysToolsLock）
        _KNOWN_NORMAL = (
            "You are an expert AI",
            "Your task is to create a",
            "GitHub.copilot-chat/debug-logs/",
        )
        is_normal_or_compaction = any(c0.startswith(p) for p in _KNOWN_NORMAL)
        has_frontmatter = c0.startswith("---\n") or c0.startswith("---\r\n")

        # 子代理判断条件（或）：
        # 1. 既不是默认对话也不是压缩（未知格式）
        # 2. 有 YAML frontmatter（---\nname: ...）
        if (not is_normal_or_compaction) or has_frontmatter:
            ctx.meta["sys_tools_skip"] = True

        return ctx
#  在 CanonicalStage 之前执行，为后续 stage 提供 session 上下文。
#  不依赖规范形式 view，只读 body_readonly 进行 session 匹配。
# ═══════════════════════════════════════════════════════════════════


class SessionResolverStage(Stage):
    """前置数据：解析 client-type 和 session。

    在 CanonicalStage 之前执行，为后续 stage 提供 session 上下文。
    不依赖规范形式 view，只读 body_readonly 进行 session 匹配。
    """

    async def process(self, ctx: Context) -> Context:
        if ctx.body_readonly is None:
            return ctx
        msgs = ctx.body_readonly.get("messages", [])
        db = ctx.request.app.get("db")

        from deepseek_tap_proxy.shared.analysis import (
            detect_intent_openai,
            resolve_session_openai,
        )
        session = resolve_session_openai(msgs, db=db)
        intent = detect_intent_openai(session.client_type, msgs) if db else "normal"
        ctx.meta["session"] = session
        ctx.meta["intent"] = intent
        ctx.persist.session_id = session.key
        ctx.persist.session_type = session.method
        ctx.persist.session_fingerprint = session.fingerprint or session.key
        return ctx


# ═══════════════════════════════════════════════════════════════════
#  层间隔离审计结果
#
#  确认以下 stage 通过 view 层操作，不直接访问 fwd_body：
#  - PrefixLockStage:     ctx.sys/tools/messages ✅
#  - CompactionRewrite:   ctx.messages ✅
#  - SessionResolverStage: ctx.body_readonly ✅（只读，未写 fwd_body）
#  - SystemRoleConvert:   ctx.messages ✅ (Phase 3 修复)
#
#  以下 stage 操作 format 特有的字段，属合理绕过：
#  - AnthropicNormalizeStage:     fwd_body["metadata"]
#  - AnthropicBlock0ClearStage:   fwd_body["system"]
#  - AnthropicCanonicalUserStage: fwd_body["metadata"]
#  - AnthropicBaselineStage:      fwd_body["system"/"tools"]
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
#  CanonicalStage — 规范形式激活
#
#  CanonicalStage 在 ParseStage 之后执行，负责：
#  1. 写入 msg_hash_chain（sys + tools + messages 的完整 hash）
#  2. 设置 orig_sys_len / orig_tools_len / orig_msgs_len
#  3. 设置 prompt_messages_count / prompt_tools_count
#  4. 计算 msg0_hash / msg0_content
#
#  ParseStage × 2 中重复的规范形式 meta 在此统一。
# ═══════════════════════════════════════════════════════════════════


class CanonicalStage(Stage):
    """激活规范形式 view，计算 hash_chain 和元数据。

    在 ParseStage 之后执行。读取 fwd_body + format，写入 ctx.meta：
      - msg_hash_chain — sys+tools+messages 的完整 hash 链
      - orig_sys_len / orig_tools_len / orig_msgs_len — 三段原始长度
      - prompt_messages_count — 总消息数（三段之和）
      - prompt_tools_count — 工具数
      - msg0_hash / msg0_content — system prompt 的 hash 和内容
      - log_msgs_before — log 用前置消息数

    Parameters
    ----------
    extensions : list[str] | None
        需要从 fwd_body 同名映射到 ctx.ext 的扩展 key 列表。
    """

    def __init__(self, extensions: list[str] | None = None):
        self.extensions = extensions or []

    async def process(self, ctx: Context) -> Context:
        if not ctx.body_readonly or not ctx.fwd_body:
            return ctx

        tools_count = len(ctx.tools)
        sys_len = len(ctx.sys)
        msgs_len = len(ctx.messages)
        total_msgs = sys_len + tools_count + msgs_len

        ctx.persist.prompt_messages_count = total_msgs
        ctx.persist.prompt_tools_count = tools_count
        ctx.meta["log_msgs_before"] = total_msgs

        # 三段原始长度（用于 log pref(stm)= 展示）
        ctx.meta["orig_sys_len"] = sys_len
        ctx.meta["orig_tools_len"] = tools_count
        ctx.meta["orig_msgs_len"] = msgs_len

        # hash chain — 延迟计算，不在热路径做遍历 hash
        # 各消费方（PrefixLock、persist、log）按需调用 _ensure_msg_hash_chain()

        # 扩展 view：同名映射 from fwd_body
        for name in self.extensions:
            ctx.ext[name] = ctx.fwd_body.get(name, {})

        return ctx


# ═══════════════════════════════════════════════════════════════════
#  Response cache analysis + log emitter — format-agnostic
#
#  These consume ctx.resp_usage / ctx.resp_model / ctx.meta only,
#  never raw format-specific structures.
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
#  响应端只读组合
#
#  以下函数由 ForwardStage 内部调用，读取 resp_usage 响应 view
#  (只读)，不修改请求或响应数据。它们不影响响应路径的延迟。
#
#  apply_cache_analysis → 从 resp_view 写入 ctx.meta
#  emit_request_log     → 从 ctx.meta 格式化输出
# ═══════════════════════════════════════════════════════════════════


def apply_cache_analysis(ctx: Context) -> None:
    """Read normalized usage from ``ctx.resp_usage`` and write to ``ctx.meta``.

    Call from any ``ForwardStage`` (OpenAI / Anthropic) after the
    upstream response is available.
    """
    usage = ctx.resp_usage
    ctx.persist.total_input_tokens = usage.get("total_input_tokens", 0)
    ctx.persist.output_tokens = usage.get("output_tokens", 0)
    ctx.persist.cache_hit_tokens = usage.get("cache_read_input_tokens", 0)
    ctx.persist.cache_miss_tokens = usage.get("cache_miss_input_tokens", 0)

    hit = ctx.persist.cache_hit_tokens
    miss = ctx.persist.cache_miss_tokens
    total = hit + miss
    ctx.persist.cache_hit_ratio = round(hit / total, 4) if total > 0 else 0.0

    ctx.persist.has_reasoning = ctx.resp_has_reasoning
    ctx.persist.has_tool_calls = ctx.resp_has_tool_calls

    ctx.persist.prompt_messages_count = len(ctx.sys) + len(ctx.messages)
    ctx.persist.prompt_tools_count = len(ctx.tools)


def emit_request_log(ctx: Context) -> None:
    """Emit one-line per-request log entry.

    Consumes ``ctx.meta`` and ``ctx`` properties only — format-agnostic.
    """
    from deepseek_tap_proxy.shared.colors import colorize, lock_symbol, blue, cyan

    status = ctx.status or 0
    duration_s = round((ctx.duration_ms or 0) / 1000, 1)
    lock_type = ctx.persist.lock_type or ctx.meta.get("log_lock_type", "none")
    lock_display = lock_symbol(lock_type)
    cache_hit = ctx.persist.cache_hit_tokens or ctx.meta.get("cache_hit_tokens", 0)
    cache_miss = ctx.persist.cache_miss_tokens or ctx.meta.get("cache_miss_tokens", 0)
    cache_total = cache_hit + cache_miss
    cache_pct = round(cache_hit / cache_total * 100) if cache_total > 0 else 0
    model = ctx.resp_model or "?"
    upstream_name = ctx.meta.get("upstream_name", "default")
    session = ctx.meta.get("session")
    client = session.client_type if session else "?"
    sid = session.key[:12] if session else "?"
    smethod = session.method if session else "?"
    intent = ctx.meta.get("intent", "normal")

    # ── sys_tools_locked status ──
    sys_lock = ctx.meta.get("sys_tools_locked", "")
    sys_lock_str = f" slock={sys_lock}" if sys_lock else ""

    # ── pref(stm)= s_hit/s_total t_hit/t_total m_hit/m_total ──
    # 三段视为一条线性链：[sys][tools][messages]
    # pref_hit 是这条链上的断点位置。
    # 着色规则：断点前=绿，断点处=黄，断点后=红。
    # (stm)= 不染色。
    pref_hit = ctx.meta.get("pref_hit_count", 0)
    s_total = ctx.meta.get("orig_sys_len", 0)
    t_total = ctx.meta.get("orig_tools_len", 0)
    m_total = ctx.meta.get("orig_msgs_len", 0)
    canon_total = s_total + t_total + m_total

    _rem = pref_hit
    s_hit = min(_rem, s_total)
    _rem -= s_hit
    t_hit = min(_rem, t_total)
    _rem -= t_hit
    m_hit = _rem

    def _seg_color(hit: int, total: int) -> str:
        """线性链断点位置着色。

        total=0 → "0/0" 不染色
        hit=total(>0) → 断点前，绿色
        hit=0(且total>0) → 断点后，红色
        0<hit<total → 断点在此段内，黄色
        """
        if total == 0:
            return "0/0"
        pct = int(hit / total * 100)
        return colorize(f"{hit}/{total}", "pref", pct)

    s_str = _seg_color(s_hit, s_total)
    t_str = _seg_color(t_hit, t_total)
    m_str = _seg_color(m_hit, m_total)
    pref_str = f"(stm)= {s_str} {t_str} {m_str}"

    # ── est=hit/total(pct%) 基于链 token+char 记录的预估命中 ──
    # est = chain_tokens / (chain_tokens + new_tokens)
    # new_tokens = new_char_count × (chain_tokens / chain_char_count)
    chain_tokens = ctx.meta.get("chain_tokens_est", 0) or ctx.persist.chain_tokens
    chain_cc = ctx.meta.get("chain_char_count", 0)
    new_cc = ctx.meta.get("new_char_count", 0)
    if chain_tokens and chain_cc and chain_tokens > 0:
        avg_tpc = chain_tokens / chain_cc          # tokens per char
        new_tokens_est = int(new_cc * avg_tpc)     # new_cc=0 → 0（compaction 后无新增消息）
        est_denom = chain_tokens + new_tokens_est
        est_pct = round(chain_tokens / est_denom * 100) if est_denom else 0
        est_str = f"{chain_tokens}/{est_denom}({est_pct}%)"
    else:
        est_str = "?"
        est_pct = 0

    log.info(
        f"[{ctx.req_id}] ← "
        f"{colorize(str(status), 'status', status)} "
        f"{colorize(f'{duration_s}s', 'duration', duration_s)} "
        f"pref{pref_str} "
        f"lock={colorize(lock_display, 'lock', lock_type)}"
        f"{sys_lock_str} "
        f"cache={cache_hit}/{cache_total}"
        f"({colorize(f'{cache_pct}%', 'cache_pct', cache_pct)}) "
        f"est={colorize(est_str, 'cache_pct', est_pct)} "
        f"{upstream_name} {model}"
        f" sess={blue(sid)}/{smethod}[{cyan(client)}]"
        + (f" [{intent}]" if intent != "normal" else "")
    )
