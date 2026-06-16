"""RequestAnalysisStage — 统一请求分类：client_type / chat_type / session_id。

在 CanonicalStage 之后、SysToolsLockStage 之前执行。
操作 canonical view (``ctx.sys`` / ``ctx.messages``)。
"""
from __future__ import annotations

import re

from deepseek_tap_proxy.pipeline import Stage, Context


class RequestAnalysisStage(Stage):
    """统一请求分析：产出 client_type / chat_type / session_id。

    产出（强类型 Context 字段）：
    - ``ctx.client_type``  — ``"copilot"`` | ``"copilot-agent"`` | ``"hermes"`` | ``"claude-code"`` | ``"generic"``
    - ``ctx.chat_type``    — ``"normal"`` | ``"compaction"`` | ``"subagent"``
    - ``ctx.session_id``   — 会话标识符字符串

    内部顺序：
    1. 检测 client_type（Anthropic: headers; OpenAI: canonical view）
    2. 检测 chat_type（基于 client_type + messages）
    3. 解析 session_id（UUID → hash chain → 合成）
    """

    # ── Copilot 特征常量 ──
    COPILOT_NORMAL_PREFIX = "You are an expert AI programming assistant"
    COMPACTION_SIGNATURE = "Your task is to create a comprehensive, detailed summary"

    _SESSION_UUID_RE = re.compile(
        r'GitHub\.copilot-chat[\\/](?:debug-logs|transcripts)[\\/]'
        r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})'
    )

    _HERMES_SESSION_ID_RE = re.compile(
        r'(?:^|\n)\s*(?:Current )?Session\s+ID[:\s]+(\S+)', re.IGNORECASE
    )

    async def process(self, ctx: Context) -> Context:
        if ctx.format == "anthropic":
            self._resolve_anthropic(ctx)
        elif ctx.format == "openai":
            self._resolve_openai(ctx)
        self._sync_persist(ctx)
        return ctx

    # ── Anthropic ──────────────────────────────────────────────────────

    def _resolve_anthropic(self, ctx: Context) -> None:
        headers = {k.lower(): v for k, v in ctx.request.headers.items()}
        sid = headers.get("x-claude-code-session-id", "")
        if sid:
            ctx.client_type = "claude-code"
            ctx.session_id = sid
            ctx.session_method = "uuid"
            self._upsert_session_db(ctx, sid)
        else:
            ctx.client_type = "generic"
            ctx.session_id = ""
            ctx.session_method = "create"

        # 检测 auto-title 生成请求
        from deepseek_tap_proxy.anthropic.persist import _detect_title_gen
        if _detect_title_gen(ctx.body_readonly):
            ctx.chat_type = "auto_title"
        else:
            ctx.chat_type = "normal"

    # ── OpenAI ─────────────────────────────────────────────────────────

    def _resolve_openai(self, ctx: Context) -> None:
        sys, msgs = ctx.sys, ctx.messages
        ctx.client_type = self._detect_client_type(sys, msgs)
        ctx.chat_type = self._detect_chat_type(ctx.client_type, sys, msgs)
        ctx.session_id = self._resolve_session_id(ctx)

    # ── client_type 检测（OpenAI，基于 canonical view）─────────────────

    def _detect_client_type(self, sys: list[dict], msgs: list[dict]) -> str:
        # ── Sys 模式：system prompt 在 role=system 的消息中 → Agent Copilot ──
        if sys:
            s0 = sys[0].get("content", "")
            if isinstance(s0, str):
                if "Session ID:" in s0 or "# Hermes" in s0:
                    return "hermes"
                # Standard Copilot 不会把 system prompt 放 sys（它用 role=user），
                # 所以 sys 里有 Copilot 前缀的一定是 Agent Copilot
                if s0.startswith(self.COPILOT_NORMAL_PREFIX) or \
                   s0.startswith(self.COMPACTION_SIGNATURE):
                    return "copilot-agent"
                return "copilot-agent"  # non-empty sys, not Hermes

        if not msgs:
            return "generic"

        m0 = msgs[0].get("content", "")
        if not isinstance(m0, str):
            return "generic"

        # ── Msgs 模式：system prompt 在 role=user 的消息中 → 标准 Copilot ──
        # <agents> 标签用于 chat_type 判定（normal/subagent），不用于 client_type
        if m0.startswith(self.COPILOT_NORMAL_PREFIX):
            return "copilot"
        if m0.startswith(self.COMPACTION_SIGNATURE):
            return "copilot"
        if m0.startswith("---\n") or m0.startswith("---\r\n"):
            return "copilot"
        if self._SESSION_UUID_RE.search(m0):
            return "copilot"
        if len(msgs) > 1:
            m1 = msgs[1].get("content", "")
            if isinstance(m1, str) and "<environment_info>" in m1:
                return "copilot"

        return "generic"

    # ── chat_type 检测 ────────────────────────────────────────────────

    def _detect_chat_type(self, client_type: str, sys: list[dict], msgs: list[dict]) -> str:
        """Copilot: Normal → Compaction → SubAgent
        Agent Copilot: Agent → Compaction → SubAgent
        """
        # Compaction 优先（所有 client_type 公共）
        for source in (sys, msgs):
            if source:
                s0 = source[0].get("content", "")
                if isinstance(s0, str) and s0.startswith(self.COMPACTION_SIGNATURE):
                    return "compaction"

        if client_type == "copilot":
            # Standard Copilot Chat — msgs[0] 有系统 prompt
            if msgs:
                m0 = msgs[0].get("content", "")
                if isinstance(m0, str):
                    if m0.startswith(self.COPILOT_NORMAL_PREFIX):
                        return "normal" if "<agents>" in m0 else "subagent"
                    if self._SESSION_UUID_RE.search(m0):
                        return "normal" if "<agents>" in m0 else "subagent"
            return "subagent"

        if client_type == "copilot-agent":
            # Agent Copilot — 有 <agents> → 主对话；无 → 子代理
            s0 = (sys[0] if sys else msgs[0]).get("content", "")
            return "normal" if "<agents>" in s0 else "subagent"

        # Hermes / Claude Code / Generic
        return "normal"

    # ── session_id 解析 ───────────────────────────────────────────────

    def _resolve_session_id(self, ctx: Context) -> str:
        uuid = self._extract_uuid(ctx.client_type, ctx.sys, ctx.messages)
        if uuid:
            ctx.session_method = "uuid"
            self._upsert_session_db(ctx, uuid)
            return uuid

        sid, method = self._match_or_create(ctx)
        if sid:
            ctx.session_method = method
            return sid

        ctx.session_method = "create"
        return self._synthesize_session_id(ctx)

    def _extract_uuid(self, client_type: str, sys: list[dict], msgs: list[dict]) -> str | None:
        if client_type in ("copilot", "copilot-agent"):
            candidates = [m.get("content", "") for m in msgs if isinstance(m.get("content"), str)]
            if sys:
                s0 = sys[0].get("content", "")
                if isinstance(s0, str):
                    candidates.insert(0, s0)
            for text in candidates:
                m = self._SESSION_UUID_RE.search(text)
                if m:
                    return m.group(1)
        elif client_type == "hermes":
            if sys:
                s0 = sys[0].get("content", "")
                if isinstance(s0, str):
                    m = self._HERMES_SESSION_ID_RE.search(s0)
                    if m:
                        return m.group(1)
        return None

    def _match_or_create(self, ctx: Context) -> tuple[str, str] | tuple[None, None]:
        from deepseek_tap_proxy.shared.analysis import is_same_session
        from deepseek_tap_proxy.shared.store import MemoryStore

        rt = ctx.request.get("runtime")
        store: MemoryStore | None = rt.store if rt else None
        curr_hashes = ctx.messages_hashes
        threshold = 3 if ctx.orig_msgs_len > 5 else 2

        if store:
            # Compaction 请求无 UUID，需放宽 client_type 过滤（双方向）
            _client_types = [ctx.client_type]
            if ctx.chat_type == "compaction":
                _client_types = ["copilot", "copilot-agent"]
            candidates = []
            for ct in _client_types:
                candidates.extend(store.get_candidate_sessions(ct))
            for candidate in candidates:
                max_chain = (
                    candidate["max_hash_chain"].split(",")
                    if candidate["max_hash_chain"] else []
                )
                if is_same_session(curr_hashes, max_chain, threshold):
                    if ctx.chat_type != "compaction":
                        store.update_session_hash_chain(candidate["session_id"], curr_hashes)
                    return candidate["session_id"], "match"

        new_id = self._synthesize_session_id(ctx)
        if store:
            store.upsert_session(new_id, ctx.client_type, 0, curr_hashes)
        return new_id, "create"

    def _synthesize_session_id(self, ctx: Context) -> str:
        import xxhash
        hashes = ctx.messages_hashes
        raw = "|".join(hashes[:4]) if hashes else ctx.msg_hash_chain[:32]
        return f"syn_{xxhash.xxh3_64(raw.encode()).hexdigest()[:16]}"

    # ── DB 辅助 ───────────────────────────────────────────────────────

    def _upsert_session_db(self, ctx: Context, session_id: str) -> None:
        from deepseek_tap_proxy.shared.store import MemoryStore
        rt = ctx.request.get("runtime")
        store: MemoryStore | None = rt.store if rt else None
        if store:
            store.upsert_session(session_id, ctx.client_type, 0, ctx.messages_hashes)

    def _sync_persist(self, ctx: Context) -> None:
        ctx.persist.session_id = ctx.session_id
        ctx.persist.client_type = ctx.client_type
        ctx.persist.chat_type = ctx.chat_type
        ctx.persist.session_method = ctx.session_method
        ctx.persist.msg_hash_chain = ctx.msg_hash_chain


# ── Backward compatibility aliases ──────────────────────────────────────

SessionResolverStage = RequestAnalysisStage
RequestClassifierStage = RequestAnalysisStage
