"""Copilot compaction optimization stages."""
from __future__ import annotations

import logging
from pathlib import Path

from deepseek_tap_proxy.pipeline import Stage, Context

log = logging.getLogger("deepseek-tap")

_PROMPT_PATH = Path(__file__).parent / "prompts" / "summarization.txt"
SUMMARIZATION_USER_MSG = _PROMPT_PATH.read_text(encoding="utf-8")


class CompactionRewrite(Stage):
    """Convert foreground compaction to background format.

    Works on ``ctx.fwd_body`` (the working copy forwarded upstream).
    Replaces the final user message with SUMMARIZATION_USER_MSG.
    Applies to all Copilot-derived client_types (copilot, copilot-agent).
    """

    async def process(self, ctx: Context) -> Context:
        if ctx.chat_type != "compaction":
            return ctx
        if not ctx.fwd_body or "messages" not in ctx.fwd_body:
            return ctx

        log.info("[%s] rewriting compaction request to background format", ctx.req_id)
        msgs = list(ctx.fwd_body["messages"])
        msgs[-1] = {"role": "user", "content": SUMMARIZATION_USER_MSG}
        ctx.fwd_body["messages"] = msgs
        ctx.persist.is_modified = True
        ctx.meta["is_compaction_rewritten"] = True
        ctx.meta["modification_note"] = "compaction: foreground->background"
        return ctx
