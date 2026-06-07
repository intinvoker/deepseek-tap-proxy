"""Copilot compaction optimization stages."""
from __future__ import annotations

import logging
from pathlib import Path

from deepseek_tap_proxy.pipeline import Stage, Context
from deepseek_tap_proxy.shared.stages import SessionResolverStage

log = logging.getLogger("deepseek-tap")

_PROMPT_PATH = Path(__file__).parent / "prompts" / "summarization.txt"
SUMMARIZATION_USER_MSG = _PROMPT_PATH.read_text(encoding="utf-8")

# Backward compatibility alias — RequestAnalyzer moved to shared/stages.py
# as SessionResolverStage in Phase 5 normalization.
RequestAnalyzer = SessionResolverStage


class CompactionRewrite(Stage):
    """Append the summarization prompt for background compaction.

    Only activates when ``ctx.meta['intent'] == 'compaction'``.
    The compaction instruction at ``ctx.sys[0]`` is the original
    Copilot foreground summarization marker (detected by RequestAnalyzer).
    SysToolsLockStage restored the real sys0; we now append the background
    summarization prompt as a new user message at the end of the
    conversation.
    """

    async def process(self, ctx: Context) -> Context:
        if ctx.meta.get("intent") != "compaction":
            return ctx

        log.info("[%s] rewriting compaction request to background format", ctx.req_id)
        msgs = ctx.messages  # canonical: conversation messages (no sys)
        msgs.append({"role": "user", "content": SUMMARIZATION_USER_MSG})
        ctx.messages = msgs  # property setter maps to fwd_body
        ctx.persist.is_modified = True
        ctx.meta["is_compaction_rewritten"] = True
        ctx.persist.modification_note = "compaction: foreground->background"
        return ctx
