"""Client type 检测常量 — 跨 Plugin 共享。"""
from __future__ import annotations
import re

# Copilot 特征
COPILOT_NORMAL_PREFIX = "You are an expert AI programming assistant"
COMPACTION_SIGNATURE = "Your task is to create a comprehensive, detailed summary"

# Session UUID 正则（匹配 GitHub.copilot-chat/debug-logs/{uuid} 或 transcripts/{uuid}）
SESSION_UUID_RE = re.compile(
    r'GitHub\.copilot-chat[\\/](?:debug-logs|transcripts)[\\/]'
    r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})'
)

# Hermes Session ID 正则
HERMES_SESSION_ID_RE = re.compile(
    r'(?:^|\n)\s*(?:Current )?Session\s+ID[:\s]+(\S+)', re.IGNORECASE
)
