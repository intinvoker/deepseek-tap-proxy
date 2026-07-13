"""格式无关 pipeline stages — 操作 canonical view。

所有 stage 通过 ``ctx.sys`` / ``ctx.tools`` / ``ctx.messages`` 操作，
不直接访问 ``fwd_body``。对 OpenAI 和 Anthropic 请求统一工作。
"""
from deepseek_tap_proxy.shared.stages.canonical import CanonicalStage
from deepseek_tap_proxy.shared.stages.sys_tools_lock import SysToolsLockStage
from deepseek_tap_proxy.shared.stages.prefix_lock import PrefixLockStage
from deepseek_tap_proxy.shared.stages.compaction_lock import CompactionLockStage
from deepseek_tap_proxy.shared.stages.cache_analysis import apply_cache_analysis, emit_request_log

__all__ = [
    "CanonicalStage",
    "SysToolsLockStage",
    "PrefixLockStage",
    "CompactionLockStage",
    "apply_cache_analysis",
    "emit_request_log",
]
