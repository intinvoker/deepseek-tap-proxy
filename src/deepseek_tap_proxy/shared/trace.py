"""Backward-compatibility re-exports — moved to shared/log.py.

TraceWriter and TraceView are now defined in shared/log.py as part of the
application-scenario view layer alongside LogView and OtlpLogWriter.
"""

from deepseek_tap_proxy.shared.log import TraceWriter, TraceView

__all__ = ["TraceWriter", "TraceView"]
