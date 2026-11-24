"""DSH（DeepSeek Harness）检测逻辑。

DSH 适配器会覆盖 pi-ai 的默认 UA，真实请求头是::

    deepseek-harness/<version> (+https://github.com/deepseek-ai/deepseek-harness)

（按真实请求样本归纳的固定形态），并且请求体不含
任何会话标识 —— UA 是当前唯一稳定的客户端特征。
"""
from __future__ import annotations

from deepseek_tap_proxy.pipeline import Context
from deepseek_tap_proxy.plugins.base import DetectionResult

# DSH 适配器自报的 UA 前缀；pi-ai 直接发请求的 ``pi (...)`` 不带 DSH 身份。
DSH_UA_PREFIX = "deepseek-harness/"


def detect_dsh(ctx: Context) -> DetectionResult | None:
    """检测 DSH。返回 None = 不匹配。"""
    ua = ctx.request.headers.get("User-Agent", "")
    if not isinstance(ua, str) or not ua.lower().startswith(DSH_UA_PREFIX):
        return None

    # 只接 OpenAI chat 形态；Anthropic 格式的 body 结构不同，不做本适配。
    if ctx.format != "openai":
        return None
    if not isinstance(ctx.fwd_body, dict):
        return None
    if not isinstance(ctx.fwd_body.get("messages"), list):
        return None

    # DSH 请求体不含 session id：如实留空，不伪造身份。
    return DetectionResult(
        client_type="dsh",
        chat_type="normal",
        session_id="",
    )
