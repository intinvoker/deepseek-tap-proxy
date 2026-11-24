"""本地 tokenizer 服务客户端 — 用于 /v1/messages/count_tokens 的 fallback 精确计数。

背景：count_tokens 在向上游请求失败后，此前只用「字符数 / 4」估算（中文误差较大）。
本模块在 fallback 时优先调用本地 tokenizer 服务（HTTP `/count-batch`）得到精确计数，
不可用时仍退回字符估算，保证端点永不失败。

配置：环境变量 `DEEPSEEK_TAP_TOKENIZER_URL`（默认空 = 关闭），例如
`http://127.0.0.1:<port>`；默认关闭是为保证 CI/单测确定性（不依赖外部服务）。

语义：只对纯文本（system / tools JSON / messages 文本块）计数，不含 chat 模板开销；
因此是「接近上游真实 input_tokens」的精确估算，可能相差少量模板 token。
"""

import json
import logging
import os

import aiohttp

log = logging.getLogger("deepseek-tap")

DEFAULT_TIMEOUT = 2.0


def tokenizer_url() -> str:
    return os.getenv("DEEPSEEK_TAP_TOKENIZER_URL", "").strip()


def _texts_from_body(body: dict) -> list[str]:
    """从 Anthropic 请求体抽取需要计数的文本片段（顺序无关，求和即可）。"""
    texts: list[str] = []
    system = body.get("system")
    if isinstance(system, str):
        texts.append(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                texts.append(block["text"])

    tools = body.get("tools")
    if tools:
        texts.append(json.dumps(tools, ensure_ascii=False))

    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    texts.append(block["text"])
    return texts


async def count_tokens_exact(body: dict | None, session=None) -> int | None:
    """向本地 tokenizer 服务请求精确 token 数；未配置/失败/超时返回 None（调用方回退估算）。"""
    url = tokenizer_url()
    if not url or not isinstance(body, dict):
        return None
    texts = _texts_from_body(body)
    if not texts:
        return None

    own_session = session is None
    try:
        if own_session:
            session = aiohttp.ClientSession()
        async with session.post(
            url.rstrip("/") + "/count-batch",
            json={"texts": texts},
            timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                log.warning(f"[count_tokens] tokenizer http {resp.status}, fallback to estimate")
                return None
            data = await resp.json()
        results = data.get("results") or []
        total = sum(int(item.get("tokenCount", 0)) for item in results)
        return total if total > 0 else None
    except Exception as exc:  # noqa: BLE001 — 任何失败都必须降级，不能影响端点
        log.warning(f"[count_tokens] tokenizer unavailable ({exc}), fallback to estimate")
        return None
    finally:
        if own_session and session is not None:
            await session.close()
