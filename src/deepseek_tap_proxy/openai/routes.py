from __future__ import annotations
from aiohttp import web
from deepseek_tap_proxy.openai.handler import chat_completions_handler


def setup_openai_routes(app: web.Application) -> None:
    app.router.add_post("/chat/completions", chat_completions_handler)
