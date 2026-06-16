from __future__ import annotations
from aiohttp import web
from deepseek_tap_proxy.openai.handler import chat_completions_handler, fim_passthrough_handler


def setup_openai_routes(app: web.Application) -> None:
    app.router.add_post("/chat/completions", chat_completions_handler)
    app.router.add_post("/v1/chat/completions", chat_completions_handler)
    # FIM (Fill-in-the-Middle) passthrough
    app.router.add_post("/completions", fim_passthrough_handler)
    app.router.add_post("/beta/completions", fim_passthrough_handler)
