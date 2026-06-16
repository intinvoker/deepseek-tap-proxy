from __future__ import annotations

from aiohttp import web

from deepseek_tap_proxy.anthropic.handler import anthropic_proxy_handler


def setup_anthropic_routes(app: web.Application) -> None:
    app.router.add_route("*", "/anthropic/{tail:.*}", anthropic_proxy_handler)
    app.router.add_post("/v1/messages", anthropic_proxy_handler)
