"""Anthropic Messages API SSE event reconstructor."""

from __future__ import annotations

import copy
import json
import orjson

from deepseek_tap_proxy.shared.sse_parser import SSEEvent, SSEParser


class AnthropicReconstructor:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self._snapshot: dict | None = None
        self._parser = SSEParser()

    def feed_bytes(self, chunk: bytes) -> None:
        for event in self._parser.feed(chunk):
            self._process_event(event)

    def _process_event(self, event: SSEEvent) -> None:
        event_type = event.event_type or "message"
        try:
            data = orjson.loads(event.data)
        except (orjson.JSONDecodeError, json.JSONDecodeError, ValueError):
            data = event.data
        self.events.append({"event": event_type, "data": data})
        if isinstance(data, dict):
            self._accumulate(event_type, data)

    def _accumulate(self, event_type: str, data: dict) -> None:
        if event_type == "message_start":
            self._snapshot = copy.deepcopy(data.get("message", {}))
        elif event_type in ("response.created", "response.completed", "response.done"):
            resp = data.get("response")
            if isinstance(resp, dict):
                self._snapshot = copy.deepcopy(resp)
        elif self._snapshot is None:
            return
        elif event_type == "content_block_start":
            block = copy.deepcopy(data.get("content_block", {}))
            if "content" not in self._snapshot:
                self._snapshot["content"] = []
            idx = data.get("index", len(self._snapshot["content"]))
            while len(self._snapshot["content"]) <= idx:
                self._snapshot["content"].append({})
            self._snapshot["content"][idx] = block
        elif event_type == "content_block_delta":
            idx = data.get("index", 0)
            delta = data.get("delta", {})
            if idx < len(self._snapshot.get("content", [])):
                block = self._snapshot["content"][idx]
                dt = delta.get("type", "")
                if dt == "text_delta":
                    block["text"] = block.get("text", "") + delta.get("text", "")
                elif dt == "thinking_delta":
                    block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
                elif dt == "input_json_delta":
                    block["_partial"] = block.get("_partial", "") + delta.get("partial_json", "")
        elif event_type == "content_block_stop":
            idx = data.get("index", 0)
            if idx < len(self._snapshot.get("content", [])):
                block = self._snapshot["content"][idx]
                if "_partial" in block:
                    try:
                        block["input"] = orjson.loads(block.pop("_partial"))
                    except (orjson.JSONDecodeError, json.JSONDecodeError, ValueError):
                        pass
        elif event_type == "message_delta":
            delta = data.get("delta", {})
            for k, v in delta.items():
                self._snapshot[k] = v
            usage = data.get("usage", {})
            if usage:
                self._snapshot.setdefault("usage", {}).update(usage)

    def reconstruct(self) -> dict | None:
        return self._snapshot
