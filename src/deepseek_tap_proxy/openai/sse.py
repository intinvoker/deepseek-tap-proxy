"""OpenAI Chat Completions SSE event reconstructor."""

from __future__ import annotations

import json

from deepseek_tap_proxy.shared.sse_parser import SSEEvent, SSEParser
from deepseek_tap_proxy.openai.usage import normalize_usage


class OpenaiReconstructor:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self._snapshot: dict | None = None
        self._parser = SSEParser()

    def feed_bytes(self, chunk: bytes) -> None:
        for event in self._parser.feed(chunk):
            self._process_event(event)

    def _process_event(self, event: SSEEvent) -> None:
        try:
            data = json.loads(event.data)
        except (json.JSONDecodeError, ValueError):
            data = event.data
        self.events.append({"event": event.event_type or "message", "data": data})
        if isinstance(data, dict):
            self._accumulate(data)

    def _accumulate(self, data: dict) -> None:
        choices = data.get("choices") or []
        usage = data.get("usage")
        if not isinstance(choices, list) or not choices:
            if isinstance(usage, dict) and self._snapshot is not None:
                self._snapshot["usage"] = normalize_usage(usage)
            return
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta") or {}
        finish_reason = choice.get("finish_reason")
        if self._snapshot is None:
            self._snapshot = {
                "id": data.get("id", ""),
                "object": "chat.completion",
                "model": data.get("model", ""),
                "choices": [{"index": 0, "message": {"role": delta.get("role") or "assistant", "content": ""}, "finish_reason": None}],
            }
        msg = self._snapshot["choices"][0]["message"]
        if delta.get("content"):
            msg["content"] = (msg.get("content") or "") + delta["content"]
        if delta.get("reasoning_content"):
            msg["reasoning_content"] = (msg.get("reasoning_content") or "") + delta["reasoning_content"]
        if finish_reason:
            self._snapshot["choices"][0]["finish_reason"] = finish_reason
        if usage:
            self._snapshot["usage"] = normalize_usage(usage)
        for tc_delta in delta.get("tool_calls") or []:
            if not isinstance(tc_delta, dict):
                continue
            idx = tc_delta.get("index", 0)
            tool_calls = msg.setdefault("tool_calls", [])
            while len(tool_calls) <= idx:
                tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            existing = tool_calls[idx]
            if tc_delta.get("id"):
                existing["id"] = tc_delta["id"]
            if tc_delta.get("type"):
                existing["type"] = tc_delta["type"]
            fn = tc_delta.get("function") or {}
            if isinstance(fn, dict):
                ef = existing.setdefault("function", {"name": "", "arguments": ""})
                if fn.get("name"):
                    ef["name"] = (ef.get("name") or "") + fn["name"]
                if fn.get("arguments"):
                    ef["arguments"] = (ef.get("arguments") or "") + fn["arguments"]

    def reconstruct(self) -> dict | None:
        return self._snapshot
