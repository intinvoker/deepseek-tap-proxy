"""IETF text/event-stream line-protocol parser.

Parses raw bytes into SSEEvent tuples. Does not understand any
specific API format's data structures.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SSEEvent:
    """A single SSE event from the wire."""
    event_type: str | None  # value of "event:" line, or None
    data: str               # concatenated value of "data:" lines


class SSEParser:
    """Parse SSE byte stream into events.

    Usage:
        parser = SSEParser()
        for chunk in response.content.iter_any():
            for event in parser.feed(chunk):
                handle(event)
        for event in parser.flush():
            handle(event)
    """

    def __init__(self) -> None:
        self._buf = b""
        self._current_event: str | None = None
        self._current_data_lines: list[str] = []

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        """Feed raw bytes, return completed events."""
        self._buf += chunk
        events: list[SSEEvent] = []
        while b"\n" in self._buf:
            line_bytes, self._buf = self._buf.split(b"\n", 1)
            event = self._feed_line(
                line_bytes.decode("utf-8", errors="replace").rstrip("\r")
            )
            if event is not None:
                events.append(event)
        return events

    def flush(self) -> list[SSEEvent]:
        """Return any remaining buffered event."""
        events: list[SSEEvent] = []
        if self._current_event is not None or self._current_data_lines:
            raw_data = "\n".join(self._current_data_lines)
            if raw_data and raw_data != "[DONE]":
                events.append(SSEEvent(
                    event_type=self._current_event,
                    data=raw_data,
                ))
            self._current_event = None
            self._current_data_lines = []
        return events

    def _feed_line(self, line: str) -> SSEEvent | None:
        if line.startswith("event:"):
            self._current_event = line[len("event:"):].strip()
            self._current_data_lines = []
        elif line.startswith("data:"):
            self._current_data_lines.append(line[len("data:"):].strip())
        elif line == "":
            if self._current_event is not None or self._current_data_lines:
                raw_data = "\n".join(self._current_data_lines)
                event_type = self._current_event
                self._current_event = None
                self._current_data_lines = []
                if raw_data == "[DONE]" and event_type is None:
                    return None  # skip terminator events
                return SSEEvent(event_type=event_type, data=raw_data)
        return None
