"""ANSI color utilities for terminal output.

Usage:
    from deepseek_tap_proxy.shared.colors import colorize, lock_symbol, format_msgs
    print(colorize("200", "status", 200, use_color=True))
"""

from __future__ import annotations

# ── ANSI escape codes ──────────────────────────────────────────────

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BLUE = "\033[34m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def green(text: str) -> str:
    return f"{_GREEN}{text}{_RESET}"


def yellow(text: str) -> str:
    return f"{_YELLOW}{text}{_RESET}"


def red(text: str) -> str:
    return f"{_RED}{text}{_RESET}"


def blue(text: str) -> str:
    return f"{_BLUE}{text}{_RESET}"


def cyan(text: str) -> str:
    return f"{_CYAN}{text}{_RESET}"


# ── Threshold-based coloring ───────────────────────────────────────


def colorize(text: str, field: str, value, use_color: bool = True) -> str:
    """Color `text` based on `field` and `value`.

    Args:
        text: The string to colorize.
        field: One of "status", "duration", "msgs", "lock", "cache_pct".
        value: The numeric/string value to evaluate.
        use_color: False = return plain text (for file output).
    """
    if not use_color:
        return text

    if field == "status":
        if value == 200:
            return green(text)
        elif 400 <= value < 500:
            return yellow(text)
        else:
            return red(text)

    elif field == "duration":
        if value < 3:
            return green(text)
        elif value < 10:
            return yellow(text)
        else:
            return red(text)

    elif field == "msgs":
        if "->" in text:
            before, after = text.split("->", 1)
            if before.isdigit() and after.isdigit():
                if int(before) <= int(after):
                    return green(text)
                else:
                    return red(text + "!!")
        return text

    elif field == "pref":
        # pref=命中/原始±增量
        # value = pref_hit_count (0=无命中 → 红, >0=部分 → 黄, 等于原始=绿)
        if value <= 0:
            return red(text)
        elif value >= 100:
            return green(text)
        return yellow(text)

    elif field == "lock":
        if value in ("layer1_perfect",):
            return green(text)
        elif value in ("layer2_partial",):
            return yellow(text)
        else:
            return red(text)

    elif field == "cache_pct":
        if value >= 50:
            return green(text)
        elif value >= 10:
            return yellow(text)
        else:
            return red(text)

    return text


def lock_symbol(lock_type: str | None) -> str:
    """Return display symbol for lock type (ASCII-only)."""
    mapping = {
        "layer1_perfect": "1OK",
        "layer2_partial": "2~~",
        "none": "none",
    }
    return mapping.get(lock_type or "none", lock_type or "none")


def format_msgs(before: int, after: int) -> str:
    return f"{before}->{after}"
