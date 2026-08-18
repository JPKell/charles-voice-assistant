from __future__ import annotations

import atexit
import os
import sys


USER_COLOR = "\033[36m"
ASSISTANT_COLOR = "\033[32m"
SYSTEM_COLOR = "\033[33m"
DIM_COLOR = "\033[90m"
COLOR_RESET = "\033[0m"
_SYSTEM_COLOR_ACTIVE = False


def terminal_colors_enabled(stream=None) -> bool:
    if stream is None:
        stream = sys.stdout
    return os.environ.get("NO_COLOR") is None and stream.isatty()


def enable_system_color() -> None:
    """Use yellow as the default color for terminal status messages."""
    global _SYSTEM_COLOR_ACTIVE
    if not terminal_colors_enabled():
        return
    _SYSTEM_COLOR_ACTIVE = True
    sys.stdout.write(SYSTEM_COLOR)
    sys.stdout.flush()
    atexit.register(lambda: sys.stdout.write(COLOR_RESET))


def colorize(text: str, color: str, stream=None, *, restore_system: bool = True) -> str:
    """Color terminal output while keeping redirected output clean."""
    if stream is None:
        stream = sys.stdout
    if not terminal_colors_enabled(stream):
        return text
    reset = (
        SYSTEM_COLOR
        if restore_system and _SYSTEM_COLOR_ACTIVE and stream is sys.stdout
        else COLOR_RESET
    )
    return f"{color}{text}{reset}"


def restore_system_color() -> None:
    if _SYSTEM_COLOR_ACTIVE and terminal_colors_enabled():
        sys.stdout.write(SYSTEM_COLOR)
        sys.stdout.flush()


def speaker_label(name: str, color: str) -> str:
    return colorize(f"{name}:", color, restore_system=False)


def print_message(name: str, text: str, color: str) -> None:
    print(f"{speaker_label(name, color)} {text}")
    restore_system_color()


def print_dim(text: str) -> None:
    """Print low-priority diagnostics in unobtrusive dark gray."""
    print(colorize(text, DIM_COLOR))
