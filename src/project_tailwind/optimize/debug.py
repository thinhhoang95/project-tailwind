"""
Centralized debugging utilities for optimization modules.

Enable by setting environment variable PT_DEBUG_ALNS=1 (or any non-zero value),
or by calling set_debug(True) from code. All debug prints use Rich.
"""

from __future__ import annotations

import os
from typing import Final

from rich.console import Console
from rich.theme import Theme


_DEFAULT_THEME: Final = Theme(
    {
        "info": "cyan",
        "warn": "yellow",
        "error": "bold red",
        "debug": "magenta",
        "success": "green",
    }
)

console: Final = Console(theme=_DEFAULT_THEME)


_debug_enabled: bool = os.getenv("PT_DEBUG_ALNS", "0") not in ("0", "false", "False", "")


def is_debug_enabled() -> bool:
    return _debug_enabled


def set_debug(enabled: bool) -> None:
    global _debug_enabled
    _debug_enabled = bool(enabled)


