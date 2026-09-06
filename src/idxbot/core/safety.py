"""
Absolute live-trading safety guards.

LIVE_TRADING is false by default and must remain false.
Any attempt to enable it results in a fatal error (fail-closed).
"""

from __future__ import annotations

from idxbot.config import get_settings


class SafetyError(RuntimeError):
    """Raised when a safety invariant is violated."""


def assert_no_live_trading() -> None:
    """
    Call this before any trading-related action.

    Raises SafetyError if LIVE_TRADING is true.
    """
    settings = get_settings()
    settings.assert_safe()
    if settings.live_trading:
        raise SafetyError(
            "SAFETY GUARD TRIGGERED: LIVE_TRADING=true is not allowed. "
            "This project is signal-only with paper trading. "
            "Live broker execution is permanently disabled."
        )
