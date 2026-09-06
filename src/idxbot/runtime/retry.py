"""
Bounded retry for transient errors; no retry for fatal errors.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")

# Errors that may be retried
TRANSIENT_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    OSError,  # includes temporary I/O
    InterruptedError,
)

# Explicit fatal markers
class FatalError(Exception):
    """Non-retryable error — configuration, safety, corruption."""


class SafetyViolation(FatalError):
    pass


class StateCorruption(FatalError):
    pass


class ConfigurationError(FatalError):
    pass


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, FatalError):
        return False
    return isinstance(exc, TRANSIENT_EXCEPTIONS)


def is_fatal(exc: BaseException) -> bool:
    return isinstance(exc, FatalError)


class RetryPolicy:
    """Bounded exponential backoff for transient failures only."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay_sec: float = 0.1,
        max_delay_sec: float = 2.0,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.max_attempts = max_attempts
        self.base_delay_sec = base_delay_sec
        self.max_delay_sec = max_delay_sec

    def run(self, fn: Callable[[], T]) -> T:
        last: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                last = exc
                if is_fatal(exc) or not is_transient(exc):
                    raise
                if attempt >= self.max_attempts:
                    raise
                delay = min(
                    self.base_delay_sec * (2 ** (attempt - 1)),
                    self.max_delay_sec,
                )
                time.sleep(delay)
        assert last is not None
        raise last
