"""
Bounded retry with exponential backoff for provider calls.

HTTP 429 → RateLimitError → retry with backoff.
Auth / schema errors → no unbounded retry.
"""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from idxbot.data.providers.base import (
    PermanentProviderError,
    ProviderError,
    RateLimitError,
    TransientProviderError,
)

T = TypeVar("T")


class ProviderRetry:
    def __init__(
        self,
        max_retries: int = 3,
        base_backoff: float = 0.05,
        max_backoff: float = 1.0,
        jitter: bool = True,
    ) -> None:
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.jitter = jitter

    def run(self, fn: Callable[[], T]) -> T:
        last: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except PermanentProviderError:
                raise
            except (RateLimitError, TransientProviderError, TimeoutError, ConnectionError) as exc:
                last = exc
                if attempt >= self.max_retries:
                    raise
                delay = min(self.base_backoff * (2**attempt), self.max_backoff)
                if self.jitter:
                    delay *= 0.5 + random.random()
                time.sleep(delay)
            except ProviderError:
                raise
        assert last is not None
        raise last
