"""
Market calendar interface.

Fail-closed: any provider failure → MARKET = CLOSED.
Timeout for external providers: 5 seconds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

JAKARTA = ZoneInfo("Asia/Jakarta")

# Hard timeout for any external calendar provider
CALENDAR_PROVIDER_TIMEOUT_SEC = 5.0


class MarketSession(str, Enum):
    CLOSED = "CLOSED"
    SESSION_1 = "SESSION_1"  # typically 09:00–12:00 WIB
    BREAK = "BREAK"  # 12:00–13:30 WIB (approx)
    SESSION_2 = "SESSION_2"  # typically 13:30–16:00 WIB
    POST_MARKET = "POST_MARKET"


@dataclass(frozen=True)
class SessionWindow:
    session: MarketSession
    start: datetime  # aware
    end: datetime  # aware

    def contains(self, ts: datetime) -> bool:
        if ts.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return self.start <= ts < self.end


class CalendarProvider(ABC):
    """Source of holiday / trading-day data. Implementations must be fast or fail."""

    @abstractmethod
    def is_trading_day(self, d: date) -> bool:
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class MarketCalendar(ABC):
    """High-level calendar used by the runtime."""

    @abstractmethod
    def is_trading_day(self, d: date) -> bool:
        ...

    @abstractmethod
    def get_market_session(self, ts: datetime) -> MarketSession:
        ...

    @abstractmethod
    def get_session_window(self, d: date) -> list[SessionWindow]:
        ...
