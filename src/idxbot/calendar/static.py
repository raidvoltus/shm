"""
Static IDX calendar for PHASE 2.

Uses weekend rules + optional holiday set.
Does NOT hardcode a long holiday list into business logic;
holidays are injectable via provider.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Optional, Set
from zoneinfo import ZoneInfo

from idxbot.calendar.base import (
    CalendarProvider,
    MarketCalendar,
    MarketSession,
    SessionWindow,
    JAKARTA,
)

# Standard IDX regular session times (WIB) — approximate for contract purposes
SESSION_1_START = time(9, 0)
SESSION_1_END = time(12, 0)
BREAK_END = time(13, 30)
SESSION_2_END = time(16, 0)


class WeekendProvider(CalendarProvider):
    """Trading day = Mon–Fri. Holidays can be layered on top."""

    def __init__(self, holidays: Optional[Set[date]] = None) -> None:
        self._holidays = holidays or set()

    def is_trading_day(self, d: date) -> bool:
        if d.weekday() >= 5:  # Sat=5, Sun=6
            return False
        if d in self._holidays:
            return False
        return True

    def name(self) -> str:
        return "weekend+holidays"


class FailingProvider(CalendarProvider):
    """Test helper: always raises or times out."""

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error or TimeoutError("calendar provider timeout")

    def is_trading_day(self, d: date) -> bool:
        raise self._error

    def name(self) -> str:
        return "failing"


class StaticIDXCalendar(MarketCalendar):
    """
    IDX market calendar with fail-closed semantics.

    If the underlying provider fails (timeout, exception, malformed),
    every query returns CLOSED / not a trading day.
    """

    def __init__(
        self,
        provider: Optional[CalendarProvider] = None,
        *,
        fail_closed: bool = True,
    ) -> None:
        self.provider = provider or WeekendProvider()
        self.fail_closed = fail_closed
        self._last_error: Optional[str] = None

    def _safe_is_trading_day(self, d: date) -> bool:
        try:
            return bool(self.provider.is_trading_day(d))
        except Exception as exc:  # noqa: BLE001 — fail-closed by design
            self._last_error = f"{type(exc).__name__}: {exc}"
            if self.fail_closed:
                return False  # CLOSED
            raise

    def is_trading_day(self, d: date) -> bool:
        return self._safe_is_trading_day(d)

    def get_market_session(self, ts: datetime) -> MarketSession:
        if ts.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        local = ts.astimezone(JAKARTA)
        d = local.date()
        if not self._safe_is_trading_day(d):
            return MarketSession.CLOSED

        t = local.time()
        if t < SESSION_1_START:
            return MarketSession.CLOSED
        if SESSION_1_START <= t < SESSION_1_END:
            return MarketSession.SESSION_1
        if SESSION_1_END <= t < BREAK_END:
            return MarketSession.BREAK
        if BREAK_END <= t < SESSION_2_END:
            return MarketSession.SESSION_2
        return MarketSession.POST_MARKET

    def get_session_window(self, d: date) -> list[SessionWindow]:
        if not self._safe_is_trading_day(d):
            return []

        def at(t: time) -> datetime:
            return datetime.combine(d, t, tzinfo=JAKARTA)

        return [
            SessionWindow(MarketSession.SESSION_1, at(SESSION_1_START), at(SESSION_1_END)),
            SessionWindow(MarketSession.BREAK, at(SESSION_1_END), at(BREAK_END)),
            SessionWindow(MarketSession.SESSION_2, at(BREAK_END), at(SESSION_2_END)),
        ]

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error
