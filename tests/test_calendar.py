"""Market calendar: sessions, trading day, fail-closed, timeout."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from idxbot.calendar.base import MarketSession
from idxbot.calendar.static import StaticIDXCalendar, WeekendProvider, FailingProvider

JAKARTA = ZoneInfo("Asia/Jakarta")


def _ts(d: date, h: int, m: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, h, m, tzinfo=JAKARTA)


def test_weekday_is_trading_day():
    cal = StaticIDXCalendar()
    assert cal.is_trading_day(date(2024, 6, 17)) is True  # Monday


def test_weekend_closed():
    cal = StaticIDXCalendar()
    assert cal.is_trading_day(date(2024, 6, 15)) is False  # Saturday
    assert cal.is_trading_day(date(2024, 6, 16)) is False  # Sunday


def test_holiday_injected():
    hol = {date(2024, 6, 17)}
    cal = StaticIDXCalendar(WeekendProvider(holidays=hol))
    assert cal.is_trading_day(date(2024, 6, 17)) is False


def test_session_1():
    cal = StaticIDXCalendar()
    d = date(2024, 6, 17)
    assert cal.get_market_session(_ts(d, 9, 30)) == MarketSession.SESSION_1
    assert cal.get_market_session(_ts(d, 11, 59)) == MarketSession.SESSION_1


def test_session_2():
    cal = StaticIDXCalendar()
    d = date(2024, 6, 17)
    assert cal.get_market_session(_ts(d, 13, 30)) == MarketSession.SESSION_2
    assert cal.get_market_session(_ts(d, 15, 0)) == MarketSession.SESSION_2


def test_break_and_closed():
    cal = StaticIDXCalendar()
    d = date(2024, 6, 17)
    assert cal.get_market_session(_ts(d, 12, 30)) == MarketSession.BREAK
    assert cal.get_market_session(_ts(d, 8, 0)) == MarketSession.CLOSED
    assert cal.get_market_session(_ts(d, 16, 30)) == MarketSession.POST_MARKET


def test_weekend_session_closed():
    cal = StaticIDXCalendar()
    d = date(2024, 6, 15)  # Sat
    assert cal.get_market_session(_ts(d, 10, 0)) == MarketSession.CLOSED


def test_session_windows():
    cal = StaticIDXCalendar()
    wins = cal.get_session_window(date(2024, 6, 17))
    assert len(wins) == 3
    assert wins[0].session == MarketSession.SESSION_1
    assert wins[2].session == MarketSession.SESSION_2


def test_fail_closed_on_provider_error():
    cal = StaticIDXCalendar(FailingProvider(TimeoutError("timeout")), fail_closed=True)
    assert cal.is_trading_day(date(2024, 6, 17)) is False
    assert cal.get_market_session(_ts(date(2024, 6, 17), 10, 0)) == MarketSession.CLOSED
    assert cal.last_error is not None
    assert "Timeout" in cal.last_error or "timeout" in cal.last_error.lower()


def test_fail_closed_malformed():
    class Bad(FailingProvider):
        def is_trading_day(self, d):
            raise ValueError("malformed calendar payload")

    cal = StaticIDXCalendar(Bad(), fail_closed=True)
    assert cal.is_trading_day(date(2024, 6, 17)) is False


def test_naive_timestamp_rejected():
    cal = StaticIDXCalendar()
    with pytest.raises(ValueError):
        cal.get_market_session(datetime(2024, 6, 17, 10, 0, 0))
