"""Rate limit / backoff, provider failures, calendar integration, performance."""

import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from idxbot.calendar.static import StaticIDXCalendar
from idxbot.calendar.base import MarketSession
from idxbot.data.normalize import normalize_batch
from idxbot.data.providers import FixtureProvider, RateLimitError, PermanentProviderError, TransientProviderError
from idxbot.data.providers.base import HistoricalRequest
from idxbot.data.rate_limit import ProviderRetry
from idxbot.data.quality import DataQualityEngine
from idxbot.data.universe import LiquidityFilter, UniverseBuilder

JAKARTA = ZoneInfo("Asia/Jakarta")


def test_http_429_retries_then_raises():
    p = FixtureProvider(scenario="http_429")
    retry = ProviderRetry(max_retries=2, base_backoff=0.01, max_backoff=0.05, jitter=False)
    with pytest.raises(RateLimitError):
        retry.run(lambda: p.get_symbols())


def test_timeout_transient():
    p = FixtureProvider(scenario="timeout")
    retry = ProviderRetry(max_retries=1, base_backoff=0.01, jitter=False)
    with pytest.raises(TransientProviderError):
        retry.run(lambda: p.get_symbols())


def test_auth_no_retry_storm():
    p = FixtureProvider(scenario="auth_error")
    calls = {"n": 0}
    retry = ProviderRetry(max_retries=5, base_backoff=0.01, jitter=False)

    def boom():
        calls["n"] += 1
        return p.get_symbols()

    with pytest.raises(PermanentProviderError):
        retry.run(boom)
    assert calls["n"] == 1  # permanent — no retry


def test_backoff_succeeds_after_transient():
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise TransientProviderError("temp")
        return "ok"

    retry = ProviderRetry(max_retries=3, base_backoff=0.01, jitter=False)
    assert retry.run(flaky) == "ok"
    assert state["n"] == 3


def test_calendar_integration_closed_weekend():
    cal = StaticIDXCalendar()
    ts = datetime(2024, 6, 15, 10, 0, tzinfo=JAKARTA)  # Saturday
    assert cal.get_market_session(ts) == MarketSession.CLOSED
    assert cal.is_trading_day(ts.date()) is False


def test_calendar_session1():
    cal = StaticIDXCalendar()
    ts = datetime(2024, 6, 17, 10, 0, tzinfo=JAKARTA)  # Monday
    assert cal.get_market_session(ts) == MarketSession.SESSION_1


def test_pipeline_performance_fixture():
    """Representative universe normalize + quality under 2 minutes."""
    t0 = time.perf_counter()
    p = FixtureProvider()
    filt = LiquidityFilter(min_avg_volume=1_000_000, min_history_days=60)
    symbols = UniverseBuilder(p, filt).symbols()
    rows = p.get_historical(
        HistoricalRequest(
            symbols=symbols,
            start=date(2024, 1, 1),
            end=date(2024, 6, 30),
        )
    )
    bars = normalize_batch(rows, provider=p.name())
    results = DataQualityEngine().check_series(bars)
    elapsed = time.perf_counter() - t0
    assert elapsed < 120
    assert len(results) == len(bars)
    assert len(bars) > 50
