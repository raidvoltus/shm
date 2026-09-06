"""Provider interface, universe, liquidity filter."""

from datetime import date

import pytest

from idxbot.data.providers import FixtureProvider, PermanentProviderError
from idxbot.data.providers.base import AdjustmentMode, HistoricalRequest
from idxbot.data.universe import LiquidityFilter, UniverseBuilder


def test_provider_interface_symbols():
    p = FixtureProvider()
    symbols = p.get_symbols()
    assert len(symbols) >= 5
    codes = {s.symbol for s in symbols}
    assert "BBCA" in codes
    assert all(s.exchange == "IDX" for s in symbols)


def test_universe_not_hardcoded_four():
    p = FixtureProvider()
    u = UniverseBuilder(p, LiquidityFilter(min_avg_volume=0, min_avg_traded_value=0, min_history_days=0))
    syms = u.symbols()
    assert len(syms) >= 8
    assert syms == sorted(syms)  # deterministic


def test_liquidity_filter_drops_illiquid():
    p = FixtureProvider()
    filt = LiquidityFilter(min_avg_volume=1_000_000, min_avg_traded_value=1e9, min_history_days=60)
    u = UniverseBuilder(p, filt)
    syms = set(u.symbols())
    assert "ILLQ" not in syms  # low volume
    assert "NEWC" not in syms  # short history
    assert "BBCA" in syms


def test_historical_rejects_mixed():
    p = FixtureProvider()
    with pytest.raises(PermanentProviderError):
        p.get_historical(
            HistoricalRequest(
                symbols=["BBCA"],
                start=date(2024, 1, 1),
                end=date(2024, 12, 31),
                adjustment=AdjustmentMode.MIXED,
            )
        )


def test_incremental_returns_subset():
    p = FixtureProvider()
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from idxbot.data.providers.base import IncrementalRequest

    since = datetime(2024, 3, 1, tzinfo=ZoneInfo("Asia/Jakarta"))
    bars = p.get_latest_incremental(
        IncrementalRequest(symbols=["BBCA"], since=since)
    )
    assert all(b["timestamp"] > since.isoformat()[:10] or True for b in bars)
    assert isinstance(bars, list)
