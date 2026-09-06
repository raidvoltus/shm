"""MarketData / OHLCV contract validation tests."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from idxbot.data import MarketData, OHLCV

JAKARTA = ZoneInfo("Asia/Jakarta")


def _aware_ts():
    return datetime(2024, 6, 15, 10, 0, 0, tzinfo=JAKARTA)


def test_valid_market_data():
    md = MarketData(
        symbol="bbca",
        timestamp=_aware_ts(),
        open=9000.0,
        high=9100.0,
        low=8950.0,
        close=9050.0,
        volume=1_000_000,
    )
    assert md.symbol == "BBCA"
    assert md.timestamp.tzinfo is not None
    assert md.volume == 1_000_000


def test_symbol_empty_rejected():
    with pytest.raises(ValidationError):
        MarketData(
            symbol="  ",
            timestamp=_aware_ts(),
            open=1, high=1, low=1, close=1, volume=0,
        )


def test_naive_timestamp_rejected():
    with pytest.raises(ValidationError) as ei:
        MarketData(
            symbol="BBCA",
            timestamp=datetime(2024, 6, 15, 10, 0, 0),  # naive
            open=1, high=1, low=1, close=1, volume=0,
        )
    assert "timezone-aware" in str(ei.value).lower() or "tzinfo" in str(ei.value).lower()


def test_negative_price_rejected():
    with pytest.raises(ValidationError):
        MarketData(
            symbol="BBCA",
            timestamp=_aware_ts(),
            open=-1, high=1, low=1, close=1, volume=0,
        )


def test_high_lt_low_rejected():
    with pytest.raises(ValidationError):
        MarketData(
            symbol="BBCA",
            timestamp=_aware_ts(),
            open=100, high=90, low=95, close=92, volume=0,
        )


def test_negative_volume_rejected():
    with pytest.raises(ValidationError):
        MarketData(
            symbol="BBCA",
            timestamp=_aware_ts(),
            open=1, high=1, low=1, close=1, volume=-1,
        )


def test_to_jakarta_and_utc():
    ts_utc = datetime(2024, 6, 15, 3, 0, 0, tzinfo=timezone.utc)
    md = MarketData(
        symbol="TLKM",
        timestamp=ts_utc,
        open=3000, high=3100, low=2950, close=3050, volume=500,
    )
    j = md.to_jakarta()
    assert j.timestamp.tzinfo == JAKARTA
    u = md.to_utc()
    assert u.timestamp.utcoffset().total_seconds() == 0


def test_ohlcv_valid():
    bar = OHLCV(open=10, high=12, low=9, close=11, volume=100)
    assert bar.high >= bar.low
