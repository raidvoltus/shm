"""Core feature calculations: returns, SMA, EMA, RSI, MACD, ATR, volume, price action."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import math
import pytest

from idxbot.data.normalize import NormalizedBar, normalize_record
from idxbot.data.providers.fixture import _bar
from idxbot.features import FeatureEngine, FeatureEngineError, FEATURE_COLUMNS, FEATURE_METADATA

JAKARTA = ZoneInfo("Asia/Jakarta")


def _series(n: int = 80, start: date = date(2024, 1, 2), symbol: str = "TEST") -> list[NormalizedBar]:
    bars = []
    price = 1000.0
    d = start
    while len(bars) < n:
        if d.weekday() < 5:
            o, c = price, price * (1 + (len(bars) % 7 - 3) * 0.005)
            h, l = max(o, c) * 1.01, min(o, c) * 0.99
            raw = _bar(symbol, d, o, h, l, c, 1_000_000 + len(bars) * 1000)
            bars.append(normalize_record(raw, provider="fixture"))
            price = c
        d += timedelta(days=1)
    return bars


def test_timeframe_reject():
    with pytest.raises(FeatureEngineError):
        FeatureEngine(timeframe="1h")


def test_returns_sma_ema_present():
    eng = FeatureEngine()
    rows = eng.transform(_series(60))
    assert len(rows) == 60
    last = rows[-1]
    assert last["return_1d"] is not None and not math.isnan(last["return_1d"])
    assert not math.isnan(last["sma_5"])
    assert not math.isnan(last["sma_50"])
    assert not math.isnan(last["ema_10"])
    assert last["feature_valid"] is True
    early = rows[3]
    assert math.isnan(early["sma_50"])
    assert early["feature_valid"] is False


def test_rsi_macd_atr():
    eng = FeatureEngine()
    rows = eng.transform(_series(80))
    last = rows[-1]
    assert 0 <= last["rsi_14"] <= 100 or math.isnan(last["rsi_14"])
    assert not math.isnan(last["macd"])
    assert not math.isnan(last["atr_14"])
    assert last["true_range"] >= 0


def test_volume_and_price_action():
    eng = FeatureEngine()
    rows = eng.transform(_series(40))
    last = rows[-1]
    assert not math.isnan(last["volume_sma_20"])
    assert last["daily_range"] >= 0
    assert last["body_size"] >= 0
    assert last["upper_wick"] >= -1e-9
    assert "gap_from_previous_close" in last


def test_regime_features():
    eng = FeatureEngine()
    rows = eng.transform(_series(60))
    last = rows[-1]
    assert last["trend_regime"] in (-1.0, 0.0, 1.0) or math.isnan(last["trend_regime"])
    assert 0 <= last["volatility_regime"] <= 1 or math.isnan(last["volatility_regime"])


def test_schema_and_metadata():
    assert "return_1d" in FEATURE_COLUMNS
    assert FEATURE_COLUMNS[0] == "symbol"
    assert "rsi_14" in FEATURE_METADATA
    meta = FEATURE_METADATA["sma_50"]
    assert meta.window == 50
    assert meta.price_mode == "adjusted"
    eng = FeatureEngine()
    md = eng.metadata()
    assert "macd" in md
    assert md["macd"]["version"] == "1"


def test_warmup_nan_not_zero():
    eng = FeatureEngine()
    rows = eng.transform(_series(10))
    assert math.isnan(rows[0]["sma_50"])
    assert rows[0]["sma_50"] != 0
