"""Normalization, quality engine, corporate-action flags, freshness."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from idxbot.data.normalize import normalize_batch, normalize_record
from idxbot.data.providers.fixture import FixtureProvider, _bar
from idxbot.data.quality import DataQualityEngine, QualityStatus, ReasonCode

JAKARTA = ZoneInfo("Asia/Jakarta")


def test_normalize_raw_and_adjusted():
    raw = _bar("BBCA", date(2024, 3, 14), 9900, 10100, 9800, 10000, 1_000_000, adj_factor=0.2)
    bar = normalize_record(raw, provider="fixture", prefer_adjusted=True)
    assert bar.symbol == "BBCA"
    assert bar.timestamp.tzinfo is not None
    assert bar.adjustment_mode == "ADJUSTED"
    assert bar.raw_close == 10000
    assert bar.adjusted_close == pytest.approx(2000.0)
    assert bar.close == pytest.approx(2000.0)  # canonical uses adjusted


def test_mixed_rejected():
    raw = _bar("X", date(2024, 1, 2), 1, 1, 1, 1, 1)
    raw["adjustment_mode"] = "MIXED"
    with pytest.raises(ValueError, match="MIXED"):
        normalize_record(raw)


def test_naive_timestamp_rejected():
    raw = _bar("X", date(2024, 1, 2), 1, 1, 1, 1, 1)
    raw["timestamp"] = "2024-01-02T16:00:00"  # naive
    with pytest.raises(ValueError):
        normalize_record(raw)


def test_stock_split_flagged_not_silent():
    """
    T-1 close 10000 → T close 2000 must FLAG corporate-action review,
    not silent-correct or auto-reject as crash.
    """
    p = FixtureProvider()
    from idxbot.data.providers.base import HistoricalRequest

    rows = p.get_historical(
        HistoricalRequest(
            symbols=["BBCA"],
            start=date(2024, 3, 14),
            end=date(2024, 3, 18),
        )
    )
    bars = normalize_batch(rows, provider=p.name())
    # only the split-related days
    focus = [b for b in bars if b.timestamp.date() in (date(2024, 3, 14), date(2024, 3, 15))]
    assert len(focus) >= 2

    engine = DataQualityEngine(gap_threshold=0.25)
    results = engine.check_series(focus)
    # find the post-split bar
    flagged = [r for r in results if ReasonCode.ABNORMAL_GAP in r.reasons]
    assert flagged, "expected ABNORMAL_GAP on 10000→2000"
    assert any(ReasonCode.CORPORATE_ACTION_REVIEW in r.reasons for r in flagged)
    # must not be REJECTED solely for the gap
    for r in flagged:
        assert r.status == QualityStatus.FLAGGED
        assert r.bar is not None
        # adjusted data present on pre-split fixture
        if r.bar.timestamp.date() == date(2024, 3, 14):
            assert r.bar.adjusted_close is not None


def test_invalid_ohlc_rejected():
    p = FixtureProvider(scenario="invalid_ohlc")
    from idxbot.data.providers.base import HistoricalRequest

    rows = p.get_historical(
        HistoricalRequest(symbols=["UNVR"], start=date(2024, 1, 1), end=date(2024, 12, 31))
    )
    # normalization itself may raise on high<low; catch either path
    bad = None
    for r in rows:
        try:
            b = normalize_record(r, provider="f")
            res = DataQualityEngine().check_bar(b)
            if res.status == QualityStatus.REJECTED and ReasonCode.INVALID_OHLC in res.reasons:
                bad = res
                break
        except ValueError:
            bad = True
            break
    assert bad is not None


def test_duplicate_rejected():
    p = FixtureProvider(scenario="duplicates")
    from idxbot.data.providers.base import HistoricalRequest

    rows = p.get_historical(
        HistoricalRequest(symbols=["ASII"], start=date(2024, 1, 1), end=date(2024, 12, 31))
    )
    bars = normalize_batch(rows, provider="f")
    results = DataQualityEngine().check_series(bars)
    assert any(ReasonCode.DUPLICATE_TIMESTAMP in r.reasons for r in results)
    assert any(r.status == QualityStatus.REJECTED for r in results)


def test_stale_flagged():
    p = FixtureProvider(scenario="stale")
    from idxbot.data.providers.base import HistoricalRequest

    rows = p.get_historical(
        HistoricalRequest(symbols=["BBRI"], start=date(2024, 1, 1), end=date(2024, 1, 31))
    )[:5]
    bars = normalize_batch(rows, provider="f")
    engine = DataQualityEngine(
        freshness_max_age=timedelta(hours=24),
        now=datetime(2024, 6, 1, tzinfo=JAKARTA),
    )
    results = [engine.check_bar(b) for b in bars]
    assert all(ReasonCode.STALE_DATA in r.reasons for r in results)
    assert all(r.status == QualityStatus.FLAGGED for r in results)


def test_abnormal_volume_flagged():
    p = FixtureProvider(scenario="abnormal_volume")
    from idxbot.data.providers.base import HistoricalRequest

    rows = p.get_historical(
        HistoricalRequest(symbols=["INDF"], start=date(2024, 1, 1), end=date(2024, 12, 31))
    )
    bars = normalize_batch(rows, provider="f")
    results = DataQualityEngine(volume_spike_factor=10.0).check_series(bars)
    assert any(ReasonCode.ABNORMAL_VOLUME in r.reasons for r in results)


def test_no_interpolation_policy():
    """Missing candles are flagged, not filled."""
    # Build sparse series manually
    from idxbot.data.normalize import NormalizedBar

    b1 = normalize_record(_bar("X", date(2024, 1, 2), 10, 11, 9, 10, 100))
    b2 = normalize_record(_bar("X", date(2024, 1, 5), 10, 11, 9, 10, 100))  # gap
    results = DataQualityEngine().check_series([b1, b2])
    # series still VALID/FLAGGED but we never invent bars between
    assert len(results) == 2
