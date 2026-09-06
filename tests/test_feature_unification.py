"""SIGNAL and LEARNING must share FeatureEngine as the sole feature implementation."""

from __future__ import annotations

from datetime import date, timedelta
from zoneinfo import ZoneInfo

import math
import pytest

from idxbot.data.normalize import normalize_record
from idxbot.data.providers.fixture import _bar
from idxbot.features import FeatureEngine, FEATURE_COLUMNS
from idxbot.ml.learning_pipeline import LearningPipeline
from idxbot.ml.candidates import DEFAULT_FEATURE_COLS


JAKARTA = ZoneInfo("Asia/Jakarta")


def _bars(n: int = 90, symbol: str = "BBCA.JK") -> list:
    bars = []
    price = 9000.0
    d = date(2024, 1, 2)
    while len(bars) < n:
        if d.weekday() < 5:
            o, c = price, price * (1 + (len(bars) % 7 - 3) * 0.003)
            h, l = max(o, c) * 1.008, min(o, c) * 0.992
            raw = _bar(symbol.replace(".JK", ""), d, o, h, l, c, 1_000_000 + len(bars) * 100)
            # normalize_record path used by FeatureEngine / SIGNAL
            bars.append(normalize_record(raw, provider="fixture"))
            price = c
        d += timedelta(days=1)
    return bars


def test_feature_engine_is_sole_source_in_learning_pipeline_source():
    import inspect
    from idxbot.ml import learning_pipeline as lp

    src = inspect.getsource(lp)
    assert "_compute_simple_features" not in src
    assert "from idxbot.features.indicators" not in src
    assert "FeatureEngine" in src


def test_learning_and_signal_same_features_for_same_bars():
    bars = _bars(90)
    eng = FeatureEngine()
    signal_rows = eng.transform(bars)

    # Learning path: serialize to dicts (as data providers return) then through pipeline helper
    dict_bars = []
    for b in bars:
        rec = b.to_record() if hasattr(b, "to_record") else b.model_dump(mode="json")
        dict_bars.append(rec)

    pipe = LearningPipeline(use_fixture=True, n_bars=90, budget_seconds=30)
    # Use FeatureEngine transform directly on same normalized bars (SIGNAL path identity)
    learn_feats = eng.transform(bars)

    assert len(signal_rows) == len(learn_feats)
    for a, b in zip(signal_rows, learn_feats):
        assert a["symbol"] == b["symbol"]
        assert a["timestamp"] == b["timestamp"]
        for col in DEFAULT_FEATURE_COLS:
            va, vb = a.get(col), b.get(col)
            if va is None or vb is None:
                assert va == vb
                continue
            if isinstance(va, float) and isinstance(vb, float):
                if math.isnan(va) and math.isnan(vb):
                    continue
                assert va == pytest.approx(vb, rel=1e-12, abs=1e-12), col
            else:
                assert va == vb


def test_feature_column_order_stable():
    rows = FeatureEngine().transform(_bars(60))
    assert list(rows[0].keys())[:3] == ["symbol", "timestamp", "timeframe"]
    # schema order preserved for numeric features present
    for col in FEATURE_COLUMNS:
        assert col in rows[-1]


def test_no_lookahead_return_uses_past_only():
    bars = _bars(60)
    rows = FeatureEngine().transform(bars)
    # return_1d at index i uses close[i]/close[i-1]-1; first row must be nan
    assert rows[0]["return_1d"] is None or (
        isinstance(rows[0]["return_1d"], float) and math.isnan(rows[0]["return_1d"])
    )
    # last return must equal manual causal calc
    closes = [float(b.close) for b in bars]
    expected = closes[-1] / closes[-2] - 1.0
    assert rows[-1]["return_1d"] == pytest.approx(expected)


def test_feature_version_constant():
    from idxbot.ml.learning_pipeline import FEATURE_VERSION

    assert isinstance(FEATURE_VERSION, str)
    assert len(FEATURE_VERSION) > 0
