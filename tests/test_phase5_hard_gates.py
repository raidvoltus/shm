"""
PHASE 5 hard gates: anti-lookahead, anti-leakage, properties, adversarial.
"""

from datetime import date, timedelta
from copy import deepcopy
import math
import random

import pytest

from idxbot.data.normalize import normalize_record
from idxbot.data.providers.fixture import _bar
from idxbot.dataset.builder import DatasetBuilder
from idxbot.dataset.labels import LabelGenerator, LabelClass
from idxbot.dataset.split import (
    PurgedTimeSeriesSplit,
    get_train_class_distribution,
)
from idxbot.features import FeatureEngine
from idxbot.experience import ExperienceStore
import tempfile


def _bars(n=80, symbol="HG"):
    bars = []
    d = date(2024, 1, 2)
    price = 4000.0
    while len(bars) < n:
        if d.weekday() < 5:
            c = price * (1 + ((len(bars) % 11) - 5) * 0.004)
            bars.append(
                normalize_record(
                    _bar(symbol, d, price, max(price, c) * 1.01, min(price, c) * 0.99, c, 2_000_000)
                )
            )
            price = c
        d += timedelta(days=1)
    return bars


def test_A_poison_future_prices_features_unchanged():
    bars = _bars(70)
    eng = FeatureEngine()
    base = eng.transform(bars)
    t = 35
    snap = {k: base[t][k] for k in base[t] if k not in ("symbol", "timestamp")}
    # poison future bar prices
    poisoned = []
    d = date(2024, 1, 2)
    price = 4000.0
    while len(poisoned) < 70:
        if d.weekday() < 5:
            i = len(poisoned)
            if i > t:
                o = c = 9e9
                h, l, vol = 9e9, 8e9, int(9e9)
            else:
                o = price
                c = price * (1 + ((i % 11) - 5) * 0.004)
                h, l = max(o, c) * 1.01, min(o, c) * 0.99
                vol = 2_000_000
                price = c
            poisoned.append(normalize_record(_bar("HG", d, o, h, l, c, vol)))
        d += timedelta(days=1)
    poison_rows = eng.transform(poisoned)
    snap2 = {k: poison_rows[t][k] for k in poison_rows[t] if k not in ("symbol", "timestamp")}
    # compare with nan-safe
    for k in snap:
        a, b = snap[k], snap2[k]
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            continue
        assert a == b, f"feature leakage on {k}"


def test_B_poison_future_vol_threshold_unchanged():
    gen = LabelGenerator(horizons=(5,), threshold_mode="vol", vol_window=15)
    closes = [4000 * (1.001**i) for i in range(60)]
    t = 30
    thr1 = gen.threshold_at(closes, t)
    poisoned = list(closes)
    for i in range(t + 1, 60):
        poisoned[i] = 1e15
    thr2 = gen.threshold_at(poisoned, t)
    assert thr1 == thr2


def test_C_poison_val_labels_train_dist_unchanged():
    bars = _bars(100)
    feats = FeatureEngine().transform(bars)
    closes = [b.close for b in bars]
    ts = [b.timestamp.isoformat() for b in bars]
    gen = LabelGenerator(horizons=(1, 5), threshold_mode="fixed")
    rows, man = DatasetBuilder(gen).build(feats, {"HG": closes}, {"HG": ts})
    rows = DatasetBuilder.training_rows(rows)
    fold = PurgedTimeSeriesSplit(embargo=man.embargo_period, val_ratio=0.25).split(rows)
    d1 = get_train_class_distribution(rows, fold.train_indices)
    poisoned = [dict(r) for r in rows]
    for i in fold.validation_indices:
        poisoned[i] = dict(poisoned[i])
        poisoned[i]["label"] = "DOWN"
    d2 = get_train_class_distribution(poisoned, fold.train_indices)
    assert d1 == d2


def test_E_horizon_increases_embargo():
    assert LabelGenerator(horizons=(1, 5, 20)).embargo_period() == 21
    assert LabelGenerator(horizons=(1, 5, 20, 60)).embargo_period() == 61


def test_F_pending_not_in_training():
    bars = _bars(25)
    feats = FeatureEngine().transform(bars)
    closes = [b.close for b in bars]
    ts = [b.timestamp.isoformat() for b in bars]
    gen = LabelGenerator(horizons=(20,), threshold_mode="fixed")
    rows, _ = DatasetBuilder(gen).build(feats, {"HG": closes}, {"HG": ts})
    assert any(r["label"] == "PENDING" for r in rows)
    train = DatasetBuilder.training_rows(rows)
    assert all(r["label"] != "PENDING" for r in train)


def test_adversarial_future_years():
    """TRAIN early period; poison 'future' bars — train features/labels thresholds stable."""
    bars = _bars(100)
    eng = FeatureEngine()
    base_feats = eng.transform(bars)
    # train boundary at 60
    boundary = 60
    base_snap = [
        {k: base_feats[i][k] for k in ("return_1d", "sma_20", "rsi_14", "feature_valid")}
        for i in range(boundary)
    ]
    # poison after boundary
    poisoned_bars = []
    d = date(2024, 1, 2)
    price = 4000.0
    while len(poisoned_bars) < 100:
        if d.weekday() < 5:
            i = len(poisoned_bars)
            if i >= boundary:
                o = c = 1e11
                h, l, vol = 1e11, 1e10, int(1e11)
            else:
                o = price
                c = price * (1 + ((i % 11) - 5) * 0.004)
                h, l = max(o, c) * 1.01, min(o, c) * 0.99
                vol = 2_000_000
                price = c
            poisoned_bars.append(normalize_record(_bar("HG", d, o, h, l, c, vol)))
        d += timedelta(days=1)
    poison_feats = eng.transform(poisoned_bars)
    for i in range(boundary):
        for k in ("return_1d", "sma_20", "rsi_14", "feature_valid"):
            a, b = base_snap[i][k], poison_feats[i][k]
            if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
                continue
            assert a == b


def test_property_duplicate_append():
    with tempfile.TemporaryDirectory() as tmp:
        store = ExperienceStore(tmp)
        bars = _bars(30)
        feats = FeatureEngine().transform(bars)
        closes = [b.close for b in bars]
        ts = [b.timestamp.isoformat() for b in bars]
        rows, _ = DatasetBuilder(LabelGenerator(horizons=(1,), threshold_mode="fixed")).build(
            feats, {"HG": closes}, {"HG": ts}
        )
        train = DatasetBuilder.training_rows(rows)
        store.append_from_dataset_rows(train)
        c1 = store.count()
        store.append_from_dataset_rows(train)
        assert store.count() == c1


def test_fuzz_100_cases():
    bars = _bars(80)
    feats = FeatureEngine().transform(bars)
    closes = [b.close for b in bars]
    ts = [b.timestamp.isoformat() for b in bars]
    gen = LabelGenerator(horizons=(1, 5, 20), threshold_mode="fixed")
    rows, man = DatasetBuilder(gen).build(feats, {"HG": closes}, {"HG": ts})
    rows = DatasetBuilder.training_rows(rows)
    rng = random.Random(12345)
    failures = 0
    for _ in range(100):
        try:
            vr = rng.uniform(0.15, 0.4)
            fold = PurgedTimeSeriesSplit(embargo=man.embargo_period, val_ratio=vr).split(rows)
            assert set(fold.train_indices).isdisjoint(fold.validation_indices)
            assert fold.embargo_size == man.embargo_period
        except Exception:
            failures += 1
    assert failures == 0
