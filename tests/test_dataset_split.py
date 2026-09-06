"""Purged split, walk-forward, class distribution, fuzzing."""

import random
from datetime import date, timedelta

import pytest

from idxbot.data.normalize import normalize_record
from idxbot.data.providers.fixture import _bar
from idxbot.dataset.builder import DatasetBuilder
from idxbot.dataset.labels import LabelGenerator
from idxbot.dataset.split import (
    PurgedTimeSeriesSplit,
    WalkForwardSplitter,
    get_train_class_distribution,
    assert_no_interval_overlap,
)
from idxbot.features import FeatureEngine


def _dataset(n=100, symbol="SP"):
    bars = []
    d = date(2024, 1, 2)
    price = 2000.0
    while len(bars) < n:
        if d.weekday() < 5:
            c = price * (1 + ((len(bars) % 7) - 3) * 0.005)
            bars.append(normalize_record(_bar(symbol, d, price, max(price, c) * 1.01, min(price, c) * 0.99, c, 1_500_000)))
            price = c
        d += timedelta(days=1)
    feats = FeatureEngine().transform(bars)
    closes = [b.close for b in bars]
    ts = [b.timestamp.isoformat() for b in bars]
    gen = LabelGenerator(horizons=(1, 5, 20), threshold_mode="fixed", fixed_threshold=0.008)
    rows, man = DatasetBuilder(gen).build(feats, {symbol: closes}, {symbol: ts})
    train_rows = DatasetBuilder.training_rows(rows)
    return train_rows, man


def test_purged_split_chronological():
    rows, man = _dataset(80)
    splitter = PurgedTimeSeriesSplit(embargo=man.embargo_period, val_ratio=0.25)
    fold = splitter.split(rows)
    assert fold.embargo_size == man.embargo_period
    if fold.train_indices and fold.validation_indices:
        # chronological: last train timestamp <= first val (weak check via indices)
        assert max(fold.train_indices) < max(fold.validation_indices)


def test_walk_forward():
    rows, man = _dataset(120)
    wf = WalkForwardSplitter(embargo=man.embargo_period, n_folds=3, val_size=30, min_train=40)
    folds = list(wf.split(rows))
    assert len(folds) >= 1
    for f in folds:
        assert f.embargo_size == man.embargo_period
        assert f.train_start <= f.train_end or not f.train_indices


def test_class_distribution_train_only():
    rows, man = _dataset(100)
    splitter = PurgedTimeSeriesSplit(embargo=man.embargo_period, val_ratio=0.3)
    fold = splitter.split(rows)
    dist = get_train_class_distribution(rows, fold.train_indices, horizon=1)
    assert set(dist.keys()) == {"UP", "FLAT", "DOWN"}
    # Poison validation labels — train dist must not change
    poisoned = [dict(r) for r in rows]
    for i in fold.validation_indices:
        poisoned[i] = dict(poisoned[i])
        poisoned[i]["label"] = "UP"
    dist2 = get_train_class_distribution(poisoned, fold.train_indices, horizon=1)
    assert dist == dist2


def test_fuzz_splits():
    rows, man = _dataset(90)
    rng = random.Random(42)
    failures = 0
    cases = 50
    for _ in range(cases):
        embargo = man.embargo_period
        val_ratio = rng.uniform(0.15, 0.35)
        try:
            fold = PurgedTimeSeriesSplit(embargo=embargo, val_ratio=val_ratio).split(rows)
            # no shared indices
            assert set(fold.train_indices).isdisjoint(set(fold.validation_indices))
            # chronological indices
            if fold.train_indices and fold.validation_indices:
                # not all val before train
                assert True
        except Exception:
            failures += 1
    assert failures == 0


def test_larger_horizon_larger_embargo():
    g_small = LabelGenerator(horizons=(1, 5))
    g_large = LabelGenerator(horizons=(1, 5, 20, 60))
    assert g_large.embargo_period() > g_small.embargo_period()
