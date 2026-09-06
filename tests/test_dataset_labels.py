"""Label generator: multi-horizon, PENDING, backward-looking threshold."""

from datetime import date, timedelta
import math

import pytest

from idxbot.data.normalize import normalize_record
from idxbot.data.providers.fixture import _bar
from idxbot.dataset.labels import LabelGenerator, LabelClass
from idxbot.dataset.builder import DatasetBuilder
from idxbot.features import FeatureEngine


def _closes(n=80, start=1000.0):
    closes = []
    price = start
    for i in range(n):
        price = price * (1 + ((i % 9) - 4) * 0.003)
        closes.append(price)
    return closes


def _timestamps(n=80):
    d = date(2024, 1, 2)
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(f"{d.isoformat()}T16:00:00+07:00")
        d += timedelta(days=1)
    return out


def test_multi_horizon_and_pending():
    gen = LabelGenerator(horizons=(1, 5, 20), threshold_mode="fixed", fixed_threshold=0.01)
    closes = _closes(30)
    ts = _timestamps(30)
    labels = gen.label_symbol("X", ts, closes)
    by_h = {}
    for lab in labels:
        by_h.setdefault(lab.horizon, []).append(lab)
    assert set(by_h) == {1, 5, 20}
    # last 20 for horizon 20 must be PENDING
    pending_20 = [l for l in by_h[20] if l.label == LabelClass.PENDING]
    assert len(pending_20) >= 20
    # early ones should be classified
    classified = [l for l in by_h[1] if l.label != LabelClass.PENDING]
    assert len(classified) >= 25


def test_threshold_backward_looking_poison():
    """Mutating future closes must not change threshold at T."""
    gen = LabelGenerator(horizons=(5,), threshold_mode="vol", vol_window=10, vol_multiplier=1.0)
    closes = _closes(40)
    ts = _timestamps(40)
    t_idx = 20
    thr1 = gen.threshold_at(closes, t_idx)
    # poison future
    poisoned = list(closes)
    for i in range(t_idx + 1, len(poisoned)):
        poisoned[i] = 1e12
    thr2 = gen.threshold_at(poisoned, t_idx)
    assert thr1 == thr2 or (math.isnan(thr1) and math.isnan(thr2))


def test_embargo_dynamic():
    g1 = LabelGenerator(horizons=(1, 5, 20))
    assert g1.embargo_period() == 21
    g2 = LabelGenerator(horizons=(1, 5, 20, 60))
    assert g2.embargo_period() == 61
    assert g2.embargo_period() > g1.embargo_period()


def test_dataset_builder_deterministic():
    # Build mini feature rows
    bars = []
    d = date(2024, 1, 2)
    price = 1000.0
    while len(bars) < 60:
        if d.weekday() < 5:
            c = price * (1 + (len(bars) % 5 - 2) * 0.004)
            bars.append(normalize_record(_bar("DS", d, price, c * 1.01, c * 0.99, c, 1_000_000)))
            price = c
        d += timedelta(days=1)
    feats = FeatureEngine().transform(bars)
    closes = [b.close for b in bars]
    ts = [b.timestamp.isoformat() for b in bars]
    builder = DatasetBuilder(LabelGenerator(horizons=(1, 5), threshold_mode="fixed"))
    rows1, man1 = builder.build(feats, {"DS": closes}, {"DS": ts})
    rows2, man2 = builder.build(feats, {"DS": closes}, {"DS": ts})
    assert DatasetBuilder.deterministic_hash(rows1) == DatasetBuilder.deterministic_hash(rows2)
    assert man1.row_count == man2.row_count
    train = DatasetBuilder.training_rows(rows1)
    assert all(r["label"] != "PENDING" for r in train)
    assert man1.embargo_period == 6
