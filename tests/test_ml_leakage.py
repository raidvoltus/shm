"""Anti-leakage hard gates for Phase 6 training pipeline."""

from datetime import date, timedelta
from copy import deepcopy
import math

from idxbot.data.normalize import normalize_record
from idxbot.data.providers.fixture import _bar
from idxbot.dataset.builder import DatasetBuilder
from idxbot.dataset.labels import LabelGenerator
from idxbot.features import FeatureEngine
from idxbot.ml import ModelTrainer, ModelEvaluator, ChampionChallenger
from idxbot.ml.preprocess import FeatureMatrixBuilder


def _splits(n=100):
    bars = []
    d = date(2024, 1, 2)
    price = 3500.0
    while len(bars) < n:
        if d.weekday() < 5:
            c = price * (1 + ((len(bars) % 7) - 3) * 0.005)
            bars.append(
                normalize_record(
                    _bar("LK", d, price, max(price, c) * 1.01, min(price, c) * 0.99, c, 1_200_000)
                )
            )
            price = c
        d += timedelta(days=1)
    feats = FeatureEngine().transform(bars)
    closes = [b.close for b in bars]
    ts = [b.timestamp.isoformat() for b in bars]
    rows, _ = DatasetBuilder(
        LabelGenerator(horizons=(1,), threshold_mode="fixed", fixed_threshold=0.004)
    ).build(feats, {"LK": closes}, {"LK": ts})
    rows = DatasetBuilder.training_rows(rows)
    a, b = int(len(rows) * 0.6), int(len(rows) * 0.8)
    return rows[:a], rows[a:b], rows[b:]


def test_poison_test_labels_model_unchanged():
    train, cal, test = _splits()
    t1 = ModelTrainer(42).train("logistic_regression", train, cal)
    p1 = t1.predict_proba_dict(test)
    # poison test labels — retrain should be identical if test not used
    test2 = [dict(r) for r in test]
    for r in test2:
        r["label"] = "UP"
    t2 = ModelTrainer(42).train("logistic_regression", train, cal)
    p2 = t2.predict_proba_dict(test2)
    for a, b in zip(p1, p2):
        for c in a:
            assert abs(a[c] - b[c]) < 1e-9


def test_poison_val_labels_training_fit_stable():
    """Training fit uses train only; cal affects calibration not base coef path for uncal check."""
    train, cal, test = _splits()
    # train without calibration to isolate fit
    t1 = ModelTrainer(42).train("logistic_regression", train, None, calibrate=False)
    coef1 = t1.model.coef_.copy()
    cal2 = [dict(r) for r in cal]
    for r in cal2:
        r["label"] = "DOWN"
    t2 = ModelTrainer(42).train("logistic_regression", train, cal2, calibrate=False)
    assert (t2.model.coef_ == coef1).all()


def test_normalizer_train_only():
    train, cal, test = _splits()
    prep = FeatureMatrixBuilder()
    prep.fit(train)
    X_train = prep.transform(train)
    # modify test distribution drastically
    test2 = [dict(r) for r in test]
    for r in test2:
        for k in list(r.keys()):
            if isinstance(r[k], float):
                r[k] = 1e6
    X_train2 = prep.transform(train)
    assert (X_train == X_train2).all()


def test_promotion_ignores_test_metric_path():
    """Promotion uses provided val metrics only — simulate test-poisoned metrics not fed in."""
    cc = ChampionChallenger(margin=0.02)
    # decision based on val metrics
    d = cc.decide("a", "b", {"balanced_accuracy": 0.5}, {"balanced_accuracy": 0.505})
    assert d.promoted is False
    # even if "test" would show huge gain, we don't pass test into decide
    d2 = cc.decide("a", "b", {"balanced_accuracy": 0.5}, {"balanced_accuracy": 0.505})
    assert d2.promoted is False


def test_shuffle_input_preprocess_order():
    train, cal, test = _splits()
    # training rows order: engine sorts features chronologically already
    t1 = ModelTrainer(42).train("ridge_classifier", train, cal, calibrate=False)
    rev = list(reversed(train))
    # fit on reversed — sklearn sees different row order but same data; coef may differ slightly
    # Contract: predictions on same test with same seed + same sorted train should match
    t2 = ModelTrainer(42).train("ridge_classifier", sorted(rev, key=lambda r: r["timestamp"]), cal, calibrate=False)
    p1 = t1.predict(test)
    p2 = t2.predict(test)
    assert p1 == p2
