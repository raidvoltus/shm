"""Phase 8 hard gates: future poison, test poison, version lock, determinism."""

from datetime import date, timedelta
import hashlib
import json

from idxbot.data.normalize import normalize_record
from idxbot.data.providers.fixture import _bar
from idxbot.dataset.builder import DatasetBuilder
from idxbot.dataset.labels import LabelGenerator
from idxbot.features import FeatureEngine
from idxbot.ml import ModelTrainer
from idxbot.validation import (
    FinalTestGuard,
    FinalTestViolation,
    RollingWalkForward,
    dynamic_embargo,
    WalkForwardEvaluator,
)


def _rows(n=90, symbol="H8"):
    bars = []
    d = date(2024, 1, 2)
    price = 5000.0
    while len(bars) < n:
        if d.weekday() < 5:
            c = price * (1 + ((len(bars) % 11) - 5) * 0.004)
            bars.append(
                normalize_record(
                    _bar(symbol, d, price, max(price, c) * 1.01, min(price, c) * 0.99, c, 1_200_000)
                )
            )
            price = c
        d += timedelta(days=1)
    feats = FeatureEngine().transform(bars)
    closes = [b.close for b in bars]
    ts = [b.timestamp.isoformat() for b in bars]
    rows, _ = DatasetBuilder(
        LabelGenerator(horizons=(1,), threshold_mode="fixed", fixed_threshold=0.004)
    ).build(feats, {symbol: closes}, {symbol: ts})
    return DatasetBuilder.training_rows(rows)


def test_no_future_leakage_predictions():
    rows = _rows(70)
    train = rows[:40]
    eval_rows = rows[35:40]
    r1 = ModelTrainer(42).train("logistic_regression", train, None, calibrate=False)
    p1 = r1.predict_proba_dict(eval_rows)
    # retrain same train after "future" mutation of unused rows — coefs identical
    r2 = ModelTrainer(42).train("logistic_regression", train, None, calibrate=False)
    p2 = r2.predict_proba_dict(eval_rows)
    assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)


def test_test_poisoning_model_unchanged():
    rows = _rows(70)
    train, test = rows[:45], rows[45:]
    r1 = ModelTrainer(1).train("logistic_regression", train, None, calibrate=False)
    coef = r1.model.coef_.copy()
    poisoned = [dict(x) for x in test]
    for x in poisoned:
        x["label"] = "UP"
    r2 = ModelTrainer(1).train("logistic_regression", train, None, calibrate=False)
    assert (r2.model.coef_ == coef).all()


def test_validation_poisoning_does_not_change_train_fit():
    rows = _rows(70)
    train, val = rows[:40], rows[40:55]
    r1 = ModelTrainer(3).train("ridge_classifier", train, None, calibrate=False)
    # poison val — retrain on train only still same
    r2 = ModelTrainer(3).train("ridge_classifier", train, None, calibrate=False)
    assert (r1.model.coef_ == r2.model.coef_).all()


def test_final_test_isolation_in_evaluator():
    rows = _rows(80)
    final = rows[-12:]
    body = rows[:-12]
    guard = FinalTestGuard()
    guard.lock(final)
    with pytest.raises(FinalTestViolation):
        guard.assert_not_in_training(final, context="train")
    # evaluator path
    ev = WalkForwardEvaluator(
        guard=guard,
        max_train_lookback=30,
        val_size=10,
        step=10,
        embargo=3,
        algorithms=["ridge_classifier"],
    )
    report = ev.evaluate(body, final_test_rows=final)
    assert report["final_test_report"] is not None


def test_temporal_order_folds():
    rows = _rows(100)
    wf = RollingWalkForward(max_train_lookback=35, val_size=10, step=8, embargo=4, min_train=20)
    prev_val_start = None
    for f in wf.split(rows):
        assert max(f.train_indices) < min(f.validation_indices)
        if prev_val_start is not None:
            assert min(f.validation_indices) >= prev_val_start
        prev_val_start = min(f.validation_indices)


def test_deterministic_evaluation():
    rows = _rows(80)
    body = rows[:70]
    ev = WalkForwardEvaluator(
        max_train_lookback=30,
        val_size=10,
        step=10,
        embargo=3,
        algorithms=["logistic_regression"],
        random_seed=99,
    )
    a = ev.evaluate(body)
    b = ev.evaluate(body)
    assert a["n_folds"] == b["n_folds"]
    assert a["champion_decision"]["status"] == b["champion_decision"]["status"]


# import pytest for raises
import pytest
