"""Ensemble core: weighted avg, diversity, meta weights, OOF, registry, subsets."""

from datetime import date, timedelta
import tempfile
import math

import pytest

from idxbot.data.normalize import normalize_record
from idxbot.data.providers.fixture import _bar
from idxbot.dataset.builder import DatasetBuilder
from idxbot.dataset.labels import LabelGenerator
from idxbot.features import FeatureEngine
from idxbot.ml import ModelTrainer, list_candidates
from idxbot.ensemble import (
    ModelPrediction,
    WeightedAverageEnsemble,
    DiversityMetrics,
    OOFGenerator,
    NonNegativeMetaLearner,
    EnsembleCalibrator,
    EnsembleRegistry,
    EnsembleEngine,
)
from idxbot.ensemble.registry import EnsembleRegistryError


def _rows(n=80, symbol="EN"):
    bars = []
    d = date(2024, 1, 2)
    price = 4500.0
    while len(bars) < n:
        if d.weekday() < 5:
            c = price * (1 + ((len(bars) % 9) - 4) * 0.005)
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


def _mp(mid, down, flat, up, **kw):
    pred = max({"DOWN": down, "FLAT": flat, "UP": up}, key=lambda k: {"DOWN": down, "FLAT": flat, "UP": up}[k])
    return ModelPrediction(
        model_id=mid,
        model_version="v1",
        symbol=kw.get("symbol", "EN"),
        timestamp=kw.get("timestamp", "2024-06-01T16:00:00+07:00"),
        horizon=1,
        probability_down=down,
        probability_flat=flat,
        probability_up=up,
        predicted_class=pred,
        confidence={"DOWN": down, "FLAT": flat, "UP": up}[pred],
    )


def test_weighted_equal_and_custom():
    preds = [
        _mp("a", 0.1, 0.2, 0.7),
        _mp("b", 0.3, 0.3, 0.4),
        _mp("c", 0.2, 0.2, 0.6),
    ]
    ep = WeightedAverageEnsemble().aggregate(preds)
    assert abs(ep.probability_down + ep.probability_flat + ep.probability_up - 1) < 0.01
    assert ep.model_count == 3
    assert ep.disagreement_score >= 0
    # custom weights
    ep2 = WeightedAverageEnsemble({"a": 0.5, "b": 0.5, "c": 0.0}).aggregate(preds)
    assert ep2.weights["c"] == 0.0 or ep2.weights["a"] + ep2.weights["b"] > 0.9


def test_subset_3_and_5():
    rows = _rows(70)
    train, rest = rows[:40], rows[40:]
    trainer = ModelTrainer(42)
    algos = [c.algorithm for c in list_candidates()]
    results = {}
    for a in algos:
        try:
            results[a] = trainer.train(a, train, None, calibrate=False)
        except Exception:
            pass
    assert len(results) >= 3
    # 3-model subset
    sub3 = dict(list(results.items())[:3])
    eng = EnsembleEngine()
    out3 = eng.predict_from_train_results(sub3, rest[:5])
    assert len(out3) == 5
    for e in out3:
        e.validate()
        assert e.model_count == 3
    # 5-model if available
    if len(results) >= 5:
        sub5 = dict(list(results.items())[:5])
        out5 = eng.predict_from_train_results(sub5, rest[:3])
        assert out5[0].model_count == 5


def test_non_negative_weights_sum_one():
    rows = _rows(90)
    with tempfile.TemporaryDirectory() as tmp:
        oof = OOFGenerator(
            n_folds=2,
            min_train=25,
            val_size=12,
            embargo=3,
            algorithms=["logistic_regression", "ridge_classifier", "random_forest"],
            random_seed=42,
        )
        summary = oof.generate(rows, tmp)
        assert summary["n_rows"] > 0
        oof_rows = OOFGenerator.load_oof(tmp)
        meta = NonNegativeMetaLearner(half_life_days=30).fit(oof_rows)
        assert all(v >= -1e-9 for v in meta.weights.values())
        assert abs(sum(meta.weights.values()) - 1.0) < 1e-6


def test_diversity_metrics():
    a = ["UP", "UP", "DOWN", "FLAT"]
    b = ["UP", "DOWN", "DOWN", "FLAT"]
    d = DiversityMetrics.disagreement_rate(a, b)
    assert d == 0.25
    corr = DiversityMetrics.prediction_correlation([0.7, 0.6, 0.2, 0.4], [0.65, 0.55, 0.25, 0.35])
    assert corr > 0.5 or math.isnan(corr)


def test_registry_immutable():
    with tempfile.TemporaryDirectory() as tmp:
        reg = EnsembleRegistry(tmp)
        reg.save("v1", base_models=["a", "b"], weights={"a": 0.5, "b": 0.5}, meta_learner_type="nnls")
        with pytest.raises(EnsembleRegistryError):
            reg.save("v1", base_models=["a"], weights={"a": 1.0}, meta_learner_type="nnls")
        meta = reg.load("v1")
        assert meta["ensemble_version"] == "v1"


def test_probability_normalization_and_confidence():
    preds = [_mp("m1", 0.05, 0.15, 0.80), _mp("m2", 0.10, 0.10, 0.80)]
    ep = WeightedAverageEnsemble().aggregate(preds)
    assert ep.predicted_class == "UP"
    assert abs(ep.confidence - ep.probability_up) < 1e-9
    ep.validate()


def test_deterministic_ensemble():
    rows = _rows(60)
    train, rest = rows[:35], rows[35:45]
    r1 = ModelTrainer(42).train("logistic_regression", train, None, calibrate=False)
    r2 = ModelTrainer(42).train("ridge_classifier", train, None, calibrate=False)
    eng = EnsembleEngine()
    a = eng.predict_from_train_results({"logistic_regression": r1, "ridge_classifier": r2}, rest)
    b = eng.predict_from_train_results({"logistic_regression": r1, "ridge_classifier": r2}, rest)
    for x, y in zip(a, b):
        assert abs(x.probability_up - y.probability_up) < 1e-9
