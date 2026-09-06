"""Model pool: 7 candidates, train, proba, metrics, determinism, registry."""

from datetime import date, timedelta
import tempfile
import math

import pytest

from idxbot.data.normalize import normalize_record
from idxbot.data.providers.fixture import _bar
from idxbot.dataset.builder import DatasetBuilder
from idxbot.dataset.labels import LabelGenerator
from idxbot.features import FeatureEngine
from idxbot.ml import (
    list_candidates,
    get_candidate,
    ModelTrainer,
    ModelEvaluator,
    ModelRegistry,
    ChampionChallenger,
    CLASS_ORDER,
)
from idxbot.ml.registry import ModelRegistryError


def _ml_rows(n=100, symbol="ML"):
    bars = []
    d = date(2024, 1, 2)
    price = 5000.0
    while len(bars) < n:
        if d.weekday() < 5:
            c = price * (1 + ((len(bars) % 9) - 4) * 0.006)
            bars.append(
                normalize_record(
                    _bar(symbol, d, price, max(price, c) * 1.01, min(price, c) * 0.99, c, 1_500_000)
                )
            )
            price = c
        d += timedelta(days=1)
    feats = FeatureEngine().transform(bars)
    closes = [b.close for b in bars]
    ts = [b.timestamp.isoformat() for b in bars]
    rows, man = DatasetBuilder(
        LabelGenerator(horizons=(1,), threshold_mode="fixed", fixed_threshold=0.005)
    ).build(feats, {symbol: closes}, {symbol: ts})
    train_rows = DatasetBuilder.training_rows(rows)
    # chronological split
    n = len(train_rows)
    a, b = int(n * 0.6), int(n * 0.8)
    return train_rows[:a], train_rows[a:b], train_rows[b:], man


def test_seven_candidates():
    cands = list_candidates()
    assert len(cands) == 7
    names = {c.algorithm for c in cands}
    assert "logistic_regression" in names
    assert "regime_aware_logreg" in names
    assert get_candidate("logistic_regression").notes


def test_train_all_candidates_proba():
    train, cal, test, _ = _ml_rows(90)
    trainer = ModelTrainer(random_seed=42)
    evaluator = ModelEvaluator()
    for cand in list_candidates():
        result = trainer.train(cand.algorithm, train, cal, calibrate=True)
        probs = result.predict_proba_dict(test[:5])
        assert len(probs) == 5
        for p in probs:
            assert set(p.keys()) == set(CLASS_ORDER)
            s = sum(p.values())
            assert abs(s - 1.0) < 0.05
            assert all(0 <= v <= 1.01 for v in p.values())
        metrics = evaluator.evaluate(result, test)
        assert metrics.n_samples == len(test)
        assert 0 <= metrics.accuracy <= 1


def test_deterministic_dual_run():
    train, cal, test, _ = _ml_rows(70)
    t1 = ModelTrainer(42).train("logistic_regression", train, cal)
    t2 = ModelTrainer(42).train("logistic_regression", train, cal)
    p1 = t1.predict_proba_dict(test)
    p2 = t2.predict_proba_dict(test)
    for a, b in zip(p1, p2):
        for c in CLASS_ORDER:
            assert abs(a[c] - b[c]) < 1e-9


def test_registry_sha_and_immutable():
    train, cal, test, man = _ml_rows(70)
    result = ModelTrainer(42).train("logistic_regression", train, cal)
    metrics = ModelEvaluator().evaluate(result, test).to_dict()
    with tempfile.TemporaryDirectory() as tmp:
        reg = ModelRegistry(tmp)
        meta = reg.save(
            result,
            version="v1",
            metrics=metrics,
            dataset_version=man.dataset_version,
            inference_time_ms=metrics["inference_time_ms"],
        )
        assert meta.artifact_sha256
        assert meta.artifact_size_bytes > 0
        payload, meta2 = reg.load("logistic_regression", "v1")
        assert meta2.artifact_sha256 == meta.artifact_sha256
        # immutable
        with pytest.raises(ModelRegistryError):
            reg.save(result, version="v1", metrics=metrics)


def test_corrupt_artifact_safe_failure():
    train, cal, test, _ = _ml_rows(60)
    result = ModelTrainer(42).train("ridge_classifier", train, cal)
    with tempfile.TemporaryDirectory() as tmp:
        reg = ModelRegistry(tmp)
        reg.save(result, version="v1", metrics={"accuracy": 0.5})
        path = reg._dir("ridge_classifier", "v1") / "model.joblib"
        path.write_bytes(b"corrupt-not-a-model")
        with pytest.raises(ModelRegistryError, match="SHA256"):
            reg.load("ridge_classifier", "v1")


def test_feature_version_mismatch():
    train, cal, test, _ = _ml_rows(60)
    result = ModelTrainer(42).train("logistic_regression", train, cal, feature_version="1")
    with tempfile.TemporaryDirectory() as tmp:
        reg = ModelRegistry(tmp)
        reg.save(result, version="v1", metrics={})
        with pytest.raises(ModelRegistryError, match="Feature version"):
            reg.load("logistic_regression", "v1", expected_feature_version="99.0")


def test_promotion_margin():
    cc = ChampionChallenger(margin=0.02)
    d = cc.decide(
        "logreg:v1",
        "rf:v1",
        {"balanced_accuracy": 0.50},
        {"balanced_accuracy": 0.51},
    )
    assert d.promoted is False
    assert d.reason == "improvement_below_margin"
    d2 = cc.decide(
        "logreg:v1",
        "rf:v2",
        {"balanced_accuracy": 0.50},
        {"balanced_accuracy": 0.55},
    )
    assert d2.promoted is True


def test_n_jobs_is_one():
    # Contract: never parallelize. Linear solvers may leave n_jobs=None
    # (single-threaded); tree ensembles must force n_jobs=1.
    for cand in list_candidates():
        est = cand.create(0)
        n_jobs = getattr(est, "n_jobs", 1)
        assert n_jobs in (1, None), f"{cand.algorithm} n_jobs={n_jobs!r} must be 1 or None"
        if cand.hyperparams.get("n_jobs") is not None:
            assert est.n_jobs == 1, f"{cand.algorithm} advertised n_jobs=1 but got {est.n_jobs}"
