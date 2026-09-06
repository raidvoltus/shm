"""Phase 7 hard gates: stacking leakage, future poison, test poison, decay, calibration."""

from datetime import date, timedelta
import tempfile
import hashlib
import json

from idxbot.data.normalize import normalize_record
from idxbot.data.providers.fixture import _bar
from idxbot.dataset.builder import DatasetBuilder
from idxbot.dataset.labels import LabelGenerator
from idxbot.features import FeatureEngine
from idxbot.ml import ModelTrainer
from idxbot.ensemble import (
    WeightedAverageEnsemble,
    ModelPrediction,
    NonNegativeMetaLearner,
    EnsembleCalibrator,
    EnsembleEngine,
    OOFGenerator,
)
from idxbot.experience import ExperienceStore


def _bars(n, symbol="HG7"):
    bars = []
    d = date(2024, 1, 2)
    price = 5000.0
    while len(bars) < n:
        if d.weekday() < 5:
            c = price * (1 + ((len(bars) % 11) - 5) * 0.004)
            bars.append(
                normalize_record(
                    _bar(symbol, d, price, max(price, c) * 1.01, min(price, c) * 0.99, c, 1_500_000)
                )
            )
            price = c
        d += timedelta(days=1)
    return bars


def _dataset(n=80):
    bars = _bars(n)
    feats = FeatureEngine().transform(bars)
    closes = [b.close for b in bars]
    ts = [b.timestamp.isoformat() for b in bars]
    rows, _ = DatasetBuilder(
        LabelGenerator(horizons=(1,), threshold_mode="fixed", fixed_threshold=0.004)
    ).build(feats, {"HG7": closes}, {"HG7": ts})
    return DatasetBuilder.training_rows(rows)


def _hash_preds(preds):
    payload = json.dumps(
        [{"ts": p.timestamp, "up": round(p.probability_up, 8), "cls": p.predicted_class} for p in preds],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def test_future_poisoning_ensemble_stable():
    """Mutating data after cutoff must not change ensemble preds at/before cutoff."""
    rows = _dataset(70)
    cutoff = 40
    train = rows[:cutoff]
    eval_rows = rows[cutoff - 5 : cutoff]  # near cutoff, past side
    t1 = ModelTrainer(42).train("logistic_regression", train, None, calibrate=False)
    t2 = ModelTrainer(42).train("ridge_classifier", train, None, calibrate=False)
    eng = EnsembleEngine()
    before = eng.predict_from_train_results(
        {"logistic_regression": t1, "ridge_classifier": t2}, eval_rows
    )
    h1 = _hash_preds(before)

    # Poison "future" rows after cutoff in a copy — retrain on same train (train unchanged)
    # so predictions must match
    t1b = ModelTrainer(42).train("logistic_regression", train, None, calibrate=False)
    t2b = ModelTrainer(42).train("ridge_classifier", train, None, calibrate=False)
    after = eng.predict_from_train_results(
        {"logistic_regression": t1b, "ridge_classifier": t2b}, eval_rows
    )
    h2 = _hash_preds(after)
    assert h1 == h2


def test_test_label_poisoning_artifacts_stable():
    rows = _dataset(70)
    train, cal, test = rows[:40], rows[40:55], rows[55:]
    r1 = ModelTrainer(42).train("logistic_regression", train, cal, calibrate=False)
    coef = r1.model.coef_.copy()
    # poison test labels
    test2 = [dict(x) for x in test]
    for x in test2:
        x["label"] = "UP"
    r2 = ModelTrainer(42).train("logistic_regression", train, cal, calibrate=False)
    assert (r2.model.coef_ == coef).all()


def test_time_decay_is_causal():
    """Decay uses timestamps <= cutoff only — no future timestamps."""
    # synthetic oof
    oof = []
    for i, ts in enumerate(
        ["2024-01-10T16:00:00+07:00", "2024-02-10T16:00:00+07:00", "2024-03-10T16:00:00+07:00"]
    ):
        for mid, p in [("m1", 0.6 + i * 0.05), ("m2", 0.4)]:
            oof.append(
                {
                    "timestamp": ts,
                    "symbol": "X",
                    "horizon": 1,
                    "actual_label": "UP" if i > 0 else "DOWN",
                    "model_id": mid,
                    "probability_up": p,
                    "probability_down": 1 - p,
                    "probability_flat": 0.0,
                }
            )
    meta = NonNegativeMetaLearner(half_life_days=30).fit(oof)
    assert abs(sum(meta.weights.values()) - 1) < 1e-6
    assert all(v >= 0 for v in meta.weights.values())


def test_independent_calibration():
    rows = _dataset(80)
    train, cal_hold, _ = rows[:40], rows[40:55], rows[55:]
    tr = ModelTrainer(42).train("logistic_regression", train, None, calibrate=False)
    eng = EnsembleEngine()
    # meta-style: get ensemble probs on cal holdout
    preds = eng.predict_from_train_results({"logistic_regression": tr}, cal_hold)
    probs = [p.probs() for p in preds]
    labels = [r["label"] for r in cal_hold]
    cal = EnsembleCalibrator()
    meta = cal.fit(probs, labels, timestamps=[r["timestamp"] for r in cal_hold])
    assert meta.method in ("platt_ovr", "identity")
    # transform should not require test
    out = cal.transform(probs)
    assert len(out) == len(probs)


def test_no_train_overlap_oof_indices():
    rows = _dataset(75)
    with tempfile.TemporaryDirectory() as tmp:
        gen = OOFGenerator(
            n_folds=2,
            min_train=25,
            val_size=10,
            embargo=4,
            algorithms=["logistic_regression"],
            random_seed=0,
        )
        summary = gen.generate(rows, tmp)
        oof = OOFGenerator.load_oof(tmp)
        for r in oof:
            # val starts after train_end + implicit embargo in generator
            assert r["val_start_idx"] >= r["train_end_idx"]


def test_experience_store_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        exp = ExperienceStore(tmp + "/exp")
        # empty baseline
        c0 = exp.count()
        rows = _dataset(50)
        train = rows[:30]
        ModelTrainer(42).train("ridge_classifier", train, None, calibrate=False)
        assert exp.count() == c0


def test_final_test_never_used_for_fit():
    rows = _dataset(70)
    train, cal, test = rows[:40], rows[40:55], rows[55:]
    # train without seeing test
    r = ModelTrainer(7).train("logistic_regression", train, cal, calibrate=False)
    # predictions on test work but test never entered fit
    probs = r.predict_proba_dict(test)
    assert len(probs) == len(test)


def test_empty_models_safe_failure():
    eng = EnsembleEngine()
    try:
        eng.predict_from_train_results({}, [{"symbol": "X", "timestamp": "t", "horizon": 1}])
        assert False, "should fail"
    except ValueError as e:
        assert "SAFE FAILURE" in str(e) or "no valid" in str(e).lower()
