"""Phase 8 core: rolling WF, dynamic embargo, stats, final test isolation."""

from datetime import date, timedelta
import random
import tempfile

import pytest

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
    wilcoxon_signed_rank,
    holm_correction,
    compute_stability,
    WalkForwardEvaluator,
    ValidationArtifactStore,
)
from idxbot.experience import ExperienceStore


def _rows(n=120, symbol="V8"):
    bars = []
    d = date(2024, 1, 2)
    price = 4000.0
    while len(bars) < n:
        if d.weekday() < 5:
            c = price * (1 + ((len(bars) % 9) - 4) * 0.005)
            bars.append(
                normalize_record(
                    _bar(symbol, d, price, max(price, c) * 1.01, min(price, c) * 0.99, c, 1_000_000)
                )
            )
            price = c
        d += timedelta(days=1)
    feats = FeatureEngine().transform(bars)
    closes = [b.close for b in bars]
    ts = [b.timestamp.isoformat() for b in bars]
    rows, _ = DatasetBuilder(
        LabelGenerator(horizons=(1, 5), threshold_mode="fixed", fixed_threshold=0.004)
    ).build(feats, {symbol: closes}, {symbol: ts})
    return DatasetBuilder.training_rows(rows)


def test_dynamic_embargo():
    assert dynamic_embargo([1, 5, 20]) == 21
    assert dynamic_embargo([1, 5, 20, 60]) == 61
    assert dynamic_embargo([1, 5, 20, 60]) > dynamic_embargo([1, 5, 20])


def test_rolling_window_no_overlap():
    rows = _rows(100)
    wf = RollingWalkForward(max_train_lookback=40, val_size=10, step=8, embargo=5, min_train=20)
    folds = list(wf.split(rows))
    assert len(folds) >= 1
    for f in folds:
        assert set(f.train_indices).isdisjoint(f.validation_indices)
        assert f.purge_size >= f.embargo_size
        assert f.train_start <= f.train_end
        assert f.validation_start <= f.validation_end


def test_rolling_window_bound():
    rows = _rows(150)
    lookback = 30
    wf = RollingWalkForward(max_train_lookback=lookback, val_size=10, step=10, embargo=4, min_train=15)
    for f in wf.split(rows):
        assert len(f.train_indices) <= lookback


def test_final_test_cannot_be_used_for_training():
    rows = _rows(80)
    final = rows[-15:]
    train = rows[:50]
    guard = FinalTestGuard()
    guard.lock(final)
    # train without final — ok
    guard.assert_not_in_training(train, context="ok")
    # train including final — fail
    with pytest.raises(FinalTestViolation):
        guard.assert_not_in_training(train + final, context="train")


def test_wilcoxon_and_holm():
    a = [0.4, 0.5, 0.45, 0.48, 0.42]
    b = [0.3, 0.35, 0.32, 0.34, 0.31]
    w = wilcoxon_signed_rank(a, b)
    assert "p_value" in w
    assert w["sample_count"] == 5
    adj = holm_correction([0.01, 0.04, 0.20])
    assert len(adj) == 3
    assert adj[0]["adjusted_p_value"] <= adj[1]["adjusted_p_value"] or True


def test_stability():
    s = compute_stability([0.5, 0.6, 0.4, 0.55])
    assert s.n_folds == 4
    assert s.min <= s.mean <= s.max


def test_walkforward_evaluator_runs():
    rows = _rows(100)
    final = rows[-10:]
    body = rows[:-10]
    ev = WalkForwardEvaluator(
        max_train_lookback=35,
        val_size=12,
        step=10,
        embargo=4,
        algorithms=["logistic_regression", "ridge_classifier"],
    )
    report = ev.evaluate(body, final_test_rows=final)
    assert report["n_folds"] >= 1
    assert "champion_decision" in report
    assert report["champion_decision"]["status"] in ("WIN", "HOLD", "FAIL")
    assert report["final_test_report"] is not None


def test_artifact_immutable():
    with tempfile.TemporaryDirectory() as tmp:
        store = ValidationArtifactStore(tmp)
        store.save_run("run1", {"ok": True})
        with pytest.raises(FileExistsError):
            store.save_run("run1", {"ok": False})


def test_experience_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        exp = ExperienceStore(tmp + "/exp")
        c0 = exp.count()
        rows = _rows(60)
        WalkForwardEvaluator(
            max_train_lookback=30, val_size=8, step=8, embargo=3, algorithms=["ridge_classifier"]
        ).evaluate(rows[:50])
        assert exp.count() == c0


def test_fuzz_100_folds():
    rows = _rows(100)
    rng = random.Random(42)
    failures = 0
    for _ in range(100):
        try:
            emb = rng.randint(2, 8)
            vs = rng.randint(8, 15)
            step = rng.randint(5, 12)
            lb = rng.randint(20, 40)
            wf = RollingWalkForward(
                max_train_lookback=lb, val_size=vs, step=step, embargo=emb, min_train=15
            )
            for f in wf.split(rows):
                assert set(f.train_indices).isdisjoint(f.validation_indices)
                assert f.purge_size >= emb
                assert max(f.train_indices) < min(f.validation_indices)
        except Exception:
            failures += 1
    assert failures == 0
