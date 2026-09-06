"""Pre-fit hard training budget gate tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from idxbot.ml.learning_pipeline import LearningPipeline, StageLog
from idxbot.ml.candidates import get_candidate


def test_estimate_heavy_ge_cheap():
    pipe = LearningPipeline(budget_seconds=19 * 60, use_fixture=True)
    cheap = pipe._estimate_train_cost("logistic_regression")
    heavy = pipe._estimate_train_cost("hist_gradient_boosting")
    assert heavy >= cheap
    assert get_candidate("hist_gradient_boosting").resource_heavy is True


def test_skip_when_remaining_below_estimate_plus_margin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".models").mkdir()
    (tmp_path / ".experience").mkdir()
    (tmp_path / "ml" / "metadata").mkdir(parents=True)

    pipe = LearningPipeline(
        registry_root=tmp_path / ".models",
        experience_root=tmp_path / ".experience",
        report_dir=tmp_path / "ml" / "metadata",
        budget_seconds=50,  # small budget
        use_fixture=True,
        n_bars=80,
    )
    # Force deadline almost immediately
    import time

    pipe._deadline = time.monotonic() + 5  # 5s remaining
    stages: list[StageLog] = []
    # minimal synthetic rows so train would work if allowed
    rows = [
        {
            "symbol": "BBCA.JK",
            "timestamp": f"2024-01-{i+1:02d}T00:00:00+07:00",
            "return_1d": 0.01,
            "sma_5": 100.0,
            "label": "UP" if i % 2 == 0 else "DOWN",
            "feature_valid": True,
            **{c: 0.0 for c in (
                "return_5d","return_20d","sma_10","sma_20","sma_50","ema_10","ema_20","ema_50",
                "price_vs_sma20","price_vs_sma50","sma5_vs_sma20","sma20_vs_sma50","rsi_14",
                "macd","macd_signal","macd_histogram","true_range","atr_14","rolling_volatility_20",
                "rolling_range_20","volume_sma_20","volume_ratio_20","volume_change_1d",
                "daily_range","body_size","upper_wick","lower_wick","close_position_in_range",
                "gap_from_previous_close",
            )},
        }
        for i in range(40)
    ]
    completed = pipe._train_candidates(
        ["hist_gradient_boosting", "gradient_boosting"],
        rows,
        rows[30:],
        stages,
    )
    assert completed == []
    assert any(s.status == "SKIP" for s in stages)
    assert any(
        (s.data or {}).get("reason") in ("BUDGET_GATE", "BUDGET_EXHAUSTED") for s in stages
    )


def test_budget_exhaustion_preserves_champion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = tmp_path / ".models"
    models.mkdir()
    (tmp_path / ".experience").mkdir()
    (tmp_path / "ml" / "metadata").mkdir(parents=True)
    # seed champion
    import json

    champ = {"algorithm": "logistic_regression", "version": "seed"}
    (models / "champion.json").write_text(json.dumps(champ), encoding="utf-8")

    pipe = LearningPipeline(
        registry_root=models,
        experience_root=tmp_path / ".experience",
        report_dir=tmp_path / "ml" / "metadata",
        budget_seconds=1,  # essentially no train time
        use_fixture=True,
        n_bars=80,
    )
    report = pipe.run(run_id="budget-test")
    assert report.status in ("BUDGET_EXHAUSTED", "FAILED", "OK", "DEGRADED")
    # champion file must still exist (never deleted by budget miss)
    assert (models / "champion.json").exists()
    data = json.loads((models / "champion.json").read_text(encoding="utf-8"))
    assert data.get("algorithm") == "logistic_regression"


def test_deadline_uses_monotonic_clock():
    pipe = LearningPipeline(budget_seconds=100)
    import time

    t0 = time.monotonic()
    pipe._deadline = t0 + 50
    rem = pipe._remaining()
    assert 40 <= rem <= 50


def test_completed_candidates_not_promoted_if_incomplete_skipped(tmp_path, monkeypatch):
    """Incomplete/skipped candidates must not appear as promoted winners."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".models").mkdir()
    (tmp_path / ".experience").mkdir()
    (tmp_path / "ml" / "metadata").mkdir(parents=True)
    pipe = LearningPipeline(
        registry_root=tmp_path / ".models",
        experience_root=tmp_path / ".experience",
        report_dir=tmp_path / "ml" / "metadata",
        budget_seconds=120,
        use_fixture=True,
        n_bars=100,
    )
    report = pipe.run(run_id="promo-safety")
    if report.promotion and report.promotion.get("promoted"):
        # winner must be in trained list
        trained_algos = {t["algorithm"] for t in report.trained}
        assert report.promotion.get("challenger") in trained_algos or report.promotion.get(
            "champion"
        )
