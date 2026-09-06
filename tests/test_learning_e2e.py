"""End-to-end LEARNING pipeline — real training, not noop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from idxbot.ml.learning_pipeline import LearningPipeline
from idxbot.ml.registry import ModelRegistry
from idxbot.experience.store import ExperienceStore
from idxbot.storage.state import PortfolioState, Position
from idxbot.trading.reset import reset_paper_state


def test_learning_pipeline_trains_and_persists(tmp_path, monkeypatch):
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
        random_seed=42,
    )
    report = pipe.run(run_id="test-learn-1")
    assert report.status in ("OK", "DEGRADED", "BUDGET_EXHAUSTED")
    assert len(report.trained) >= 1
    assert any(s.name == "data" and s.status == "OK" for s in report.stages)
    assert any(s.name.startswith("train_") and s.status == "OK" for s in report.stages)
    # report file written
    reports = list((tmp_path / "ml" / "metadata").glob("learning_report_*.json"))
    assert reports


def test_learning_promotes_or_keeps_champion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".models").mkdir()
    (tmp_path / ".experience").mkdir()
    (tmp_path / "ml" / "metadata").mkdir(parents=True)
    pipe = LearningPipeline(
        registry_root=tmp_path / ".models",
        experience_root=tmp_path / ".experience",
        report_dir=tmp_path / "ml" / "metadata",
        budget_seconds=90,
        use_fixture=True,
        n_bars=100,
    )
    report = pipe.run(run_id="test-learn-2")
    assert report.promotion is not None
    # either promoted or rejected with reason
    assert "promoted" in report.promotion
    assert "reason" in report.promotion


def test_paper_reset_preserves_ml_and_experience(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = tmp_path / ".models"
    exp = tmp_path / ".experience"
    models.mkdir()
    exp.mkdir()
    # fake champion + experience file
    (models / "champion.json").write_text(
        json.dumps({"algorithm": "logistic_regression", "version": "v1"}), encoding="utf-8"
    )
    (exp / "marker.json").write_text("{}", encoding="utf-8")

    state = PortfolioState.create_initial()
    state.account.cash = 1_000_000
    state.account.positions["BBCA.JK"] = Position(
        symbol="BBCA.JK", quantity=100, avg_price=9000.0
    )
    state.trade_history.append({"side": "BUY"})
    state.model_metadata = {"champion": "logistic_regression:v1"}

    new_state = reset_paper_state(state)
    assert new_state.account.cash == 10_000_000
    assert new_state.account.positions == {}
    assert new_state.model_metadata.get("champion") == "logistic_regression:v1"
    assert (models / "champion.json").exists()
    assert (exp / "marker.json").exists()


def test_colab_unavailable_local_fallback():
    from idxbot.ml.providers import ColabComputeProvider, LocalComputeProvider

    colab = ColabComputeProvider(enabled=False)
    assert colab.availability() is False
    local = LocalComputeProvider()
    assert local.availability() is True


def test_invalid_colab_artifact_rejected(tmp_path):
    from idxbot.ml.providers.colab import validate_colab_artifact, ColabArtifactValidationError

    with pytest.raises(ColabArtifactValidationError):
        validate_colab_artifact({"exec": "bad"})
    with pytest.raises(ColabArtifactValidationError):
        validate_colab_artifact(
            {"algorithm": "x", "job_id": "1", "metrics": {}},
            artifact_path="../secret.pkl",
        )
