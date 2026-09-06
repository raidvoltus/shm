"""Self-learning: no auto-promote without all gates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from idxbot.self_learning.loop import SelfLearningLoop
from idxbot.signals.order_intent import create_order_intent


def test_prediction_persisted(tmp_path):
    loop = SelfLearningLoop(tmp_path / "exp")
    oi = create_order_intent(
        symbol="BBCA.JK",
        timestamp=datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc),
        intent="BUY",
        confidence=0.7,
        governor_state="FULL_7",
        feature_version="features-v1",
        model_version="m1",
    )
    n = loop.persist_predictions([oi])
    assert n >= 1


def test_outcome_not_attached_before_horizon(tmp_path):
    loop = SelfLearningLoop(tmp_path / "exp", default_horizon_days=5)
    now = datetime(2026, 8, 2, tzinfo=timezone.utc)
    # prediction yesterday — horizon 5 days not elapsed
    oi = create_order_intent(
        symbol="BBCA.JK",
        timestamp=datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc),
        intent="BUY",
        confidence=0.7,
        governor_state="FULL_7",
    )
    loop.persist_predictions([oi])
    resolved = loop.resolve_pending(now=now, price_lookup={"BBCA.JK": 7000.0})
    assert resolved == 0


def test_candidate_rejected_on_gate_failure(tmp_path):
    loop = SelfLearningLoop(tmp_path / "exp")
    dec = loop.evaluate_promotion(
        champion_id="logreg:v1",
        challenger_id="rf:v2",
        champion_metrics={"balanced_accuracy": 0.55},
        candidate_metrics={"balanced_accuracy": 0.70},
        anti_leakage_pass=False,  # fail
        walk_forward_pass=True,
        adversarial_pass=True,
        stability_pass=True,
    )
    assert dec.promoted is False
    assert "GATE" in dec.reason or dec.promoted is False


def test_candidate_promoted_only_all_gates_pass(tmp_path):
    loop = SelfLearningLoop(tmp_path / "exp")
    dec = loop.evaluate_promotion(
        champion_id="logreg:v1",
        challenger_id="rf:v2",
        champion_metrics={"balanced_accuracy": 0.50},
        candidate_metrics={"balanced_accuracy": 0.60},
        anti_leakage_pass=True,
        walk_forward_pass=True,
        adversarial_pass=True,
        stability_pass=True,
        baseline_pass=True,
    )
    # margin 0.02 → 0.60 > 0.50+0.02 → should promote
    assert dec.promoted is True
