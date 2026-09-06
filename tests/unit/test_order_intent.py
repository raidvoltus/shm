"""Unit tests for OrderIntent contract and determinism."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from idxbot.signals.order_intent import create_order_intent, make_signal_id
from idxbot.signals.hysteresis import apply_hysteresis
from idxbot.signals.multi_horizon import resolve_horizons
from idxbot.signals.signal_engine import SignalEngine


def test_signal_id_deterministic():
    ts = datetime(2026, 9, 1, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta"))
    a = make_signal_id("BBCA.JK", ts, "fv1", "mv1", "BUY")
    b = make_signal_id("BBCA.JK", ts, "fv1", "mv1", "BUY")
    assert a == b
    assert len(a) == 64


def test_signal_id_changes_with_inputs():
    ts = datetime(2026, 9, 1, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta"))
    a = make_signal_id("BBCA.JK", ts, "fv1", "mv1")
    b = make_signal_id("BBRI.JK", ts, "fv1", "mv1")
    c = make_signal_id("BBCA.JK", ts, "fv2", "mv1")
    assert a != b
    assert a != c


def test_create_order_intent_immutable_fields():
    ts = datetime(2026, 9, 1, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta"))
    oi = create_order_intent(
        symbol="bbca.jk",
        timestamp=ts,
        intent="BUY",
        confidence=0.7215,
        governor_state="DEGRADED_5",
        active_models=5,
        feature_version="fv1",
        model_version="mv1",
    )
    assert oi.symbol == "BBCA.JK"
    assert oi.intent == "BUY"
    assert 0.72 < oi.confidence < 0.73
    assert oi.governor_state == "DEGRADED_5"
    d = oi.to_dict()
    assert "signal_id" in d
    assert d["intent"] == "BUY"


def test_hysteresis_prevents_flip():
    assert apply_hysteresis(0.61, "HOLD") == "BUY"
    assert apply_hysteresis(0.55, "BUY") == "BUY"  # still above exit
    assert apply_hysteresis(0.50, "BUY") == "HOLD"
    assert apply_hysteresis(0.39, "HOLD") == "SELL"
    assert apply_hysteresis(0.45, "SELL") == "SELL"
    assert apply_hysteresis(0.50, "SELL") == "HOLD"


def test_multi_horizon_conflict():
    intent, reason = resolve_horizons(0.30, 0.50, 0.70)  # 1D DOWN, 20D UP
    assert intent == "HOLD"
    assert "CONFLICT" in reason

    intent2, reason2 = resolve_horizons(0.70, 0.65, 0.80)
    assert intent2 == "BUY"

    intent3, reason3 = resolve_horizons(0.30, 0.35, 0.20)
    assert intent3 == "SELL"


def test_signal_engine_safe_exit():
    eng = SignalEngine()
    ts = datetime(2026, 9, 1, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta"))
    oi = eng.generate(
        symbol="BBCA.JK",
        timestamp=ts,
        probability=0.90,
        governor_state="SAFE_EXIT",
        active_models=0,
    )
    assert oi.intent == "HOLD"
    assert "GOVERNOR_SAFE_EXIT" in oi.reason_codes
