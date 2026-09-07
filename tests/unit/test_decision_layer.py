"""ML adaptive decision + hard gates + message composer tests."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from idxbot.decision.engine import DecisionEngine
from idxbot.telegram.message_composer import (
    compose_report,
    compose_with_fallback,
    validate_composed_message,
)


def _ts():
    return datetime(2026, 9, 7, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta"))


def test_buy_decision_and_message():
    eng = DecisionEngine()
    sd = eng.decide(
        symbol="BBCA.JK",
        timestamp=_ts(),
        raw_side="BUY",
        signal_probability=0.72,
        confidence=0.70,
        model_agreement=0.85,
        volatility_score=0.3,
        liquidity_score=0.7,
        market_regime="TRENDING",
        price=8000.0,
        model_version="mv1",
        feature_version="fv1",
    )
    assert sd.decision == "BUY"
    text = compose_report(sd)
    assert "Decision: BUY" in text
    assert sd.signal_id in text
    assert validate_composed_message(sd, text)


def test_sell_decision():
    eng = DecisionEngine()
    sd = eng.decide(
        symbol="BBCA.JK",
        timestamp=_ts(),
        raw_side="SELL",
        signal_probability=0.25,
        confidence=0.70,
        model_agreement=0.85,
        volatility_score=0.3,
        liquidity_score=0.7,
        price=8000.0,
    )
    assert sd.decision == "SELL"
    assert "SELL" in compose_report(sd)


def test_no_signal_low_confidence():
    eng = DecisionEngine()
    sd = eng.decide(
        symbol="BBCA.JK",
        timestamp=_ts(),
        raw_side="BUY",
        signal_probability=0.60,
        confidence=0.40,
        model_agreement=0.9,
        volatility_score=0.2,
        liquidity_score=0.8,
        price=8000.0,
    )
    assert sd.decision == "NO_SIGNAL"
    text = compose_report(sd)
    assert "NO SIGNAL" in text or "NO TRADE" in text


def test_hard_gate_stale_data():
    eng = DecisionEngine()
    sd = eng.decide(
        symbol="BBCA.JK",
        timestamp=_ts(),
        raw_side="BUY",
        signal_probability=0.9,
        confidence=0.9,
        model_agreement=0.9,
        volatility_score=0.1,
        liquidity_score=0.9,
        price=8000.0,
        data_stale=True,
    )
    assert sd.decision == "NO_SIGNAL"
    assert "STALE_DATA" in sd.risk_results


def test_hard_gate_invalid_price():
    eng = DecisionEngine()
    sd = eng.decide(
        symbol="XX",
        timestamp=_ts(),
        raw_side="BUY",
        signal_probability=0.9,
        confidence=0.9,
        model_agreement=0.9,
        price=0.0,
    )
    assert sd.decision == "NO_SIGNAL"
    assert "INVALID_PRICE" in sd.risk_results


def test_llm_mutation_buy_to_sell_rejected():
    eng = DecisionEngine()
    sd = eng.decide(
        symbol="BBCA.JK",
        timestamp=_ts(),
        raw_side="BUY",
        signal_probability=0.8,
        confidence=0.75,
        model_agreement=0.9,
        volatility_score=0.2,
        liquidity_score=0.8,
        price=8000.0,
    )
    assert sd.decision == "BUY"
    evil = f"Decision: SELL\nFinal: SELL SIGNAL\nSignal ID: {sd.signal_id}"
    assert validate_composed_message(sd, evil) is False
    out = compose_with_fallback(sd, evil)
    assert "Decision: BUY" in out


def test_llm_mutation_nosignal_to_buy_rejected():
    eng = DecisionEngine()
    sd = eng.decide(
        symbol="BBCA.JK",
        timestamp=_ts(),
        raw_side="HOLD",
        signal_probability=0.5,
        confidence=0.2,
        price=8000.0,
    )
    assert sd.decision == "NO_SIGNAL"
    evil = f"Decision: BUY\nFinal: BUY SIGNAL\nSignal ID: {sd.signal_id}"
    assert validate_composed_message(sd, evil) is False


def test_llm_signal_id_mutation_rejected():
    eng = DecisionEngine()
    sd = eng.decide(
        symbol="BBCA.JK",
        timestamp=_ts(),
        raw_side="BUY",
        signal_probability=0.8,
        confidence=0.75,
        model_agreement=0.9,
        volatility_score=0.2,
        liquidity_score=0.8,
        price=8000.0,
    )
    fake = "a" * 64
    text = compose_report(sd).replace(sd.signal_id, fake)
    assert validate_composed_message(sd, text) is False


def test_determinism_100():
    eng = DecisionEngine()
    results = [
        eng.decide(
            symbol="BBCA.JK",
            timestamp=_ts(),
            raw_side="BUY",
            signal_probability=0.7,
            confidence=0.7,
            model_agreement=0.8,
            volatility_score=0.3,
            liquidity_score=0.7,
            price=8000.0,
            model_version="mv1",
            feature_version="fv1",
        )
        for _ in range(100)
    ]
    assert len({r.signal_id for r in results}) == 1
    assert len({r.decision for r in results}) == 1


def test_high_vol_tightens_to_nosignal():
    eng = DecisionEngine()
    sd = eng.decide(
        symbol="BBCA.JK",
        timestamp=_ts(),
        raw_side="BUY",
        signal_probability=0.58,
        confidence=0.58,
        model_agreement=0.6,
        volatility_score=0.9,
        liquidity_score=0.7,
        market_regime="HIGH_VOL",
        price=8000.0,
    )
    assert sd.decision == "NO_SIGNAL"
