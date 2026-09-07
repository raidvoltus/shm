"""DecisionEngine: soft ML metrics + adaptive filter + hard risk gate → SignalDecision."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from idxbot.decision.adaptive_filter import AdaptiveFilter
from idxbot.decision.risk_gate import HardRiskGate
from idxbot.decision.signal_decision import DecisionType, SignalDecision
from idxbot.signals.order_intent import make_signal_id


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


class DecisionEngine:
    def __init__(
        self,
        adaptive: Optional[AdaptiveFilter] = None,
        hard_gate: Optional[HardRiskGate] = None,
    ) -> None:
        self.adaptive = adaptive or AdaptiveFilter()
        self.hard_gate = hard_gate or HardRiskGate()

    def decide(
        self,
        *,
        symbol: str,
        timestamp: datetime | str,
        raw_side: str,
        signal_probability: float,
        confidence: float,
        model_agreement: float = 0.5,
        volatility_score: float = 0.5,
        liquidity_score: float = 0.5,
        expected_value: float = 0.0,
        risk_reward: float = 1.0,
        market_regime: str = "UNKNOWN",
        model_version: str = "v1",
        feature_version: str = "v1",
        governor_state: str = "",
        active_models: int = 0,
        price: float | None = None,
        data_stale: bool = False,
        model_corrupt: bool = False,
        missing_data: bool = False,
        extra_reasons: tuple[str, ...] = (),
    ) -> SignalDecision:
        signal_probability = _clamp01(signal_probability)
        confidence = _clamp01(confidence)
        model_agreement = _clamp01(model_agreement)
        volatility_score = _clamp01(volatility_score)
        liquidity_score = _clamp01(liquidity_score)
        risk_score = float(
            max(
                0.0,
                min(
                    100.0,
                    (1.0 - confidence) * 40.0
                    + volatility_score * 35.0
                    + (1.0 - liquidity_score) * 25.0,
                ),
            )
        )

        thresholds = self.adaptive.compute(
            market_regime=market_regime,
            volatility_score=volatility_score,
            liquidity_score=liquidity_score,
            model_agreement=model_agreement,
            confidence=confidence,
        )
        filter_results: list[str] = list(thresholds.adjustments)

        hard = self.hard_gate.evaluate(
            signal_probability=signal_probability,
            risk_score=risk_score,
            risk_reward=risk_reward,
            volatility_score=volatility_score,
            liquidity_score=liquidity_score,
            price=price,
            data_stale=data_stale,
            model_corrupt=model_corrupt,
            missing_data=missing_data,
        )
        risk_results = list(hard.failures)

        decision: DecisionType = "NO_SIGNAL"
        soft_side = (raw_side or "HOLD").upper()
        if soft_side not in ("BUY", "SELL", "HOLD"):
            soft_side = "HOLD"
            filter_results.append("INVALID_RAW_SIDE")

        if not hard.passed:
            decision = "NO_SIGNAL"
            risk_results.append("HARD_GATE_BLOCK")
        elif soft_side == "HOLD":
            decision = "NO_SIGNAL"
            filter_results.append("NO_SETUP")
        elif confidence < thresholds.min_confidence:
            decision = "NO_SIGNAL"
            filter_results.append("CONFIDENCE_BELOW_ADAPTIVE")
        elif signal_probability < thresholds.min_probability and soft_side == "BUY":
            decision = "NO_SIGNAL"
            filter_results.append("PROBABILITY_BELOW_ADAPTIVE")
        elif signal_probability > (1.0 - thresholds.min_probability) and soft_side == "SELL":
            decision = "NO_SIGNAL"
            filter_results.append("SELL_PROBABILITY_INCONSISTENT")
        elif risk_score > thresholds.max_risk_score:
            decision = "NO_SIGNAL"
            filter_results.append("RISK_ABOVE_ADAPTIVE")
        elif model_agreement < thresholds.min_model_agreement:
            decision = "NO_SIGNAL"
            filter_results.append("AGREEMENT_BELOW_ADAPTIVE")
        elif liquidity_score < thresholds.min_liquidity_score:
            decision = "NO_SIGNAL"
            filter_results.append("LIQUIDITY_BELOW_ADAPTIVE")
        elif volatility_score > thresholds.max_volatility_score:
            decision = "NO_SIGNAL"
            filter_results.append("VOLATILITY_ABOVE_ADAPTIVE")
        else:
            decision = soft_side

        if extra_reasons:
            filter_results.extend(extra_reasons)

        intent_for_id = "HOLD" if decision == "NO_SIGNAL" else decision
        sid = make_signal_id(
            symbol,
            timestamp,
            feature_version,
            model_version,
            intent=intent_for_id,
        )
        ts_str = timestamp if isinstance(timestamp, str) else timestamp.isoformat()

        ctx: list[str] = []
        if decision in ("BUY", "SELL"):
            ctx.append(f"Soft side {soft_side} passed adaptive and hard gates")
            ctx.append(f"Confidence {confidence:.2f} >= {thresholds.min_confidence:.2f}")
        else:
            ctx.append("No trade: blocked by adaptive filter and/or hard risk gate")
            ctx.extend(filter_results[-5:])
            ctx.extend(risk_results)

        return SignalDecision(
            symbol=symbol.upper(),
            timestamp=ts_str,
            decision=decision,
            confidence=confidence,
            risk_score=risk_score,
            expected_value=float(expected_value),
            risk_reward=float(risk_reward),
            market_regime=market_regime or "UNKNOWN",
            signal_probability=signal_probability,
            volatility_score=volatility_score,
            liquidity_score=liquidity_score,
            model_agreement=model_agreement,
            model_version=model_version,
            feature_version=feature_version,
            signal_id=sid,
            filter_results=tuple(filter_results),
            risk_results=tuple(risk_results),
            explanation_context=tuple(ctx),
            governor_state=governor_state,
            active_models=active_models,
        )
