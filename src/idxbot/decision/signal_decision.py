"""Immutable SignalDecision contract. LLM cannot mutate after finalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DecisionType = Literal["BUY", "SELL", "NO_SIGNAL"]


@dataclass(frozen=True)
class SignalDecision:
    """Final immutable trading decision for SIGNAL_ONLY mode."""

    symbol: str
    timestamp: str
    decision: DecisionType
    confidence: float
    risk_score: float
    expected_value: float
    risk_reward: float
    market_regime: str
    signal_probability: float
    volatility_score: float
    liquidity_score: float
    model_agreement: float
    model_version: str
    feature_version: str
    signal_id: str
    filter_results: tuple[str, ...] = field(default_factory=tuple)
    risk_results: tuple[str, ...] = field(default_factory=tuple)
    explanation_context: tuple[str, ...] = field(default_factory=tuple)
    governor_state: str = ""
    active_models: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["filter_results"] = list(self.filter_results)
        d["risk_results"] = list(self.risk_results)
        d["explanation_context"] = list(self.explanation_context)
        return d

    def to_order_intent_fields(self) -> dict[str, Any]:
        intent = "HOLD" if self.decision == "NO_SIGNAL" else self.decision
        return {
            "symbol": self.symbol,
            "intent": intent,
            "confidence": self.confidence,
            "governor_state": self.governor_state or "UNKNOWN",
            "feature_version": self.feature_version,
            "model_version": self.model_version,
            "active_models": self.active_models,
            "reason_codes": self.filter_results + self.risk_results,
            "regime": self.market_regime,
            "signal_id": self.signal_id,
        }
