"""Hard safety limits — cannot be overridden by ML or LLM."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskGateResult:
    passed: bool
    failures: tuple[str, ...]


class HardRiskGate:
    def __init__(
        self,
        *,
        min_liquidity: float = 0.15,
        max_volatility: float = 0.95,
        max_risk_score: float = 90.0,
        min_probability: float = 0.01,
        max_probability: float = 0.99,
        min_rr: float = 0.0,
        require_valid_price: bool = True,
    ) -> None:
        self.min_liquidity = min_liquidity
        self.max_volatility = max_volatility
        self.max_risk_score = max_risk_score
        self.min_probability = min_probability
        self.max_probability = max_probability
        self.min_rr = min_rr
        self.require_valid_price = require_valid_price

    def evaluate(
        self,
        *,
        signal_probability: float,
        risk_score: float,
        risk_reward: float,
        volatility_score: float,
        liquidity_score: float,
        price: float | None = None,
        data_stale: bool = False,
        model_corrupt: bool = False,
        missing_data: bool = False,
    ) -> RiskGateResult:
        fails: list[str] = []
        if missing_data:
            fails.append("MISSING_DATA")
        if data_stale:
            fails.append("STALE_DATA")
        if model_corrupt:
            fails.append("MODEL_CORRUPT")
        if not (self.min_probability <= signal_probability <= self.max_probability):
            fails.append("INVALID_PROBABILITY")
        if risk_score > self.max_risk_score:
            fails.append("RISK_SCORE_HARD_LIMIT")
        if volatility_score > self.max_volatility:
            fails.append("VOLATILITY_HARD_LIMIT")
        if liquidity_score < self.min_liquidity:
            fails.append("LIQUIDITY_HARD_LIMIT")
        if risk_reward < self.min_rr:
            fails.append("INVALID_RR")
        if self.require_valid_price and (price is None or price <= 0):
            fails.append("INVALID_PRICE")
        return RiskGateResult(passed=len(fails) == 0, failures=tuple(fails))
