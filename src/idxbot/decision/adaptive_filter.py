"""Bounded deterministic adaptive thresholds. Explainable adjustments only."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveThresholds:
    min_confidence: float
    min_probability: float
    max_risk_score: float
    min_model_agreement: float
    min_liquidity_score: float
    max_volatility_score: float
    adjustments: tuple[str, ...]


BASE_MIN_CONFIDENCE = 0.55
BASE_MIN_PROBABILITY = 0.55
BASE_MAX_RISK = 70.0
BASE_MIN_AGREEMENT = 0.50
BASE_MIN_LIQUIDITY = 0.30
BASE_MAX_VOLATILITY = 0.85
MAX_LOOSEN = 0.05


class AdaptiveFilter:
    def compute(
        self,
        *,
        market_regime: str,
        volatility_score: float,
        liquidity_score: float,
        model_agreement: float,
        confidence: float,
    ) -> AdaptiveThresholds:
        min_conf = BASE_MIN_CONFIDENCE
        min_prob = BASE_MIN_PROBABILITY
        max_risk = BASE_MAX_RISK
        min_agree = BASE_MIN_AGREEMENT
        min_liq = BASE_MIN_LIQUIDITY
        max_vol = BASE_MAX_VOLATILITY
        adj: list[str] = []

        regime = (market_regime or "UNKNOWN").upper()
        if regime in ("HIGH_VOL", "CRISIS", "STRESS"):
            min_conf = min(0.95, min_conf + 0.08)
            min_prob = min(0.95, min_prob + 0.08)
            max_risk = max(40.0, max_risk - 10.0)
            adj.append("REGIME_TIGHTEN")
        elif regime in ("TRENDING", "LOW_VOL"):
            min_conf = max(0.45, min_conf - 0.03)
            adj.append("REGIME_LOOSEN_BOUNDED")

        if volatility_score >= 0.70:
            min_conf = min(0.95, min_conf + 0.06)
            max_risk = max(40.0, max_risk - 8.0)
            adj.append("HIGH_VOL_TIGHTEN")
        if liquidity_score < 0.40:
            min_conf = min(0.95, min_conf + 0.05)
            min_agree = min(0.90, min_agree + 0.10)
            adj.append("LOW_LIQ_TIGHTEN")
        if model_agreement >= 0.80:
            min_conf = max(0.45, min_conf - MAX_LOOSEN)
            adj.append("STRONG_AGREEMENT_LOOSEN")
        elif model_agreement < 0.40:
            min_conf = min(0.95, min_conf + 0.10)
            min_agree = min(0.90, min_agree + 0.05)
            adj.append("DISAGREEMENT_TIGHTEN")

        min_conf = float(max(0.45, min(0.95, min_conf)))
        min_prob = float(max(0.45, min(0.95, min_prob)))
        max_risk = float(max(30.0, min(85.0, max_risk)))
        min_agree = float(max(0.30, min(0.90, min_agree)))
        min_liq = float(max(0.10, min(0.80, min_liq)))
        max_vol = float(max(0.50, min(0.95, max_vol)))

        return AdaptiveThresholds(
            min_confidence=min_conf,
            min_probability=min_prob,
            max_risk_score=max_risk,
            min_model_agreement=min_agree,
            min_liquidity_score=min_liq,
            max_volatility_score=max_vol,
            adjustments=tuple(adj),
        )
