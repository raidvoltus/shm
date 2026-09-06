"""
Deterministic Signal Engine.

Inputs: calibrated ensemble probability, multi-horizon, regime,
portfolio constraints, governor state.

Output: OrderIntent (BUY / SELL / HOLD).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from idxbot.signals.hysteresis import apply_hysteresis
from idxbot.signals.multi_horizon import resolve_horizons
from idxbot.signals.order_intent import OrderIntent, create_order_intent


class SignalEngine:
    """Pure function style engine; state of previous intent is injected."""

    def __init__(
        self,
        *,
        feature_version: str = "v1",
        model_version: str = "v1",
        timezone: str = "Asia/Jakarta",
    ) -> None:
        self.feature_version = feature_version
        self.model_version = model_version
        self.tz = ZoneInfo(timezone)

    def generate(
        self,
        *,
        symbol: str,
        timestamp: datetime,
        probability: float,
        horizons: Mapping[str, float] | None = None,
        previous_intent: str = "HOLD",
        governor_state: str = "FULL_7",
        active_models: int = 0,
        portfolio_allowed: bool = True,
        regime: str = "",
        run_id: str = "",
        extra_reasons: Sequence[str] | None = None,
    ) -> OrderIntent:
        """
        Produce a single OrderIntent.

        If governor_state == SAFE_EXIT → forced HOLD.
        If not portfolio_allowed → forced HOLD (or keep previous) with reason.
        """
        reasons: list[str] = list(extra_reasons or [])

        if governor_state == "SAFE_EXIT":
            intent = "HOLD"
            reasons.append("GOVERNOR_SAFE_EXIT")
            conf = 0.0
        elif not portfolio_allowed:
            intent = "HOLD"
            reasons.append("PORTFOLIO_BLOCKED")
            conf = float(probability)
        else:
            # Multi-horizon resolution first
            h = horizons or {}
            p1 = float(h.get("1D", probability))
            p5 = float(h.get("5D", probability))
            p20 = float(h.get("20D", probability))
            candidate, reason = resolve_horizons(p1, p5, p20)
            reasons.append(reason)

            if candidate == "HOLD":
                intent = "HOLD"
                conf = float(probability)
            else:
                # Hysteresis on the primary probability
                intent = apply_hysteresis(probability, previous_intent)  # type: ignore[arg-type]
                # If hysteresis disagrees with horizon candidate, prefer HOLD
                if intent != candidate and intent != "HOLD":
                    intent = "HOLD"
                    reasons.append("HYSTERESIS_HORIZON_DISAGREE")
                conf = float(probability)

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=self.tz)

        return create_order_intent(
            symbol=symbol,
            timestamp=timestamp,
            intent=intent,  # type: ignore[arg-type]
            confidence=conf,
            governor_state=governor_state,
            feature_version=self.feature_version,
            model_version=self.model_version,
            active_models=active_models,
            portfolio_allowed=portfolio_allowed,
            reason_codes=reasons,
            horizons=dict(horizons or {}),
            regime=regime,
            run_id=run_id,
        )
