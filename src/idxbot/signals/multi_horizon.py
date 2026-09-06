"""
Multi-horizon conflict resolution (1D / 5D / 20D).

Policy (deterministic, documented):

  20D UP + 1D DOWN → HOLD
  20D DOWN + 1D UP → HOLD
  all UP          → BUY candidate
  all DOWN        → SELL candidate
  mixed otherwise → HOLD

UP  = prob >= 0.55
DOWN = prob <= 0.45
"""

from __future__ import annotations

from typing import Literal

IntentType = Literal["BUY", "SELL", "HOLD"]

UP_THRESHOLD = 0.55
DOWN_THRESHOLD = 0.45


def _direction(p: float) -> Literal["UP", "DOWN", "NEUTRAL"]:
    if p >= UP_THRESHOLD:
        return "UP"
    if p <= DOWN_THRESHOLD:
        return "DOWN"
    return "NEUTRAL"


def resolve_horizons(
    prob_1d: float,
    prob_5d: float,
    prob_20d: float,
) -> tuple[IntentType, str]:
    """
    Returns (intent_candidate, reason_code).
    """
    d1 = _direction(prob_1d)
    d5 = _direction(prob_5d)
    d20 = _direction(prob_20d)

    # Strong conflict: long-term vs short-term opposite
    if d20 == "UP" and d1 == "DOWN":
        return "HOLD", "CONFLICT_20U_1D"
    if d20 == "DOWN" and d1 == "UP":
        return "HOLD", "CONFLICT_20D_1U"

    directions = {d1, d5, d20}
    if directions == {"UP"}:
        return "BUY", "ALL_HORIZONS_UP"
    if directions == {"DOWN"}:
        return "SELL", "ALL_HORIZONS_DOWN"

    # Majority or partial
    ups = sum(1 for d in (d1, d5, d20) if d == "UP")
    downs = sum(1 for d in (d1, d5, d20) if d == "DOWN")
    if ups >= 2 and downs == 0:
        return "BUY", "MAJORITY_UP"
    if downs >= 2 and ups == 0:
        return "SELL", "MAJORITY_DOWN"

    return "HOLD", "MIXED_OR_NEUTRAL"
