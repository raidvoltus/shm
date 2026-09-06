"""
Deterministic hysteresis for signal state machine.

BUY_ENTER >= 0.60
BUY_EXIT  < 0.52
SELL_ENTER <= 0.40
SELL_EXIT  > 0.48

Marginal values → HOLD. Prevents flipping.
"""

from __future__ import annotations

from typing import Literal

IntentType = Literal["BUY", "SELL", "HOLD"]

# Default thresholds (documented, deterministic)
BUY_ENTER = 0.60
BUY_EXIT = 0.52
SELL_ENTER = 0.40
SELL_EXIT = 0.48


def apply_hysteresis(
    probability: float,
    previous_intent: IntentType = "HOLD",
    *,
    buy_enter: float = BUY_ENTER,
    buy_exit: float = BUY_EXIT,
    sell_enter: float = SELL_ENTER,
    sell_exit: float = SELL_EXIT,
) -> IntentType:
    """
    Apply hysteresis band around neutral zone.

    probability is calibrated P(up) in [0, 1].
    """
    p = max(0.0, min(1.0, float(probability)))

    if previous_intent == "BUY":
        if p >= buy_exit:
            return "BUY"
        if p <= sell_enter:
            return "SELL"
        return "HOLD"

    if previous_intent == "SELL":
        if p <= sell_exit:
            return "SELL"
        if p >= buy_enter:
            return "BUY"
        return "HOLD"

    # previous HOLD or unknown
    if p >= buy_enter:
        return "BUY"
    if p <= sell_enter:
        return "SELL"
    return "HOLD"
