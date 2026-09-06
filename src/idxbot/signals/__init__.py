"""Signal generation package."""

from idxbot.signals.hysteresis import apply_hysteresis
from idxbot.signals.multi_horizon import resolve_horizons
from idxbot.signals.order_intent import OrderIntent, create_order_intent, make_signal_id
from idxbot.signals.signal_engine import SignalEngine

__all__ = [
    "OrderIntent",
    "create_order_intent",
    "make_signal_id",
    "apply_hysteresis",
    "resolve_horizons",
    "SignalEngine",
]
