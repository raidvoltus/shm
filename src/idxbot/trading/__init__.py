"""
Paper trading package.

No broker execution. No live orders.
Paper account foundation lives in idxbot.storage.state.
Execution simulation: idxbot.trading.paper.PaperBroker
Safe reset: idxbot.trading.reset
"""

from idxbot.trading.paper import PaperBroker, PaperExecutionError
from idxbot.trading.reset import reset_paper_state, run_reset

__all__ = [
    "PaperBroker",
    "PaperExecutionError",
    "reset_paper_state",
    "run_reset",
]
