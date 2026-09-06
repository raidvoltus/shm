"""
IDX Signal Bot — Indonesian Stock Exchange (IDX) signal-only system.

This package produces trading signals for IDX equities and supports
paper trading with a virtual balance. Live trading is permanently
disabled by design.
"""

__version__ = "0.10.0"
__phase__ = "10+/15-integration"

# Safety constants exposed at package level
LIVE_TRADING_DEFAULT = False
PAPER_TRADING_DEFAULT = True
INITIAL_BALANCE_DEFAULT = 10_000_000  # IDR
