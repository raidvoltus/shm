"""
Paper trading execution engine.

100% simulated. No broker API. No real orders.
PAPER_TRADING_ONLY is enforced; any accidental broker config fails closed.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from idxbot.config import get_settings
from idxbot.core.safety import assert_no_live_trading, SafetyError
from idxbot.storage.state import AccountState, PortfolioState, Position

logger = logging.getLogger(__name__)

LOT_SIZE = 100  # IDX board lot


class PaperExecutionError(Exception):
    """Raised when a paper trade cannot be applied."""


class PaperBroker:
    """
    In-memory / state-backed paper broker.

    Applies BUY / SELL against PortfolioState.
    Transaction costs are configurable (bps of notional).
    """

    def __init__(
        self,
        *,
        lot_size: int = LOT_SIZE,
        cost_bps: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        assert_no_live_trading()
        if not settings.paper_trading_only and not settings.paper_trading:
            raise SafetyError("PAPER_TRADING_ONLY must remain true.")
        self.lot_size = lot_size
        self.cost_bps = cost_bps if cost_bps is not None else settings.transaction_cost_bps

    def _cost(self, notional: float) -> float:
        return abs(notional) * (self.cost_bps / 10_000.0)

    def apply_buy(
        self,
        state: PortfolioState,
        *,
        symbol: str,
        price: float,
        quantity: Optional[int] = None,
        max_position_pct: float = 0.15,
        signal_id: str = "",
        model_version: str = "",
    ) -> dict[str, Any]:
        assert_no_live_trading()
        if price <= 0:
            raise PaperExecutionError("Invalid price for BUY")
        account = state.account
        equity = max(account.equity, account.cash)
        max_notional = equity * max_position_pct
        if quantity is None:
            # size to max lots that fit cash + cost + position limit
            affordable = int(account.cash / (price * self.lot_size * (1 + self.cost_bps / 10_000)))
            lots = max(1, min(affordable, int(max_notional / (price * self.lot_size))))
            quantity = lots * self.lot_size
        if quantity <= 0 or quantity % self.lot_size != 0:
            raise PaperExecutionError(f"Quantity must be positive multiple of lot_size={self.lot_size}")

        notional = quantity * price
        fee = self._cost(notional)
        total = notional + fee
        if total > account.cash + 1e-6:
            raise PaperExecutionError(
                f"Insufficient cash: need {total:.0f}, have {account.cash:.0f}"
            )

        pos = account.positions.get(symbol)
        if pos is None:
            account.positions[symbol] = Position(
                symbol=symbol, quantity=quantity, avg_price=price, market_price=price
            )
        else:
            new_qty = pos.quantity + quantity
            new_avg = ((pos.avg_price * pos.quantity) + notional) / new_qty
            account.positions[symbol] = Position(
                symbol=symbol, quantity=new_qty, avg_price=new_avg, market_price=price
            )

        account.cash -= total
        account.updated_at = datetime.now(timezone.utc)
        self._revalue(account)

        trade = {
            "trade_id": str(uuid.uuid4()),
            "signal_id": signal_id,
            "symbol": symbol,
            "side": "BUY",
            "quantity": quantity,
            "price": price,
            "fee": fee,
            "notional": notional,
            "model_version": model_version,
            "timestamp": account.updated_at.isoformat(),
        }
        state.trade_history.append(trade)
        logger.info("PAPER BUY %s qty=%s price=%.2f fee=%.2f", symbol, quantity, price, fee)
        return trade

    def apply_sell(
        self,
        state: PortfolioState,
        *,
        symbol: str,
        price: float,
        quantity: Optional[int] = None,
        signal_id: str = "",
        model_version: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        assert_no_live_trading()
        if price <= 0:
            raise PaperExecutionError("Invalid price for SELL")
        account = state.account
        pos = account.positions.get(symbol)
        if pos is None or pos.quantity <= 0:
            raise PaperExecutionError(f"No position to sell for {symbol}")

        qty = quantity if quantity is not None else pos.quantity
        if qty <= 0 or qty > pos.quantity:
            raise PaperExecutionError("Invalid sell quantity")
        if qty % self.lot_size != 0 and qty != pos.quantity:
            # allow full close even if not multiple (edge)
            if qty != pos.quantity:
                raise PaperExecutionError(f"Quantity must be multiple of {self.lot_size}")

        notional = qty * price
        fee = self._cost(notional)
        proceeds = notional - fee
        cost_basis = pos.avg_price * qty
        realized = proceeds - cost_basis

        remaining = pos.quantity - qty
        if remaining <= 0:
            del account.positions[symbol]
        else:
            account.positions[symbol] = Position(
                symbol=symbol,
                quantity=remaining,
                avg_price=pos.avg_price,
                market_price=price,
            )

        account.cash += proceeds
        account.realized_pnl += realized
        account.updated_at = datetime.now(timezone.utc)
        self._revalue(account)

        trade = {
            "trade_id": str(uuid.uuid4()),
            "signal_id": signal_id,
            "symbol": symbol,
            "side": "SELL",
            "quantity": qty,
            "price": price,
            "fee": fee,
            "notional": notional,
            "realized_pnl": realized,
            "reason": reason,
            "model_version": model_version,
            "timestamp": account.updated_at.isoformat(),
        }
        state.trade_history.append(trade)
        logger.info(
            "PAPER SELL %s qty=%s price=%.2f realized=%.2f", symbol, qty, price, realized
        )
        return trade

    def _revalue(self, account: AccountState) -> None:
        unrealized = 0.0
        holdings = 0.0
        for pos in account.positions.values():
            px = pos.market_price if pos.market_price is not None else pos.avg_price
            mkt = pos.quantity * px
            holdings += mkt
            unrealized += mkt - (pos.quantity * pos.avg_price)
        account.unrealized_pnl = unrealized
        account.equity = account.cash + holdings

    def mark_to_market(self, state: PortfolioState, prices: Mapping[str, float]) -> None:
        account = state.account
        for sym, pos in list(account.positions.items()):
            if sym in prices:
                account.positions[sym] = Position(
                    symbol=sym,
                    quantity=pos.quantity,
                    avg_price=pos.avg_price,
                    market_price=float(prices[sym]),
                )
        self._revalue(account)
        account.updated_at = datetime.now(timezone.utc)

    def portfolio_summary(self, state: PortfolioState) -> dict[str, Any]:
        account = state.account
        peak = account.initial_balance
        for t in state.trade_history:
            # simple peak tracking via equity snapshots if stored; else initial
            pass
        drawdown = 0.0
        if peak > 0 and account.equity < peak:
            drawdown = (peak - account.equity) / peak
        return {
            "cash": account.cash,
            "equity": account.equity,
            "realized_pnl": account.realized_pnl,
            "unrealized_pnl": account.unrealized_pnl,
            "positions": {k: v.model_dump() for k, v in account.positions.items()},
            "n_trades": len(state.trade_history),
            "drawdown_pct": round(drawdown * 100, 4),
            "initial_balance": account.initial_balance,
        }
