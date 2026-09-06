"""
Portfolio governor — paper constraints only.

Can force BUY → HOLD. Never executes broker orders.
SELL does not require cash check. HOLD always allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class PortfolioDecision:
    allowed: bool
    final_intent: str  # BUY / SELL / HOLD
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    exposure_pct: float = 0.0
    cash: float = 0.0


class PortfolioGovernor:
    """
    Simple deterministic checks against paper account state.

    Limits (defaults suitable for 10M IDR paper):
    - max_position_pct of equity per symbol
    - max_total_exposure_pct
    - min_cash_reserve
    """

    def __init__(
        self,
        *,
        max_position_pct: float = 0.15,
        max_total_exposure_pct: float = 0.80,
        min_cash_reserve: float = 500_000.0,
        lot_size: int = 100,
    ) -> None:
        self.max_position_pct = max_position_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.min_cash_reserve = min_cash_reserve
        self.lot_size = lot_size

    def evaluate(
        self,
        *,
        intent: str,
        symbol: str,
        confidence: float,
        price: float,
        account: Mapping[str, Any],
        positions: Mapping[str, Any] | None = None,
    ) -> PortfolioDecision:
        cash = float(account.get("cash", 0))
        equity = float(account.get("equity", cash))
        positions = positions or account.get("positions") or {}
        reasons: list[str] = []

        if intent == "HOLD":
            return PortfolioDecision(True, "HOLD", ("HOLD_PASSTHROUGH",), cash=cash)

        if intent == "SELL":
            # Need existing position
            pos = positions.get(symbol) or positions.get(symbol.upper())
            qty = 0
            if isinstance(pos, dict):
                qty = int(pos.get("quantity", 0))
            elif hasattr(pos, "quantity"):
                qty = int(pos.quantity)
            if qty <= 0:
                return PortfolioDecision(False, "HOLD", ("NO_POSITION_TO_SELL",), cash=cash)
            return PortfolioDecision(True, "SELL", ("SELL_ALLOWED",), cash=cash)

        # BUY path
        if price <= 0:
            return PortfolioDecision(False, "HOLD", ("INVALID_PRICE",), cash=cash)

        # rough notional for 1 lot
        notional = price * self.lot_size
        if cash - notional < self.min_cash_reserve:
            reasons.append("INSUFFICIENT_CASH")
            return PortfolioDecision(False, "HOLD", tuple(reasons), cash=cash)

        # exposure
        position_value = 0.0
        for sym, p in positions.items():
            if isinstance(p, dict):
                q = float(p.get("quantity", 0))
                px = float(p.get("market_price") or p.get("avg_price") or 0)
            else:
                q = float(getattr(p, "quantity", 0))
                px = float(getattr(p, "market_price", None) or getattr(p, "avg_price", 0) or 0)
            position_value += q * px

        total_exposure_pct = (position_value / equity) if equity > 0 else 0.0
        if total_exposure_pct + (notional / equity if equity else 1) > self.max_total_exposure_pct:
            reasons.append("MAX_EXPOSURE")
            return PortfolioDecision(False, "HOLD", tuple(reasons), exposure_pct=total_exposure_pct, cash=cash)

        # concentration
        cur = positions.get(symbol) or positions.get(symbol.upper())
        cur_val = 0.0
        if isinstance(cur, dict):
            cur_val = float(cur.get("quantity", 0)) * float(cur.get("avg_price", price))
        if equity > 0 and (cur_val + notional) / equity > self.max_position_pct:
            reasons.append("MAX_POSITION_PCT")
            return PortfolioDecision(False, "HOLD", tuple(reasons), exposure_pct=total_exposure_pct, cash=cash)

        return PortfolioDecision(
            True,
            "BUY",
            ("BUY_ALLOWED",),
            exposure_pct=total_exposure_pct,
            cash=cash,
        )
