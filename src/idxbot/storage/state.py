"""
Portfolio / account state models for paper trading.

Initial balance: Rp 10.000.000 (deterministic).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class Position(BaseModel):
    """Single paper position."""

    symbol: str
    quantity: int = Field(..., ge=0)
    avg_price: float = Field(..., ge=0)
    market_price: Optional[float] = None


class AccountState(BaseModel):
    """
    Paper account foundation.

    Initial state must satisfy:
      cash = 10_000_000
      equity = 10_000_000
      realized_pnl = 0
      unrealized_pnl = 0
      positions = empty
    """

    cash: float = Field(default=10_000_000.0, ge=0)
    equity: float = Field(default=10_000_000.0, ge=0)
    initial_balance: float = Field(default=10_000_000.0, ge=0)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    positions: dict[str, Position] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def create_initial(cls, balance: float = 10_000_000.0) -> "AccountState":
        """Factory for deterministic initial paper account."""
        return cls(
            cash=balance,
            equity=balance,
            initial_balance=balance,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            positions={},
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AccountState":
        return cls.model_validate(data)


class PortfolioState(BaseModel):
    """Aggregate portfolio state container for persistence."""

    model_config = {"protected_namespaces": ()}

    account: AccountState
    trade_history: list[dict[str, Any]] = Field(default_factory=list)
    signal_history: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    model_metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PortfolioState":
        return cls.model_validate(data)

    @classmethod
    def create_initial(cls, balance: float = 10_000_000.0) -> "PortfolioState":
        return cls(account=AccountState.create_initial(balance))
