"""
Dynamic IDX symbol universe + configurable liquidity filter.

Not hardcoded to BBCA/BBRI/… only.
Not permanently locked to LQ45/IDX80.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from idxbot.data.providers.base import MarketDataProvider, SymbolInfo


@dataclass
class LiquidityFilter:
    min_avg_volume: float = 1_000_000
    min_avg_traded_value: float = 1e9  # IDR
    min_history_days: int = 60
    require_active: bool = True
    exchange: str = "IDX"

    def accepts(self, info: SymbolInfo) -> bool:
        if info.exchange.upper() != self.exchange.upper():
            return False
        if self.require_active and info.status.upper() not in ("ACTIVE", "LISTED"):
            if info.listing_status.upper() not in ("LISTED", "ACTIVE"):
                return False
        if info.avg_volume is not None and info.avg_volume < self.min_avg_volume:
            return False
        if (
            info.avg_traded_value is not None
            and info.avg_traded_value < self.min_avg_traded_value
        ):
            return False
        if info.history_days is not None and info.history_days < self.min_history_days:
            return False
        return True


class UniverseBuilder:
    def __init__(
        self,
        provider: MarketDataProvider,
        liquidity: Optional[LiquidityFilter] = None,
    ) -> None:
        self.provider = provider
        self.liquidity = liquidity or LiquidityFilter()

    def build(self) -> list[SymbolInfo]:
        symbols = self.provider.get_symbols()
        # deterministic order
        accepted = [s for s in symbols if self.liquidity.accepts(s)]
        accepted.sort(key=lambda s: s.symbol)
        return accepted

    def symbols(self) -> list[str]:
        return [s.symbol for s in self.build()]
