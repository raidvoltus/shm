"""
Market data provider interface — vendor-agnostic.

Providers can be swapped without touching features / ML / signals / risk.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional, Sequence


class AdjustmentMode(str, Enum):
    RAW = "RAW"
    ADJUSTED = "ADJUSTED"
    MIXED = "MIXED"  # rejected by normalizer


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    exchange: str = "IDX"
    name: str = ""
    status: str = "ACTIVE"
    listing_status: str = "LISTED"
    avg_volume: Optional[float] = None
    avg_traded_value: Optional[float] = None
    history_days: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper())


@dataclass
class HistoricalRequest:
    symbols: Sequence[str]
    start: date
    end: date
    adjustment: AdjustmentMode = AdjustmentMode.ADJUSTED
    timeframe: str = "1d"


@dataclass
class IncrementalRequest:
    symbols: Sequence[str]
    since: Optional[datetime] = None
    adjustment: AdjustmentMode = AdjustmentMode.ADJUSTED
    timeframe: str = "1d"


class ProviderError(Exception):
    """Base provider failure."""


class TransientProviderError(ProviderError):
    """Network timeout, 5xx, temporary unavailability — may retry."""


class RateLimitError(TransientProviderError):
    """HTTP 429 or equivalent."""


class PermanentProviderError(ProviderError):
    """Auth failure, schema error, invalid request — do not retry blindly."""


class MarketDataProvider(ABC):
    """Vendor-agnostic market data source."""

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def get_symbols(self) -> list[SymbolInfo]:
        """Return dynamic IDX universe (not a hardcoded shortlist)."""
        ...

    @abstractmethod
    def get_historical(self, request: HistoricalRequest) -> list[dict[str, Any]]:
        """
        Historical bars. Each record must explicitly carry raw and/or adjusted
        fields — never an ambiguous bare 'close' without mode metadata.
        """
        ...

    @abstractmethod
    def get_latest_incremental(
        self, request: IncrementalRequest
    ) -> list[dict[str, Any]]:
        """Incremental bars since last successful timestamp."""
        ...

    def get_market_status(self) -> dict[str, Any]:
        return {"status": "UNKNOWN"}

    def get_adjustments(
        self, symbol: str, start: date, end: date
    ) -> list[dict[str, Any]]:
        return []

    def get_metadata(self) -> dict[str, Any]:
        return {"provider": self.name()}
