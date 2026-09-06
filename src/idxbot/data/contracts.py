"""
Market data contracts with strict validation.

All timestamps must be timezone-aware.
IDX symbols are normalized to uppercase.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator, model_validator

JAKARTA = ZoneInfo("Asia/Jakarta")


class OHLCV(BaseModel):
    """Single OHLCV bar with validation."""

    open: float = Field(..., description="Open price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Close price")
    volume: int = Field(..., ge=0, description="Volume (non-negative integer)")

    @field_validator("open", "high", "low", "close")
    @classmethod
    def prices_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Price cannot be negative")
        return v

    @model_validator(mode="after")
    def ohlc_consistency(self) -> "OHLCV":
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) must be >= low ({self.low})")
        if self.high < self.open or self.high < self.close:
            raise ValueError("high must be >= open and close")
        if self.low > self.open or self.low > self.close:
            raise ValueError("low must be <= open and close")
        return self


class MarketData(BaseModel):
    """
    Canonical market data contract for IDX.

    - symbol: uppercase ticker (e.g. BBCA)
    - timestamp: timezone-aware datetime (prefer Asia/Jakarta)
    - OHLCV fields validated
    """

    symbol: str = Field(..., min_length=1, description="IDX ticker symbol")
    timestamp: datetime = Field(..., description="Timezone-aware bar timestamp")
    open: float
    high: float
    low: float
    close: float
    volume: int = Field(..., ge=0)

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase_nonempty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("symbol cannot be empty")
        return v.upper()

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError(
                "timestamp must be timezone-aware. "
                "Naive datetime is rejected. Use Asia/Jakarta or UTC."
            )
        return v

    @field_validator("open", "high", "low", "close")
    @classmethod
    def prices_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Price cannot be negative")
        return v

    @model_validator(mode="after")
    def ohlc_consistency(self) -> "MarketData":
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) must be >= low ({self.low})")
        if self.high < self.open or self.high < self.close:
            raise ValueError("high must be >= open and close")
        if self.low > self.open or self.low > self.close:
            raise ValueError("low must be <= open and close")
        return self

    def to_jakarta(self) -> "MarketData":
        """Return a copy with timestamp normalized to Asia/Jakarta."""
        ts = self.timestamp.astimezone(JAKARTA)
        return self.model_copy(update={"timestamp": ts})

    def to_utc(self) -> "MarketData":
        """Return a copy with timestamp normalized to UTC."""
        ts = self.timestamp.astimezone(timezone.utc)
        return self.model_copy(update={"timestamp": ts})
