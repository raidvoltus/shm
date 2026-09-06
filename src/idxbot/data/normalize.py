"""
OHLCV normalization to canonical schema.

Rejects MIXED adjustment datasets.
Timestamps → timezone-aware (canonical Asia/Jakarta for market-local).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator, model_validator

JAKARTA = ZoneInfo("Asia/Jakarta")


class NormalizedBar(BaseModel):
    """Canonical bar — raw fields always present; adjusted optional but explicit."""

    symbol: str
    timestamp: datetime
    open: float = Field(..., description="Canonical open (adjusted if mode=ADJUSTED else raw)")
    high: float
    low: float
    close: float
    volume: int = Field(..., ge=0)

    raw_open: float
    raw_high: float
    raw_low: float
    raw_close: float

    adjusted_open: Optional[float] = None
    adjusted_high: Optional[float] = None
    adjusted_low: Optional[float] = None
    adjusted_close: Optional[float] = None

    adjustment_mode: str  # RAW | ADJUSTED
    price_adjustment: float = 1.0
    adjustment_source: str = "none"
    adjustment_version: str = "v0"

    source_timestamp: Optional[datetime] = None
    retrieved_at: Optional[datetime] = None
    provider: str = "unknown"
    timeframe: str = "1d"
    schema_version: str = "3.0"
    data_version: str = "1"
    dataset_type: str = "ohlcv"

    @field_validator("symbol")
    @classmethod
    def upper_symbol(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("symbol empty")
        return v

    @field_validator("timestamp", "source_timestamp", "retrieved_at", mode="before")
    @classmethod
    def parse_dt(cls, v: Any) -> Any:
        if v is None:
            return v
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if isinstance(v, datetime) and v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return v

    @model_validator(mode="after")
    def ohlc_ok(self) -> "NormalizedBar":
        if self.high < self.low:
            raise ValueError("high < low")
        for p in (
            self.open,
            self.high,
            self.low,
            self.close,
            self.raw_open,
            self.raw_high,
            self.raw_low,
            self.raw_close,
        ):
            if p < 0:
                raise ValueError("negative price")
        if self.adjustment_mode not in ("RAW", "ADJUSTED"):
            raise ValueError(f"invalid adjustment_mode: {self.adjustment_mode}")
        return self

    def to_record(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def normalize_record(
    raw: dict[str, Any],
    *,
    provider: str = "unknown",
    retrieved_at: Optional[datetime] = None,
    prefer_adjusted: bool = True,
) -> NormalizedBar:
    """
    Map provider record → NormalizedBar.

    MIXED mode is rejected.
    If adjusted fields present and prefer_adjusted, canonical OHLC = adjusted.
    """
    mode = str(raw.get("adjustment_mode", "RAW")).upper()
    if mode == "MIXED":
        raise ValueError("MIXED adjustment dataset rejected")

    def f(key: str, *alts: str) -> Optional[float]:
        for k in (key, *alts):
            if k in raw and raw[k] is not None:
                return float(raw[k])
        return None

    raw_o = f("raw_open", "open")
    raw_h = f("raw_high", "high")
    raw_l = f("raw_low", "low")
    raw_c = f("raw_close", "close")
    if None in (raw_o, raw_h, raw_l, raw_c):
        raise ValueError("missing raw OHLC fields")

    adj_o = f("adjusted_open")
    adj_h = f("adjusted_high")
    adj_l = f("adjusted_low")
    adj_c = f("adjusted_close")

    has_adj = all(x is not None for x in (adj_o, adj_h, adj_l, adj_c))
    if mode == "ADJUSTED" and not has_adj:
        # treat raw as already adjusted series from provider
        adj_o, adj_h, adj_l, adj_c = raw_o, raw_h, raw_l, raw_c
        has_adj = True

    use_adj = prefer_adjusted and has_adj
    if use_adj:
        o, h, l, c = adj_o, adj_h, adj_l, adj_c  # type: ignore
        mode = "ADJUSTED"
    else:
        o, h, l, c = raw_o, raw_h, raw_l, raw_c
        mode = "RAW"

    ts = raw.get("timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if not isinstance(ts, datetime):
        raise ValueError("missing timestamp")
    if ts.tzinfo is None:
        raise ValueError("naive timestamp rejected")
    # normalize display to Jakarta
    ts = ts.astimezone(JAKARTA)

    src_ts = raw.get("source_timestamp")
    if isinstance(src_ts, str):
        src_ts = datetime.fromisoformat(src_ts.replace("Z", "+00:00"))

    return NormalizedBar(
        symbol=str(raw["symbol"]),
        timestamp=ts,
        open=float(o),  # type: ignore
        high=float(h),  # type: ignore
        low=float(l),  # type: ignore
        close=float(c),  # type: ignore
        volume=int(raw.get("volume", 0)),
        raw_open=float(raw_o),  # type: ignore
        raw_high=float(raw_h),  # type: ignore
        raw_low=float(raw_l),  # type: ignore
        raw_close=float(raw_c),  # type: ignore
        adjusted_open=float(adj_o) if adj_o is not None else None,
        adjusted_high=float(adj_h) if adj_h is not None else None,
        adjusted_low=float(adj_l) if adj_l is not None else None,
        adjusted_close=float(adj_c) if adj_c is not None else None,
        adjustment_mode=mode,
        price_adjustment=float(raw.get("price_adjustment", 1.0)),
        adjustment_source=str(raw.get("adjustment_source", "none")),
        adjustment_version=str(raw.get("adjustment_version", "v0")),
        source_timestamp=src_ts if isinstance(src_ts, datetime) else ts,
        retrieved_at=retrieved_at or datetime.now(JAKARTA),
        provider=provider,
        timeframe=str(raw.get("timeframe", "1d")),
        schema_version="3.0",
        data_version=str(raw.get("data_version", "1")),
        dataset_type="ohlcv",
    )


def normalize_batch(
    records: list[dict[str, Any]],
    *,
    provider: str = "unknown",
    retrieved_at: Optional[datetime] = None,
    prefer_adjusted: bool = True,
) -> list[NormalizedBar]:
    return [
        normalize_record(r, provider=provider, retrieved_at=retrieved_at, prefer_adjusted=prefer_adjusted)
        for r in records
    ]
