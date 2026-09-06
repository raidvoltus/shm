"""
Immutable OrderIntent contract.

LIVE_TRADING is permanently disabled. This produces intents only:
BUY / SELL / HOLD.

signal_id is deterministic (SHA-256 of canonical payload) for idempotency.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal, Sequence

IntentType = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True)
class OrderIntent:
    """
    Immutable signal contract. No broker execution path exists.

    Fields required by contract:
    - signal_id, symbol, timestamp, intent, confidence, governor_state
    """

    signal_id: str
    symbol: str
    timestamp: str  # ISO-8601 Asia/Jakarta
    intent: IntentType
    confidence: float
    governor_state: str

    # Optional / extended
    active_models: int = 0
    feature_version: str = ""
    model_version: str = ""
    portfolio_allowed: bool = True
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    horizons: dict[str, float] = field(default_factory=dict)
    regime: str = ""
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reason_codes"] = list(self.reason_codes)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _canonical_bucket(ts: datetime | str, bucket_minutes: int = 5) -> str:
    """Bucket timestamp for idempotency (avoid nanosecond noise)."""
    if isinstance(ts, str):
        # assume already ISO
        return ts[:16]  # YYYY-MM-DDTHH:MM
    return ts.strftime("%Y-%m-%dT%H:%M")


def make_signal_id(
    symbol: str,
    timestamp: datetime | str,
    feature_version: str,
    model_version: str,
    intent: str = "",
    extra: str = "",
) -> str:
    """
    Deterministic SHA-256 signal_id.

    SHA256(symbol + signal_timestamp_bucket + feature_version + model_version [+ intent + extra])
    """
    bucket = _canonical_bucket(timestamp)
    payload = f"{symbol}|{bucket}|{feature_version}|{model_version}|{intent}|{extra}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_order_intent(
    *,
    symbol: str,
    timestamp: datetime | str,
    intent: IntentType,
    confidence: float,
    governor_state: str,
    feature_version: str = "v1",
    model_version: str = "v1",
    active_models: int = 0,
    portfolio_allowed: bool = True,
    reason_codes: Sequence[str] | None = None,
    horizons: dict[str, float] | None = None,
    regime: str = "",
    run_id: str = "",
) -> OrderIntent:
    """Factory that produces a fully populated immutable OrderIntent."""
    conf = max(0.0, min(1.0, float(confidence)))
    sid = make_signal_id(
        symbol=symbol,
        timestamp=timestamp,
        feature_version=feature_version,
        model_version=model_version,
        intent=intent,
    )
    ts_str = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
    return OrderIntent(
        signal_id=sid,
        symbol=symbol.upper(),
        timestamp=ts_str,
        intent=intent,
        confidence=round(conf, 6),
        governor_state=governor_state,
        active_models=active_models,
        feature_version=feature_version,
        model_version=model_version,
        portfolio_allowed=portfolio_allowed,
        reason_codes=tuple(reason_codes or ()),
        horizons=dict(horizons or {}),
        regime=regime,
        run_id=run_id,
    )
