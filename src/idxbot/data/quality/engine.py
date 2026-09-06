"""
Data quality engine — VALID / FLAGGED / REJECTED with reason codes.

No silent correction. No automatic price interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Sequence

from idxbot.data.normalize import NormalizedBar


class QualityStatus(str, Enum):
    VALID = "VALID"
    FLAGGED = "FLAGGED"
    REJECTED = "REJECTED"


class ReasonCode(str, Enum):
    OK = "OK"
    DUPLICATE_TIMESTAMP = "DUPLICATE_TIMESTAMP"
    INVALID_OHLC = "INVALID_OHLC"
    NEGATIVE_VOLUME = "NEGATIVE_VOLUME"
    NEGATIVE_PRICE = "NEGATIVE_PRICE"
    STALE_DATA = "STALE_DATA"
    MISSING_INTERVAL = "MISSING_INTERVAL"
    CORPORATE_ACTION_REVIEW = "CORPORATE_ACTION_REVIEW"
    ABNORMAL_GAP = "ABNORMAL_GAP"
    ABNORMAL_VOLUME = "ABNORMAL_VOLUME"
    INVALID_SYMBOL = "INVALID_SYMBOL"
    INVALID_TIMEZONE = "INVALID_TIMEZONE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    PROVIDER_MISSING = "PROVIDER_MISSING"
    DATA_CORRUPTION = "DATA_CORRUPTION"
    UNKNOWN_GAP = "UNKNOWN_GAP"
    MARKET_CLOSED = "MARKET_CLOSED"
    EXPECTED_BREAK = "EXPECTED_BREAK"


@dataclass
class QualityResult:
    status: QualityStatus
    reasons: list[ReasonCode] = field(default_factory=list)
    bar: Optional[NormalizedBar] = None
    detail: str = ""

    @property
    def is_usable(self) -> bool:
        return self.status != QualityStatus.REJECTED


class DataQualityEngine:
    def __init__(
        self,
        *,
        gap_threshold: float = 0.25,
        volume_spike_factor: float = 20.0,
        freshness_max_age: timedelta = timedelta(hours=48),
        now: Optional[datetime] = None,
    ) -> None:
        self.gap_threshold = gap_threshold
        self.volume_spike_factor = volume_spike_factor
        self.freshness_max_age = freshness_max_age
        self._now = now

    def _now_ts(self) -> datetime:
        from zoneinfo import ZoneInfo

        return self._now or datetime.now(ZoneInfo("Asia/Jakarta"))

    def check_bar(self, bar: NormalizedBar) -> QualityResult:
        reasons: list[ReasonCode] = []
        status = QualityStatus.VALID

        if not bar.symbol:
            return QualityResult(QualityStatus.REJECTED, [ReasonCode.INVALID_SYMBOL], bar)

        if bar.timestamp.tzinfo is None:
            return QualityResult(
                QualityStatus.REJECTED, [ReasonCode.INVALID_TIMEZONE], bar
            )

        if bar.volume < 0:
            return QualityResult(
                QualityStatus.REJECTED, [ReasonCode.NEGATIVE_VOLUME], bar
            )

        for p in (bar.open, bar.high, bar.low, bar.close, bar.raw_open, bar.raw_high, bar.raw_low, bar.raw_close):
            if p < 0:
                return QualityResult(
                    QualityStatus.REJECTED, [ReasonCode.NEGATIVE_PRICE], bar
                )

        if bar.high < bar.low or bar.high < bar.open or bar.high < bar.close:
            return QualityResult(QualityStatus.REJECTED, [ReasonCode.INVALID_OHLC], bar)
        if bar.low > bar.open or bar.low > bar.close:
            return QualityResult(QualityStatus.REJECTED, [ReasonCode.INVALID_OHLC], bar)

        # Freshness
        src = bar.source_timestamp or bar.timestamp
        age = self._now_ts() - src.astimezone(self._now_ts().tzinfo)
        if age > self.freshness_max_age:
            reasons.append(ReasonCode.STALE_DATA)
            status = QualityStatus.FLAGGED

        return QualityResult(status, reasons or [ReasonCode.OK], bar)

    def check_series(
        self, bars: Sequence[NormalizedBar]
    ) -> list[QualityResult]:
        """Series-level checks: duplicates, gaps, abnormal jumps, volume spikes."""
        if not bars:
            return [
                QualityResult(
                    QualityStatus.REJECTED,
                    [ReasonCode.INSUFFICIENT_DATA],
                    detail="empty series",
                )
            ]

        sorted_bars = sorted(bars, key=lambda b: (b.symbol, b.timestamp))
        results: list[QualityResult] = []
        seen: dict[tuple[str, datetime], int] = {}
        prev_by_sym: dict[str, NormalizedBar] = {}
        vol_hist: dict[str, list[int]] = {}

        for bar in sorted_bars:
            base = self.check_bar(bar)
            reasons = list(base.reasons)
            status = base.status

            key = (bar.symbol, bar.timestamp)
            if key in seen:
                reasons.append(ReasonCode.DUPLICATE_TIMESTAMP)
                status = QualityStatus.REJECTED
            seen[key] = seen.get(key, 0) + 1

            prev = prev_by_sym.get(bar.symbol)
            if prev is not None:
                # abnormal gap on raw close (detect split-like jumps)
                if prev.raw_close > 0:
                    gap = abs(bar.raw_close - prev.raw_close) / prev.raw_close
                    if gap > self.gap_threshold:
                        reasons.append(ReasonCode.ABNORMAL_GAP)
                        reasons.append(ReasonCode.CORPORATE_ACTION_REVIEW)
                        if status == QualityStatus.VALID:
                            status = QualityStatus.FLAGGED
                        # Do NOT auto-reject or silent-correct

                # volume spike
                hist = vol_hist.setdefault(bar.symbol, [])
                if hist:
                    avg = sum(hist) / len(hist)
                    if avg > 0 and bar.volume > avg * self.volume_spike_factor:
                        reasons.append(ReasonCode.ABNORMAL_VOLUME)
                        if status == QualityStatus.VALID:
                            status = QualityStatus.FLAGGED
                hist.append(bar.volume)
                if len(hist) > 30:
                    hist.pop(0)

            if ReasonCode.OK in reasons and len(reasons) > 1:
                reasons = [r for r in reasons if r != ReasonCode.OK]
            if not reasons:
                reasons = [ReasonCode.OK]

            results.append(QualityResult(status, reasons, bar))
            prev_by_sym[bar.symbol] = bar

        return results
