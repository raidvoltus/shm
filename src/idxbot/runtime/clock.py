"""
Market clock and execution context with Asia/Jakarta timezone.

All timestamps are timezone-aware. Never rely solely on datetime.now().
Supports scheduled_at / execution_at and 30-minute scheduler latency tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

JAKARTA = ZoneInfo("Asia/Jakarta")
UTC = ZoneInfo("UTC")

# Maximum acceptable delay of a scheduled GitHub Actions run (minutes)
SCHEDULER_TOLERANCE_MINUTES = 30


class LatencyDecision(str, Enum):
    ON_TIME = "on_time"
    WITHIN_TOLERANCE = "within_tolerance"
    TOO_LATE = "too_late"


@dataclass(frozen=True)
class ExecutionContext:
    """Immutable context for a single batch run."""

    run_id: str
    scheduled_at: datetime  # intended fire time (aware)
    execution_at: datetime  # actual start time (aware)
    market_date: date
    window_start: datetime
    window_end: datetime
    latency_seconds: float
    latency_decision: LatencyDecision
    dry_run: bool = False

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "scheduled_at": self.scheduled_at.isoformat(),
            "execution_at": self.execution_at.isoformat(),
            "market_date": self.market_date.isoformat(),
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "latency_seconds": self.latency_seconds,
            "latency_decision": self.latency_decision.value,
            "dry_run": self.dry_run,
        }


class MarketClock:
    """Timezone-aware market clock for IDX (Asia/Jakarta)."""

    def __init__(self, tz: ZoneInfo = JAKARTA) -> None:
        self.tz = tz

    def now(self) -> datetime:
        """Current time in the market timezone (always aware)."""
        return datetime.now(self.tz)

    def to_market_tz(self, dt: datetime) -> datetime:
        """Normalize any aware datetime to market timezone."""
        if dt.tzinfo is None:
            raise ValueError("Naive datetime rejected. Provide timezone-aware value.")
        return dt.astimezone(self.tz)

    def market_date(self, dt: Optional[datetime] = None) -> date:
        dt = dt or self.now()
        return self.to_market_tz(dt).date()

    def build_context(
        self,
        *,
        run_id: str,
        scheduled_at: datetime,
        execution_at: Optional[datetime] = None,
        dry_run: bool = False,
        tolerance_minutes: int = SCHEDULER_TOLERANCE_MINUTES,
    ) -> ExecutionContext:
        """
        Build execution context with latency compensation.

        scheduled_at = intended cron fire time
        execution_at = when the runner actually started
        window covers [scheduled_at, execution_at] for catch-up processing
        """
        scheduled_at = self.to_market_tz(scheduled_at)
        execution_at = self.to_market_tz(execution_at or self.now())

        latency = (execution_at - scheduled_at).total_seconds()
        if latency < 0:
            # Runner early — treat as on-time, window still starts at scheduled
            latency = 0.0
            decision = LatencyDecision.ON_TIME
        elif latency <= tolerance_minutes * 60:
            decision = (
                LatencyDecision.ON_TIME
                if latency <= 60
                else LatencyDecision.WITHIN_TOLERANCE
            )
        else:
            decision = LatencyDecision.TOO_LATE

        window_start = scheduled_at
        window_end = execution_at if execution_at >= scheduled_at else scheduled_at

        return ExecutionContext(
            run_id=run_id,
            scheduled_at=scheduled_at,
            execution_at=execution_at,
            market_date=scheduled_at.date(),
            window_start=window_start,
            window_end=window_end,
            latency_seconds=latency,
            latency_decision=decision,
            dry_run=dry_run,
        )

    def should_process(self, ctx: ExecutionContext) -> tuple[bool, str]:
        """
        Decide whether to process this run given latency.

        Returns (ok, reason).
        """
        if ctx.latency_decision == LatencyDecision.TOO_LATE:
            return (
                False,
                f"SKIP: latency {ctx.latency_seconds:.0f}s exceeds "
                f"{SCHEDULER_TOLERANCE_MINUTES} min tolerance",
            )
        return True, f"OK: {ctx.latency_decision.value}"
