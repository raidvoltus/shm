"""Timezone, scheduled_at, execution_at, latency tolerance tests."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from idxbot.runtime.clock import MarketClock, LatencyDecision, SCHEDULER_TOLERANCE_MINUTES

JAKARTA = ZoneInfo("Asia/Jakarta")


def test_now_is_aware():
    clock = MarketClock()
    n = clock.now()
    assert n.tzinfo is not None


def test_naive_rejected():
    clock = MarketClock()
    with pytest.raises(ValueError):
        clock.to_market_tz(datetime(2024, 6, 15, 9, 0, 0))


def test_scheduled_execution_on_time():
    clock = MarketClock()
    scheduled = datetime(2024, 6, 17, 9, 0, 0, tzinfo=JAKARTA)  # Monday
    execution = scheduled + timedelta(seconds=30)
    ctx = clock.build_context(
        run_id="r1", scheduled_at=scheduled, execution_at=execution
    )
    assert ctx.latency_decision == LatencyDecision.ON_TIME
    assert ctx.market_date.isoformat() == "2024-06-17"
    ok, _ = clock.should_process(ctx)
    assert ok


def test_within_30min_tolerance():
    clock = MarketClock()
    scheduled = datetime(2024, 6, 17, 9, 0, 0, tzinfo=JAKARTA)
    execution = scheduled + timedelta(minutes=17)
    ctx = clock.build_context(
        run_id="r2", scheduled_at=scheduled, execution_at=execution
    )
    assert ctx.latency_decision == LatencyDecision.WITHIN_TOLERANCE
    assert ctx.window_start == scheduled
    assert ctx.window_end == execution
    ok, reason = clock.should_process(ctx)
    assert ok


def test_over_30min_too_late():
    clock = MarketClock()
    scheduled = datetime(2024, 6, 17, 9, 0, 0, tzinfo=JAKARTA)
    execution = scheduled + timedelta(minutes=31)
    ctx = clock.build_context(
        run_id="r3", scheduled_at=scheduled, execution_at=execution
    )
    assert ctx.latency_decision == LatencyDecision.TOO_LATE
    ok, reason = clock.should_process(ctx)
    assert not ok
    assert "SKIP" in reason
    assert str(SCHEDULER_TOLERANCE_MINUTES) in reason


def test_utc_input_normalized():
    clock = MarketClock()
    # 02:00 UTC = 09:00 WIB
    scheduled_utc = datetime(2024, 6, 17, 2, 0, 0, tzinfo=ZoneInfo("UTC"))
    ctx = clock.build_context(
        run_id="r4",
        scheduled_at=scheduled_utc,
        execution_at=scheduled_utc + timedelta(minutes=5),
    )
    assert ctx.scheduled_at.tzinfo == JAKARTA
    assert ctx.scheduled_at.hour == 9
