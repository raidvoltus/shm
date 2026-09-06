"""Idempotency, concurrency version, retry policy tests."""

import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from idxbot.runtime.idempotency import ExecutionIdentity, IdempotencyStore
from idxbot.runtime.retry import (
    RetryPolicy,
    is_transient,
    is_fatal,
    FatalError,
    StateCorruption,
    ConfigurationError,
)
from idxbot.storage.backend import LocalStorageBackend

JAKARTA = ZoneInfo("Asia/Jakarta")


def test_duplicate_identity_detected():
    with tempfile.TemporaryDirectory() as tmp:
        be = LocalStorageBackend(tmp)
        store = IdempotencyStore(be)
        ident = ExecutionIdentity(
            run_id="abc",
            scheduled_at=datetime(2024, 6, 17, 9, 0, tzinfo=JAKARTA),
            market_window="w1",
            state_version=1,
        )
        assert store.already_executed(ident) is False
        store.mark_completed(ident)
        assert store.already_executed(ident) is True
        # same identity again
        ident2 = ExecutionIdentity(
            run_id="abc",
            scheduled_at=datetime(2024, 6, 17, 9, 0, tzinfo=JAKARTA),
            market_window="w1",
            state_version=1,
        )
        assert store.already_executed(ident2) is True


def test_different_run_id_not_duplicate():
    with tempfile.TemporaryDirectory() as tmp:
        be = LocalStorageBackend(tmp)
        store = IdempotencyStore(be)
        a = ExecutionIdentity(
            run_id="a",
            scheduled_at=datetime(2024, 6, 17, 9, 0, tzinfo=JAKARTA),
            market_window="w",
            state_version=0,
        )
        store.mark_completed(a)
        b = ExecutionIdentity(
            run_id="b",
            scheduled_at=datetime(2024, 6, 17, 9, 0, tzinfo=JAKARTA),
            market_window="w",
            state_version=0,
        )
        assert store.already_executed(b) is False


def test_transient_retry_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("temp")
        return "ok"

    policy = RetryPolicy(max_attempts=3, base_delay_sec=0.01)
    assert policy.run(flaky) == "ok"
    assert calls["n"] == 3


def test_fatal_no_retry():
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise ConfigurationError("LIVE_TRADING")

    policy = RetryPolicy(max_attempts=5, base_delay_sec=0.01)
    with pytest.raises(ConfigurationError):
        policy.run(bad)
    assert calls["n"] == 1  # no retry


def test_is_transient_and_fatal():
    assert is_transient(TimeoutError())
    assert is_transient(ConnectionError())
    assert not is_transient(ConfigurationError("x"))
    assert is_fatal(StateCorruption("x"))
    assert is_fatal(FatalError("x"))
