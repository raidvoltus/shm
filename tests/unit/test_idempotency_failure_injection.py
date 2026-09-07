"""Failure injection & recovery tests for cross-run Telegram idempotency."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from idxbot.idempotency.ledger import (
    DeliveryStatus,
    FileLedgerStore,
    IdempotencyLedger,
    LedgerUnavailable,
    MemoryLedgerStore,
    validate_ledger_payload,
)
from idxbot.signals.order_intent import create_order_intent
from idxbot.telegram.notifier import TelegramNotifier
from zoneinfo import ZoneInfo


def _sid(symbol="BBCA.JK", intent="BUY"):
    ts = datetime(2026, 9, 7, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta"))
    return create_order_intent(
        symbol=symbol,
        timestamp=ts,
        intent=intent,
        confidence=0.9,
        governor_state="FULL_7",
        feature_version="fv1",
        model_version="mv1",
    )


def test_crash_before_reservation_no_send(monkeypatch):
    monkeypatch.setenv("IDXBOT_LEDGER_REQUIRED", "true")
    store = MemoryLedgerStore()
    led = IdempotencyLedger(store)
    oi = _sid()
    assert led.should_skip(oi.signal_id) is False
    assert store.load()["entries"] == []


def test_crash_after_reservation_pending_blocks(monkeypatch):
    store = MemoryLedgerStore()
    led = IdempotencyLedger(store)
    oi = _sid()
    assert led.reserve(oi.signal_id, symbol=oi.symbol, intent=oi.intent) is True
    led2 = IdempotencyLedger(store)
    assert led2.should_skip(oi.signal_id) is True
    n = TelegramNotifier(token="x", chat_id="1", enabled=True, ledger=led2)
    monkeypatch.setenv("IDXBOT_LEDGER_REQUIRED", "false")
    assert n.notify_signal(oi) is False


def test_timeout_maps_to_unknown_and_blocks(monkeypatch):
    store = MemoryLedgerStore()
    led = IdempotencyLedger(store)
    oi = _sid()
    n = TelegramNotifier(token="x", chat_id="1", enabled=True, ledger=led)
    monkeypatch.setenv("IDXBOT_LEDGER_REQUIRED", "false")

    def boom_timeout(*a, **k):
        raise TimeoutError("read timed out")

    with patch("urllib.request.urlopen", side_effect=boom_timeout):
        assert n.notify_signal(oi) is False

    led2 = IdempotencyLedger(store)
    assert led2.should_skip(oi.signal_id) is True
    snap = led2.snapshot()
    statuses = [e["delivery_status"] for e in snap["entries"] if e["signal_id"] == oi.signal_id]
    assert DeliveryStatus.UNKNOWN_DELIVERY_STATE.value in statuses


def test_success_before_finalize_pending_then_unknown_on_expiry():
    store = MemoryLedgerStore()
    led = IdempotencyLedger(store, pending_lease_minutes=0, ttl_hours=24)
    oi = _sid()
    past = (datetime.now(timezone.utc) - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    future = (datetime.now(timezone.utc) + timedelta(hours=23)).strftime("%Y-%m-%dT%H:%M:%SZ")
    store.save(
        {
            "schema_version": 1,
            "entries": [
                {
                    "signal_id": oi.signal_id,
                    "created_at": past,
                    "expires_at": future,
                    "delivery_status": "PENDING",
                    "symbol": "BBCA.JK",
                    "intent": "BUY",
                }
            ],
        }
    )
    assert led.should_skip(oi.signal_id) is True
    snap = led.snapshot()
    st = [e["delivery_status"] for e in snap["entries"] if e["signal_id"] == oi.signal_id][0]
    assert st == DeliveryStatus.UNKNOWN_DELIVERY_STATE.value


def test_finalize_success_next_run_skips():
    store = MemoryLedgerStore()
    a = IdempotencyLedger(store)
    b = IdempotencyLedger(store)
    oi = _sid()
    a.reserve(oi.signal_id)
    a.finalize(oi.signal_id, DeliveryStatus.SUCCESS)
    assert b.should_skip(oi.signal_id) is True


def test_sha_conflict_fail_closed(monkeypatch):
    monkeypatch.setenv("IDXBOT_LEDGER_REQUIRED", "true")

    class ConflictStore:
        def load(self):
            return {"schema_version": 1, "entries": []}

        def save(self, payload):
            raise LedgerUnavailable("github ledger conflict HTTP 409")

    led = IdempotencyLedger(ConflictStore())
    n = TelegramNotifier(token="x", chat_id="1", enabled=True, ledger=led)
    assert n.notify_signal(_sid()) is False


def test_ledger_unavailable_fail_closed(monkeypatch):
    monkeypatch.setenv("IDXBOT_LEDGER_REQUIRED", "true")

    class Boom:
        def load(self):
            raise LedgerUnavailable("down")

        def save(self, payload):
            raise LedgerUnavailable("down")

    n = TelegramNotifier(token="x", chat_id="1", enabled=True, ledger=IdempotencyLedger(Boom()))
    assert n.notify_signal(_sid()) is False


def test_corrupt_ledger_fail_closed(tmp_path: Path, monkeypatch):
    path = tmp_path / "ledger.json"
    path.write_text("{not json", encoding="utf-8")
    store = FileLedgerStore(path)
    with pytest.raises(LedgerUnavailable):
        store.load()
    monkeypatch.setenv("IDXBOT_LEDGER_REQUIRED", "true")
    led = IdempotencyLedger(store)
    n = TelegramNotifier(token="x", chat_id="1", enabled=True, ledger=led)
    assert n.notify_signal(_sid()) is False
    assert path.read_text().startswith("{not")


def test_corrupt_missing_schema():
    with pytest.raises(LedgerUnavailable):
        validate_ledger_payload({"entries": []})


def test_corrupt_unknown_status():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fut = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with pytest.raises(LedgerUnavailable):
        validate_ledger_payload(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "signal_id": "x",
                        "created_at": now,
                        "expires_at": fut,
                        "delivery_status": "HACKED",
                    }
                ],
            }
        )


def test_corrupt_naive_timestamp():
    with pytest.raises(LedgerUnavailable):
        validate_ledger_payload(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "signal_id": "x",
                        "created_at": "2026-09-07T09:00:00",
                        "expires_at": "2026-09-08T09:00:00",
                        "delivery_status": "SUCCESS",
                    }
                ],
            }
        )


def test_concurrent_reservation_one_owner():
    store = MemoryLedgerStore()
    a = IdempotencyLedger(store)
    b = IdempotencyLedger(store)
    oi = _sid()
    assert a.reserve(oi.signal_id) is True
    assert b.reserve(oi.signal_id) is False


def test_unknown_blocks_future_delivery():
    store = MemoryLedgerStore()
    led = IdempotencyLedger(store)
    oi = _sid()
    led.reserve(oi.signal_id)
    led.finalize(oi.signal_id, DeliveryStatus.UNKNOWN_DELIVERY_STATE)
    assert led.should_skip(oi.signal_id) is True
    assert led.reserve(oi.signal_id) is False


def test_pending_expiry_promotes_to_unknown_not_retryable():
    store = MemoryLedgerStore()
    led = IdempotencyLedger(store, pending_lease_minutes=0, ttl_hours=24)
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    future = (datetime.now(timezone.utc) + timedelta(hours=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    store.save(
        {
            "schema_version": 1,
            "entries": [
                {
                    "signal_id": "oldpend",
                    "created_at": past,
                    "expires_at": future,
                    "delivery_status": "PENDING",
                }
            ],
        }
    )
    assert led.should_skip("oldpend") is True
    assert led.reserve("oldpend") is False


def test_ttl_cleanup_removes_fully_expired():
    store = MemoryLedgerStore()
    led = IdempotencyLedger(store)
    past = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    store.save(
        {
            "schema_version": 1,
            "entries": [
                {
                    "signal_id": "gone",
                    "created_at": past,
                    "expires_at": past,
                    "delivery_status": "SUCCESS",
                }
            ],
        }
    )
    assert led.should_skip("gone") is False
    assert led.snapshot()["entries"] == []


def test_max_entries_deterministic():
    store = MemoryLedgerStore()
    led = IdempotencyLedger(store, max_entries=3, ttl_hours=24)
    for i in range(5):
        led.reserve(f"id{i}")
        led.finalize(f"id{i}", DeliveryStatus.SUCCESS)
    assert len(led.snapshot()["entries"]) <= 3


def test_secret_not_serialized():
    store = MemoryLedgerStore()
    led = IdempotencyLedger(store)
    led.reserve("abc", symbol="BBCA.JK", intent="BUY")
    led.finalize("abc", DeliveryStatus.SUCCESS, symbol="BBCA.JK", intent="BUY")
    raw = str(store.load())
    assert "TELEGRAM" not in raw
    assert "Bearer" not in raw


def test_duplicate_signal_cross_instance():
    store = MemoryLedgerStore()
    a = IdempotencyLedger(store)
    b = IdempotencyLedger(store)
    oi = _sid()
    a.reserve(oi.signal_id)
    a.finalize(oi.signal_id, DeliveryStatus.SUCCESS)
    n = TelegramNotifier(token="x", chat_id="1", enabled=True, ledger=b)
    assert n.notify_signal(oi) is False
