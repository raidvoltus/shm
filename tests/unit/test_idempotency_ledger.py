"""Cross-run signal delivery ledger tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from idxbot.idempotency.ledger import (
    DeliveryStatus,
    FileLedgerStore,
    IdempotencyLedger,
    LedgerUnavailable,
    MemoryLedgerStore,
)
from idxbot.signals.order_intent import create_order_intent, make_signal_id
from idxbot.telegram.notifier import TelegramNotifier, _ledger_required
from zoneinfo import ZoneInfo


def test_signal_id_deterministic():
    ts = datetime(2026, 9, 7, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta"))
    a = make_signal_id("BBCA.JK", ts, "fv1", "mv1", "BUY")
    b = make_signal_id("BBCA.JK", ts, "fv1", "mv1", "BUY")
    assert a == b


def test_cross_instance_skip():
    store = MemoryLedgerStore()
    a = IdempotencyLedger(store)
    b = IdempotencyLedger(store)
    sid = "abc123deadbeef"
    assert a.reserve(sid, symbol="BBCA.JK", intent="BUY") is True
    a.finalize(sid, DeliveryStatus.SUCCESS, symbol="BBCA.JK", intent="BUY")
    assert b.should_skip(sid) is True
    assert b.reserve(sid) is False


def test_ttl_expires():
    store = MemoryLedgerStore()
    led = IdempotencyLedger(store, ttl_hours=0)
    past = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    store.save(
        {
            "schema_version": 1,
            "entries": [
                {
                    "signal_id": "expired1",
                    "created_at": past,
                    "expires_at": past,
                    "delivery_status": "SUCCESS",
                }
            ],
        }
    )
    assert led.should_skip("expired1") is False
    assert led.reserve("expired1") is True


def test_max_entries_eviction():
    store = MemoryLedgerStore()
    led = IdempotencyLedger(store, max_entries=3, ttl_hours=24)
    for i in range(5):
        assert led.reserve(f"id{i}") is True
        led.finalize(f"id{i}", DeliveryStatus.SUCCESS)
    snap = led.snapshot()
    assert len(snap["entries"]) <= 3


def test_unknown_blocks_future():
    store = MemoryLedgerStore()
    led = IdempotencyLedger(store)
    sid = "unk1"
    led.reserve(sid)
    led.finalize(sid, DeliveryStatus.UNKNOWN_DELIVERY_STATE)
    assert led.should_skip(sid) is True


def test_pending_blocks_duplicate():
    store = MemoryLedgerStore()
    led = IdempotencyLedger(store, pending_lease_minutes=30)
    sid = "pend1"
    assert led.reserve(sid) is True
    assert led.should_skip(sid) is True
    assert led.reserve(sid) is False


def test_expired_pending_allows():
    store = MemoryLedgerStore()
    led = IdempotencyLedger(store, pending_lease_minutes=0)
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
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
    assert led.should_skip("oldpend") is False


def test_file_ledger_roundtrip(tmp_path: Path):
    path = tmp_path / "ledger.json"
    store = FileLedgerStore(path)
    led = IdempotencyLedger(store)
    led.reserve("f1", symbol="BBRI.JK", intent="SELL")
    led.finalize("f1", DeliveryStatus.SUCCESS, symbol="BBRI.JK", intent="SELL")
    led2 = IdempotencyLedger(FileLedgerStore(path))
    assert led2.should_skip("f1") is True
    text = path.read_text()
    assert "TELEGRAM" not in text


def test_notify_fail_closed_ledger_unavailable(monkeypatch):
    monkeypatch.setenv("IDXBOT_LEDGER_REQUIRED", "true")

    class BoomStore:
        def load(self):
            raise LedgerUnavailable("boom")

        def save(self, payload):
            raise LedgerUnavailable("boom")

    led = IdempotencyLedger(BoomStore())
    n = TelegramNotifier(token="x", chat_id="1", enabled=True, ledger=led)
    ts = datetime(2026, 9, 7, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta"))
    oi = create_order_intent(
        symbol="BBCA.JK",
        timestamp=ts,
        intent="BUY",
        confidence=0.9,
        governor_state="FULL_7",
        feature_version="fv1",
        model_version="mv1",
    )
    assert n.notify_signal(oi) is False


def test_notify_skips_when_ledger_has_success(monkeypatch):
    monkeypatch.setenv("IDXBOT_LEDGER_REQUIRED", "false")
    store = MemoryLedgerStore()
    led = IdempotencyLedger(store)
    ts = datetime(2026, 9, 7, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta"))
    oi = create_order_intent(
        symbol="BBCA.JK",
        timestamp=ts,
        intent="BUY",
        confidence=0.9,
        governor_state="FULL_7",
        feature_version="fv1",
        model_version="mv1",
    )
    led.reserve(oi.signal_id, symbol=oi.symbol, intent=oi.intent)
    led.finalize(oi.signal_id, DeliveryStatus.SUCCESS, symbol=oi.symbol, intent=oi.intent)
    n = TelegramNotifier(token="x", chat_id="1", enabled=True, ledger=led)
    assert n.notify_signal(oi) is False


def test_ledger_required_default_actions(monkeypatch):
    monkeypatch.delenv("IDXBOT_LEDGER_REQUIRED", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert _ledger_required() is True
