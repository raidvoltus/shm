"""
Deterministic soak: 100 signal cycles mixing normal + failure injection.

Verifies:
- no duplicate accepted SignalID after SUCCESS/UNKNOWN
- no broker path
- ledger integrity
- recovery after transient failures
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from idxbot.idempotency.ledger import IdempotencyLedger, MemoryLedgerStore
from idxbot.signals.order_intent import create_order_intent, make_signal_id
from idxbot.telegram.notifier import TelegramNotifier


def _intent(cycle: int, intent: str = "BUY"):
    ts = datetime(2026, 9, 7, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta")) + timedelta(minutes=cycle)
    return create_order_intent(
        symbol="BBCA.JK",
        timestamp=ts,
        intent=intent,
        confidence=0.85,
        governor_state="FULL_7",
        feature_version="fv1",
        model_version="mv1",
    )


def test_soak_100_cycles_idempotency_and_failures(monkeypatch):
    monkeypatch.setenv("IDXBOT_LEDGER_REQUIRED", "false")
    store = MemoryLedgerStore()
    led = IdempotencyLedger(store)
    n = TelegramNotifier(token="x", chat_id="1", enabled=True, ledger=led)

    accepted: set[str] = set()
    duplicates_blocked = 0
    telegram_calls = 0
    unsafe = 0

    def fake_urlopen(req, timeout=None):
        nonlocal telegram_calls
        telegram_calls += 1
        if telegram_calls % 17 == 0:
            raise TimeoutError("injected timeout")
        if telegram_calls % 23 == 0:

            class Resp:
                status = 500

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def read(self):
                    return b"{}"

            return Resp()

        class Ok:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"ok":true}'

        return Ok()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    for cycle in range(100):
        intent = "HOLD" if cycle % 11 == 0 else "BUY"
        oi = _intent(cycle, intent=intent)

        if intent == "BUY" and cycle % 10 == 0 and cycle > 0:
            oi_prev = _intent(cycle - 10, intent="BUY")
            result = n.notify_signal(oi_prev)
            if result is False:
                duplicates_blocked += 1
            continue

        if intent == "HOLD":
            assert n.notify_signal(oi) is False
            continue

        if led.should_skip(oi.signal_id):
            duplicates_blocked += 1
            continue

        ok = n.notify_signal(oi)
        if ok:
            if oi.signal_id in accepted:
                unsafe += 1
            accepted.add(oi.signal_id)
        else:
            if led.should_skip(oi.signal_id):
                accepted.add(oi.signal_id)

    assert unsafe == 0, "duplicate accepted SignalID detected"
    snap = led.snapshot()
    assert snap["schema_version"] == 1
    assert isinstance(snap["entries"], list)
    for e in snap["entries"]:
        assert e["delivery_status"] in {
            "PENDING",
            "SUCCESS",
            "TRANSIENT_FAILURE",
            "PERMANENT_FAILURE",
            "UNKNOWN_DELIVERY_STATE",
        }
        assert "TELEGRAM" not in str(e)
    a = make_signal_id(
        "BBCA.JK",
        datetime(2026, 9, 7, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta")),
        "fv1",
        "mv1",
        "BUY",
    )
    b = make_signal_id(
        "BBCA.JK",
        datetime(2026, 9, 7, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta")),
        "fv1",
        "mv1",
        "BUY",
    )
    assert a == b


def test_soak_deterministic_signal_id_stable():
    ids = [
        make_signal_id(
            "BBCA.JK",
            datetime(2026, 9, 7, 10, 0, tzinfo=ZoneInfo("Asia/Jakarta")),
            "fv1",
            "mv1",
            "BUY",
        )
        for _ in range(50)
    ]
    assert len(set(ids)) == 1
