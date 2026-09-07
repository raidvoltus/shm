"""1000-cycle soak for decision + composer + idempotency."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from idxbot.decision.engine import DecisionEngine
from idxbot.idempotency.ledger import IdempotencyLedger, MemoryLedgerStore
from idxbot.signals.order_intent import create_order_intent
from idxbot.telegram.message_composer import compose_with_fallback, validate_composed_message
from idxbot.telegram.notifier import TelegramNotifier


def test_soak_1000_decision_cycles(monkeypatch):
    eng = DecisionEngine()
    store = MemoryLedgerStore()
    led = IdempotencyLedger(store)
    n = TelegramNotifier(token="x", chat_id="1", enabled=True, ledger=led)
    monkeypatch.setenv("IDXBOT_LEDGER_REQUIRED", "false")

    calls = 0

    def fake_urlopen(req, timeout=None):
        nonlocal calls
        calls += 1
        if calls % 31 == 0:
            raise TimeoutError("inject")

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

    counts = {"BUY": 0, "SELL": 0, "NO_SIGNAL": 0}
    unsafe = 0
    mutations = 0
    accepted: set[str] = set()

    for i in range(1000):
        ts = datetime(2026, 9, 7, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta")) + timedelta(minutes=i)
        stale = False
        if i % 17 == 0:
            raw, p, c = "BUY", 0.9, 0.2
        elif i % 13 == 0:
            raw, p, c = "BUY", 0.9, 0.9
            stale = True
        elif i % 5 == 0:
            raw, p, c = "SELL", 0.2, 0.75
        elif i % 3 == 0:
            raw, p, c = "BUY", 0.75, 0.72
        else:
            raw, p, c = "HOLD", 0.5, 0.5

        sd = eng.decide(
            symbol="BBCA.JK",
            timestamp=ts,
            raw_side=raw,
            signal_probability=p,
            confidence=c,
            model_agreement=0.8,
            volatility_score=0.3 if i % 11 else 0.92,
            liquidity_score=0.7,
            price=8000.0,
            data_stale=stale,
            model_version="mv1",
            feature_version="fv1",
        )
        counts[sd.decision] += 1
        text = compose_with_fallback(sd)
        if not validate_composed_message(sd, text):
            mutations += 1

        mapped = sd.to_order_intent_fields()
        oi = create_order_intent(
            symbol=sd.symbol,
            timestamp=ts,
            intent=mapped["intent"],
            confidence=sd.confidence,
            governor_state="FULL_7",
            feature_version="fv1",
            model_version="mv1",
        )

        if led.should_skip(sd.signal_id):
            continue
        ok = n.notify_signal(oi, decision=sd, message_text=text)
        if ok:
            if sd.signal_id in accepted:
                unsafe += 1
            accepted.add(sd.signal_id)
        elif led.should_skip(sd.signal_id):
            accepted.add(sd.signal_id)

    assert unsafe == 0
    assert mutations == 0
    assert counts["BUY"] + counts["SELL"] + counts["NO_SIGNAL"] == 1000
    assert counts["NO_SIGNAL"] > 0
