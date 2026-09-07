"""Telegram process-level dedupe and delivery classification."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from idxbot.signals.order_intent import create_order_intent
from idxbot.telegram.notifier import DeliveryClass, TelegramNotifier


def _buy_intent():
    ts = datetime(2026, 9, 7, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta"))
    return create_order_intent(
        symbol="BBCA.JK",
        timestamp=ts,
        intent="BUY",
        confidence=0.8,
        governor_state="FULL_7",
        feature_version="fv1",
        model_version="mv1",
    )


def test_process_dedupe_skips_second_send():
    n = TelegramNotifier(token="x", chat_id="1", enabled=True)
    oi = _buy_intent()
    n._sent_signal_ids.add(oi.signal_id)
    result = n.send_message("test", idempotency_key=oi.signal_id)
    assert result == DeliveryClass.SKIPPED


def test_hold_not_sent_by_default():
    n = TelegramNotifier(token="x", chat_id="1", enabled=True, send_hold=False)
    ts = datetime(2026, 9, 7, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta"))
    oi = create_order_intent(
        symbol="BBCA.JK",
        timestamp=ts,
        intent="HOLD",
        confidence=0.5,
        governor_state="FULL_7",
    )
    assert n.notify_signal(oi) is False


def test_format_includes_signal_id():
    n = TelegramNotifier(enabled=False)
    oi = _buy_intent()
    text = n.format_signal(oi)
    assert oi.signal_id in text
    assert "SignalID:" in text


def test_missing_creds_permanent():
    n = TelegramNotifier(token="", chat_id="", enabled=True)
    assert n.send_message("x") == DeliveryClass.PERMANENT_FAILURE
