"""Data / ML failure paths must not emit unsafe BUY/SELL."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from idxbot.runtime.pipeline import AutonomousPipeline
from idxbot.data.providers.registry import ProviderResult, ProviderStatus


def _ts():
    return datetime(2026, 9, 7, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta"))


def test_empty_market_data_no_buy_sell(monkeypatch):
    monkeypatch.setenv("IDXBOT_USE_FIXTURE", "true")
    monkeypatch.setenv("TELEGRAM_ENABLED", "false")
    monkeypatch.setenv("IDXBOT_LEDGER_REQUIRED", "false")
    pipe = AutonomousPipeline(allow_fixture=True)

    def empty_fetch(symbol, start, end):
        return ProviderResult(
            provider_name="test",
            symbol=symbol,
            data=[],
            status=ProviderStatus.EMPTY_DATA,
            error="empty",
        )

    pipe.registry.fetch_historical = empty_fetch  # type: ignore[method-assign]
    result = pipe.run(
        run_id="empty-data",
        scheduled_at=_ts(),
        account={"cash": 10_000_000, "equity": 10_000_000, "positions": {}},
        dry_run=True,
    )
    buys = [i for i in result.intents if i.intent in ("BUY", "SELL")]
    assert buys == []
    if result.status == "FAILED":
        assert result.intents == []


def test_quality_fail_skips_symbol_no_buy(monkeypatch):
    monkeypatch.setenv("IDXBOT_USE_FIXTURE", "true")
    monkeypatch.setenv("TELEGRAM_ENABLED", "false")
    monkeypatch.setenv("IDXBOT_LEDGER_REQUIRED", "false")
    pipe = AutonomousPipeline(allow_fixture=True)
    pipe._quality_check = lambda rows: "INSUFFICIENT_DATA"  # type: ignore[method-assign]
    result = pipe.run(
        run_id="quality-fail",
        scheduled_at=_ts(),
        account={"cash": 10_000_000, "equity": 10_000_000, "positions": {}},
        dry_run=True,
    )
    buys = [i for i in result.intents if i.intent in ("BUY", "SELL")]
    assert buys == []


def test_model_unavailable_hold_only(monkeypatch):
    monkeypatch.setenv("IDXBOT_USE_FIXTURE", "true")
    monkeypatch.setenv("IDXBOT_ALLOW_MOMENTUM_FALLBACK", "false")
    monkeypatch.setenv("TELEGRAM_ENABLED", "false")
    monkeypatch.setenv("IDXBOT_LEDGER_REQUIRED", "false")
    pipe = AutonomousPipeline(allow_fixture=True)
    pipe.allow_momentum_fallback = False

    class BadInf:
        status = "MISSING"
        probability_up = 0.5
        model_version = ""
        feature_version = ""
        active_models = 0

    pipe.inferencer.predict_row = lambda row: BadInf()  # type: ignore[method-assign]
    result = pipe.run(
        run_id="no-model",
        scheduled_at=_ts(),
        account={"cash": 10_000_000, "equity": 10_000_000, "positions": {}},
        dry_run=True,
    )
    for oi in result.intents:
        assert oi.intent == "HOLD"
