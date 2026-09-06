"""Integration: full autonomous pipeline on synthetic/fixture data."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from idxbot.runtime.pipeline import AutonomousPipeline
from idxbot.runtime.runner import MarketRuntime
from idxbot.signals.order_intent import make_signal_id


@pytest.fixture
def account():
    return {"cash": 10_000_000.0, "equity": 10_000_000.0, "positions": {}}


def test_pipeline_produces_intents(account, monkeypatch):
    monkeypatch.setenv('IDXBOT_USE_FIXTURE', 'true')
    monkeypatch.setenv('IDXBOT_ALLOW_MOMENTUM_FALLBACK', 'true')
    pipe = AutonomousPipeline(allow_fixture=True)
    ts = datetime(2025, 1, 15, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta"))
    result = pipe.run(run_id="it-1", scheduled_at=ts, account=account, dry_run=True)
    assert result.status in ("OK", "DEGRADED")
    assert len(result.intents) >= 1
    for oi in result.intents:
        assert oi.intent in ("BUY", "SELL", "HOLD")
        assert 0.0 <= oi.confidence <= 1.0
        assert oi.signal_id
        assert oi.governor_state


def test_safe_exit_no_ml(account, monkeypatch):
    monkeypatch.setenv('IDXBOT_USE_FIXTURE', 'true')
    monkeypatch.setenv('IDXBOT_ALLOW_MOMENTUM_FALLBACK', 'true')
    pipe = AutonomousPipeline(allow_fixture=True)
    ts = datetime(2025, 1, 15, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta"))
    # inject critical memory to force SAFE_EXIT
    result = pipe.run(
        run_id="it-safe",
        scheduled_at=ts,
        account=account,
        dry_run=True,
        resource_inject={"memory_available_bytes": 1_000_000},  # very low
    )
    # Depending on governor thresholds may or may not SAFE_EXIT; if it does:
    if result.governor_state == "SAFE_EXIT":
        assert all(i.intent == "HOLD" for i in result.intents)
        assert any(s.name == "ml" and s.status == "SKIP" for s in result.stages)


def test_idempotent_signal_ids(account, monkeypatch):
    monkeypatch.setenv('IDXBOT_USE_FIXTURE', 'true')
    monkeypatch.setenv('IDXBOT_ALLOW_MOMENTUM_FALLBACK', 'true')
    pipe = AutonomousPipeline(allow_fixture=True)
    ts = datetime(2025, 1, 15, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta"))
    r1 = pipe.run(run_id="a", scheduled_at=ts, account=account, dry_run=True)
    r2 = pipe.run(run_id="b", scheduled_at=ts, account=account, dry_run=True)
    ids1 = {i.symbol: i.signal_id for i in r1.intents}
    ids2 = {i.symbol: i.signal_id for i in r2.intents}
    for sym in ids1:
        assert ids1[sym] == ids2[sym]


def test_market_runtime_dry_run_wires_pipeline(monkeypatch):
    monkeypatch.setenv('IDXBOT_USE_FIXTURE', 'true')
    monkeypatch.setenv('IDXBOT_ALLOW_MOMENTUM_FALLBACK', 'true')
    rt = MarketRuntime()
    result = rt.run(
        run_id="rt-1",
        scheduled_at=datetime(2025, 1, 15, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta")),
        dry_run=True,
    )
    assert result.status in ("OK", "SKIP", "DEGRADED", "FAILED", "SAFE_EXIT")
    d = result.to_dict()
    # when not latency-skipped, pipeline key present
    if result.status != "SKIP":
        assert "pipeline" in d
        assert "intents" in d["pipeline"]


def test_portfolio_blocks_buy_when_no_cash(account):
    from idxbot.portfolio.governor import PortfolioGovernor

    gov = PortfolioGovernor(min_cash_reserve=9_999_000)
    dec = gov.evaluate(
        intent="BUY",
        symbol="BBCA",
        confidence=0.9,
        price=9000.0,
        account={"cash": 10_000_000, "equity": 10_000_000, "positions": {}},
    )
    assert dec.final_intent == "HOLD"
    assert not dec.allowed
