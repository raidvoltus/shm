"""SIGNAL_ONLY invariant: OrderIntent may reach Telegram, never a live broker."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def test_no_broker_execution_imports_in_runtime_pipeline():
    """pipeline/runner/signals must not import live broker clients."""
    forbidden_substrings = (
        "ccxt",
        "binance",
        "alpaca",
        "ib_insync",
        "metatrader",
        "oandapy",
        "interactivebrokers",
    )
    paths = list((SRC / "idxbot" / "runtime").rglob("*.py"))
    paths += list((SRC / "idxbot" / "signals").rglob("*.py"))
    for p in paths:
        text = p.read_text(encoding="utf-8").lower()
        for bad in forbidden_substrings:
            assert bad not in text, f"{p} mentions forbidden broker module {bad}"


def test_pipeline_module_does_not_reference_paperbroker_execute():
    text = (SRC / "idxbot" / "runtime" / "pipeline.py").read_text(encoding="utf-8")
    assert "PaperBroker" not in text
    assert "place_order" not in text
    assert "submit_order" not in text


def test_live_trading_cannot_enable(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")
    from idxbot.config import reload_settings

    with pytest.raises((ValueError, RuntimeError, Exception)):
        reload_settings()


def test_signal_mode_produces_order_intent_not_broker_call(monkeypatch):
    """End-to-end SIGNAL path yields OrderIntent list; no PaperBroker.buy."""
    monkeypatch.setenv("LIVE_TRADING", "false")
    monkeypatch.setenv("IDXBOT_USE_FIXTURE", "true")
    monkeypatch.setenv("IDXBOT_LEDGER_BACKEND", "memory")
    monkeypatch.setenv("IDXBOT_LEDGER_REQUIRED", "false")
    monkeypatch.setenv("TELEGRAM_ENABLED", "false")
    from idxbot.config import reload_settings

    reload_settings()
    from idxbot.runtime.pipeline import AutonomousPipeline
    from datetime import datetime
    from zoneinfo import ZoneInfo

    pipe = AutonomousPipeline(allow_fixture=True)
    pipe.notifier.enabled = False

    result = pipe.run(
        run_id="invariant-1",
        scheduled_at=datetime(2026, 9, 7, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta")),
        account={"cash": 10_000_000, "equity": 10_000_000, "positions": {}},
        dry_run=True,
    )
    assert result.intents is not None
    for oi in result.intents:
        assert oi.intent in ("BUY", "SELL", "HOLD")
        assert oi.signal_id


def test_ast_no_broker_call_outside_trading_package():
    broker_attrs = {"place_order", "submit_order", "execute_live", "create_market_order"}
    for path in (SRC / "idxbot").rglob("*.py"):
        if "trading" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in broker_attrs:
                pytest.fail(f"{path}:{node.lineno} references {node.attr}")
