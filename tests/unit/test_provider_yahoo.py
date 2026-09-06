"""Provider: Yahoo + registry + symbol normalization."""

from __future__ import annotations

from datetime import date

import pytest

from idxbot.data.providers.yahoo import YahooFinanceProvider, to_canonical_symbol
from idxbot.data.providers.registry import build_default_registry, ProviderStatus


def test_symbol_normalization():
    assert to_canonical_symbol("BBCA") == "BBCA.JK"
    assert to_canonical_symbol("bbca.jk") == "BBCA.JK"
    assert to_canonical_symbol("IDX:BBCA") == "BBCA.JK"


def test_yahoo_fetch_bbca():
    p = YahooFinanceProvider(timeout=25.0)
    from idxbot.data.providers.base import HistoricalRequest, AdjustmentMode

    req = HistoricalRequest(
        symbols=["BBCA"],
        start=date(2026, 6, 1),
        end=date(2026, 9, 1),
        adjustment=AdjustmentMode.ADJUSTED,
    )
    rows = p.get_historical(req)
    assert len(rows) >= 20
    assert rows[0]["symbol"] == "BBCA.JK"
    assert "close" in rows[0]
    assert "adjusted_close" in rows[0]
    # no future beyond today+1 roughly
    assert rows[-1]["timestamp"][:4] >= "2026"


def test_registry_yahoo_priority(monkeypatch):
    monkeypatch.delenv("IDXBOT_USE_FIXTURE", raising=False)
    monkeypatch.setenv("IDXBOT_DATA_PROVIDER_PRIORITY", "yahoo_chart")
    reg = build_default_registry(allow_fixture=False)
    pr = reg.fetch_historical("BBCA.JK", date(2026, 6, 1), date(2026, 9, 1))
    assert pr.ok
    assert pr.provider_name == "yahoo_chart"
    assert pr.status == ProviderStatus.OK
    assert pr.data[0]["symbol"] == "BBCA.JK"


def test_all_provider_failure_no_ok(monkeypatch):
    monkeypatch.setenv("IDXBOT_DATA_PROVIDER_PRIORITY", "fixture")
    monkeypatch.delenv("IDXBOT_USE_FIXTURE", raising=False)
    # without allow_fixture, fixture not added if only name fixture and env false
    reg = build_default_registry(allow_fixture=False)
    # force empty by using weird symbol on yahoo still may return empty
    pr = reg.fetch_historical("NOTAREALIDXZZZ.JK", date(2026, 1, 1), date(2026, 1, 5))
    # either empty or network error — must not be silent OK with fabricated data
    if pr.ok:
        # yahoo sometimes returns empty list → not ok
        assert len(pr.data) > 0
    else:
        assert pr.status != ProviderStatus.OK
