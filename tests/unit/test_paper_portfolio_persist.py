"""Persistent paper portfolio: cash, positions, duplicate protection."""

from __future__ import annotations

from pathlib import Path

import pytest

from idxbot.portfolio.store import (
    LocalPortfolioStore,
    PaperPortfolioEngine,
    PortfolioCorruptError,
    PortfolioStoreError,
    format_portfolio_telegram,
)


def test_initial_and_reload(tmp_path: Path):
    store = LocalPortfolioStore(tmp_path)
    b = store.load()
    assert b.account.cash_available == 10_000_000
    assert b.account.equity == 10_000_000
    assert b.positions == []
    store.save(b)
    b2 = LocalPortfolioStore(tmp_path).load()
    assert b2.account.cash_available == 10_000_000
    assert b2.state_version >= 1


def test_buy_decreases_cash_and_creates_position(tmp_path: Path):
    eng = PaperPortfolioEngine(LocalPortfolioStore(tmp_path), cost_bps=15.0)
    b = eng.load()
    b, tx = eng.apply_buy(
        b,
        symbol="BBCA",
        price=8750.0,
        signal_id="sig-buy-1",
        cycle_id="cycle-1",
        quantity=100,
    )
    eng.store.save(b)
    assert tx["side"] == "BUY"
    assert tx["cash_after"] == pytest.approx(tx["cash_before"] - tx["net_value"])
    assert b.account.cash_available == tx["cash_after"]
    assert b.account.cash_available < 10_000_000
    assert any(p.symbol == "BBCA" and p.quantity == 100 for p in b.positions)
    b2 = eng.load()
    assert b2.account.cash_available == b.account.cash_available
    assert len(b2.transactions) >= 1


def test_existing_positions_preserved(tmp_path: Path):
    eng = PaperPortfolioEngine(LocalPortfolioStore(tmp_path))
    b = eng.load()
    b, _ = eng.apply_buy(
        b, symbol="BBCA", price=8000.0, signal_id="s1", cycle_id="c1", quantity=100
    )
    eng.store.save(b)
    b, _ = eng.apply_buy(
        b, symbol="BMRI", price=5000.0, signal_id="s2", cycle_id="c2", quantity=100
    )
    eng.store.save(b)
    symbols = {p.symbol for p in b.positions if p.status == "OPEN"}
    assert "BBCA" in symbols and "BMRI" in symbols


def test_duplicate_signal_blocked(tmp_path: Path):
    eng = PaperPortfolioEngine(LocalPortfolioStore(tmp_path))
    b = eng.load()
    b, _ = eng.apply_buy(
        b, symbol="BBCA", price=8000.0, signal_id="dup", cycle_id="c1", quantity=100
    )
    with pytest.raises(PortfolioStoreError):
        eng.apply_buy(
            b, symbol="BBCA", price=8000.0, signal_id="dup", cycle_id="c2", quantity=100
        )


def test_duplicate_cycle_blocked(tmp_path: Path):
    eng = PaperPortfolioEngine(LocalPortfolioStore(tmp_path))
    b = eng.load()
    b, _ = eng.apply_buy(
        b, symbol="BBCA", price=8000.0, signal_id="s1", cycle_id="same", quantity=100
    )
    with pytest.raises(PortfolioStoreError):
        eng.apply_buy(
            b, symbol="BMRI", price=5000.0, signal_id="s2", cycle_id="same", quantity=100
        )


def test_insufficient_cash(tmp_path: Path):
    eng = PaperPortfolioEngine(LocalPortfolioStore(tmp_path))
    b = eng.load()
    b.account.cash_available = 1000.0
    with pytest.raises(PortfolioStoreError):
        eng.apply_buy(
            b, symbol="BBCA", price=8750.0, signal_id="s", cycle_id="c", quantity=100
        )


def test_corrupt_state_fail_closed(tmp_path: Path):
    store = LocalPortfolioStore(tmp_path)
    store.load()
    (tmp_path / "portfolio" / "account.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(PortfolioCorruptError):
        LocalPortfolioStore(tmp_path).load()


def test_telegram_contains_cash_and_positions(tmp_path: Path):
    eng = PaperPortfolioEngine(LocalPortfolioStore(tmp_path))
    b = eng.load()
    b, tx = eng.apply_buy(
        b,
        symbol="BBCA",
        price=8750.0,
        signal_id="s",
        cycle_id="c",
        quantity=100,
        confidence=0.78,
        score=87.4,
    )
    text = format_portfolio_telegram(
        b,
        top={
            "action": "BUY",
            "symbol": "BBCA",
            "price": 8750,
            "quantity": 100,
            "score": 87.4,
            "confidence": 0.78,
        },
    )
    assert "TOP 1" in text
    assert "BBCA" in text
    assert "Saldo tersedia" in text
    assert "Aset yang dimiliki" in text
    assert "PAPER TRADING" in text
    assert "BUY BBCA tercatat" in text


def test_no_signal_preserves_cash(tmp_path: Path):
    eng = PaperPortfolioEngine(LocalPortfolioStore(tmp_path))
    b = eng.load()
    cash = b.account.cash_available
    text = format_portfolio_telegram(b, top={"action": "NO_SIGNAL"})
    assert "portfolio tidak diubah" in text or "Tidak ada BUY" in text
    assert b.account.cash_available == cash
