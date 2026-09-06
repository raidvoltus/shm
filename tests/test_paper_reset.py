"""Paper reset must clear paper state and preserve ML artifacts."""

from pathlib import Path

from idxbot.storage.state import PortfolioState, AccountState, Position
from idxbot.trading.paper import PaperBroker
from idxbot.trading.reset import reset_paper_state, assert_ml_preserved, PROTECTED_ML_PATHS


def test_reset_clears_positions_and_cash():
    state = PortfolioState.create_initial(balance=10_000_000)
    broker = PaperBroker(cost_bps=0)
    # Simulate a buy by mutating state
    state.account.cash = 5_000_000
    state.account.positions["BBCA.JK"] = Position(
        symbol="BBCA.JK", quantity=100, avg_price=9000.0, market_price=9100.0
    )
    state.trade_history.append({"side": "BUY", "symbol": "BBCA.JK"})
    state.model_metadata = {"champion": "logistic_regression-v1"}

    new_state = reset_paper_state(state, balance=10_000_000)
    assert new_state.account.cash == 10_000_000
    assert new_state.account.positions == {}
    assert new_state.account.realized_pnl == 0
    # ML metadata preserved
    assert new_state.model_metadata.get("champion") == "logistic_regression-v1"
    # history has reset boundary only
    assert any(t.get("side") == "RESET" for t in new_state.trade_history)


def test_paper_buy_sell_pl():
    state = PortfolioState.create_initial()
    broker = PaperBroker(cost_bps=0)
    trade = broker.apply_buy(
        state, symbol="BBCA.JK", price=10_000.0, quantity=100, signal_id="s1"
    )
    assert trade["side"] == "BUY"
    assert state.account.cash == 10_000_000 - 1_000_000
    assert "BBCA.JK" in state.account.positions

    sell = broker.apply_sell(
        state, symbol="BBCA.JK", price=11_000.0, quantity=100, signal_id="s2"
    )
    assert sell["side"] == "SELL"
    assert sell["realized_pnl"] == 100_000.0
    assert state.account.positions == {}
    assert abs(state.account.cash - 10_100_000) < 1e-6


def test_protected_paths_constant():
    assert ".models" in PROTECTED_ML_PATHS or "ml/models" in PROTECTED_ML_PATHS
    assert ".experience" in PROTECTED_ML_PATHS or "ml/experience" in PROTECTED_ML_PATHS


def test_assert_ml_preserved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".models").mkdir()
    (tmp_path / ".experience").mkdir()
    result = assert_ml_preserved(tmp_path)
    assert isinstance(result, dict)
