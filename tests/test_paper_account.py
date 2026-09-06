"""Paper account foundation tests."""

from idxbot.storage.state import AccountState, PortfolioState


def test_paper_balance_exactly_10m():
    acc = AccountState.create_initial()
    assert acc.cash == 10_000_000
    assert acc.equity == 10_000_000
    assert acc.initial_balance == 10_000_000
    assert acc.realized_pnl == 0
    assert acc.unrealized_pnl == 0
    assert len(acc.positions) == 0


def test_custom_balance():
    acc = AccountState.create_initial(balance=5_000_000)
    assert acc.cash == 5_000_000
    assert acc.equity == 5_000_000


def test_portfolio_initial_empty_histories():
    ps = PortfolioState.create_initial()
    assert ps.account.cash == 10_000_000
    assert ps.trade_history == []
    assert ps.signal_history == []
    assert ps.metrics == {}
    assert ps.model_metadata == {}
