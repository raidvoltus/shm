"""Storage backend and state serialization tests."""

import tempfile
from pathlib import Path

import pytest

from idxbot.storage import LocalStorageBackend, AccountState, PortfolioState
from idxbot.storage.backend import ObjectStorageBackend


def test_local_storage_save_load_delete():
    with tempfile.TemporaryDirectory() as tmp:
        backend = LocalStorageBackend(root=tmp)
        data = {"hello": "world", "n": 42}
        assert backend.exists("k1") is False
        assert backend.load_state("k1") is None

        backend.save_state("k1", data)
        assert backend.exists("k1") is True
        loaded = backend.load_state("k1")
        assert loaded is not None
        assert loaded["hello"] == "world"
        assert loaded["n"] == 42

        backend.delete_state("k1")
        assert backend.exists("k1") is False
        assert backend.load_state("k1") is None


def test_account_state_initial():
    acc = AccountState.create_initial()
    assert acc.cash == 10_000_000.0
    assert acc.equity == 10_000_000.0
    assert acc.initial_balance == 10_000_000.0
    assert acc.realized_pnl == 0.0
    assert acc.unrealized_pnl == 0.0
    assert acc.positions == {}


def test_account_state_serialization():
    acc = AccountState.create_initial(balance=10_000_000)
    d = acc.to_dict()
    restored = AccountState.from_dict(d)
    assert restored.cash == 10_000_000.0
    assert restored.equity == 10_000_000.0
    assert restored.positions == {}


def test_portfolio_state_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        backend = LocalStorageBackend(root=tmp)
        ps = PortfolioState.create_initial()
        backend.save_state("portfolio", ps.to_dict())
        loaded = backend.load_state("portfolio")
        assert loaded is not None
        restored = PortfolioState.from_dict(loaded)
        assert restored.account.cash == 10_000_000.0
        assert restored.trade_history == []
        assert restored.signal_history == []


def test_object_storage_not_implemented():
    with pytest.raises(NotImplementedError):
        ObjectStorageBackend(bucket="test-bucket")
