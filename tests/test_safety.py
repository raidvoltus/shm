"""LIVE_TRADING safety guard unit tests."""

from __future__ import annotations

import os

import pytest

from idxbot.config import Settings, get_settings, reload_settings
from idxbot.core.safety import assert_no_live_trading, SafetyError


def _clear_live():
    os.environ.pop("LIVE_TRADING", None)
    reload_settings()


def test_assert_no_live_trading_passes():
    _clear_live()
    assert_no_live_trading()  # no raise


def test_live_trading_false_ok():
    os.environ["LIVE_TRADING"] = "false"
    try:
        reload_settings()
        s = get_settings()
        assert s.live_trading is False
        assert_no_live_trading()
    finally:
        _clear_live()


def test_live_trading_cannot_be_set_true_via_constructor():
    with pytest.raises((ValueError, Exception)):
        Settings(live_trading=True)


@pytest.mark.parametrize(
    "value",
    ["true", "TRUE", "True", "1", "yes", "YES", "on", "ON"],
)
def test_live_trading_env_truthy_variants_rejected(value: str):
    """All truthy LIVE_TRADING env values must fail-closed."""
    os.environ["LIVE_TRADING"] = value
    try:
        with pytest.raises((ValueError, Exception)) as ei:
            reload_settings()
            get_settings()
        msg = str(ei.value).lower()
        assert (
            "live_trading" in msg
            or "fatal" in msg
            or "cannot be enabled" in msg
            or "live trading" in msg
        )
    finally:
        _clear_live()


def test_n_jobs_forced_one():
    _clear_live()
    s = get_settings()
    assert s.n_jobs == 1


def test_paper_trading_forced_true():
    _clear_live()
    s = get_settings()
    assert s.paper_trading is True
    assert s.paper_trading_only is True


def test_market_currency_timezone_defaults():
    _clear_live()
    s = get_settings()
    assert s.market == "IDX"
    assert s.currency == "IDR"
    assert s.timezone == "Asia/Jakarta"
    assert s.initial_balance == 10_000_000
    assert s.max_models <= 7
    assert s.max_training_minutes <= 19
