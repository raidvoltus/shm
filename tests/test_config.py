"""Configuration defaults and safety tests."""

import os
import pytest
from pydantic import ValidationError

from idxbot.config import Settings, get_settings, reload_settings


def test_defaults():
    """Configuration defaults match PHASE 1 requirements."""
    # Ensure clean env for this test
    for key in ("LIVE_TRADING", "PAPER_TRADING", "INITIAL_BALANCE", "TELEGRAM_ENABLED"):
        os.environ.pop(key, None)
    reload_settings()
    s = Settings()  # direct instantiation ignores cache
    assert s.market == "IDX"
    assert s.currency == "IDR"
    assert s.paper_trading is True
    assert s.live_trading is False
    assert s.initial_balance == 10_000_000
    assert s.telegram_enabled is False
    assert s.timezone == "Asia/Jakarta"


def test_live_trading_env_true_raises():
    """LIVE_TRADING=true must raise FATAL CONFIGURATION ERROR (fail-closed)."""
    os.environ["LIVE_TRADING"] = "true"
    try:
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        msg = str(exc_info.value).lower()
        assert "live_trading" in msg or "fatal" in msg or "cannot be enabled" in msg
    finally:
        os.environ.pop("LIVE_TRADING", None)


def test_live_trading_env_1_raises():
    os.environ["LIVE_TRADING"] = "1"
    try:
        with pytest.raises(ValidationError):
            Settings()
    finally:
        os.environ.pop("LIVE_TRADING", None)


def test_assert_safe_passes_when_false():
    for key in ("LIVE_TRADING",):
        os.environ.pop(key, None)
    reload_settings()
    s = get_settings()
    s.assert_safe()  # must not raise


def test_get_settings_cached():
    reload_settings()
    a = get_settings()
    b = get_settings()
    assert a is b
