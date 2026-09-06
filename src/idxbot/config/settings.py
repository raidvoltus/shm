"""
Centralized, fail-safe configuration for IDX Signal Bot.

All secrets must come from environment variables / GitHub Secrets.
No credentials are stored in source code.
Compatible with pure pydantic (no pydantic-settings required).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class Settings(BaseModel):
    """Application settings with strict safety defaults."""

    # Market identity
    market: Literal["IDX"] = "IDX"
    currency: Literal["IDR"] = "IDR"
    timezone: str = "Asia/Jakarta"

    # Trading mode — LIVE_TRADING is forced false
    paper_trading: bool = True
    live_trading: bool = False
    initial_balance: int = Field(default=10_000_000, ge=0)

    # Feature flags
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Runtime knobs
    log_level: str = "INFO"
    data_cache_dir: str = ".cache/data"
    state_dir: str = ".state"
    random_seed: int = 42

    # ML / Governor
    max_models: int = 7
    min_models: int = 1
    n_jobs: int = 1  # HARD: always 1

    # Training / compute
    max_training_minutes: int = Field(default=19, ge=1, le=60)
    budget_safety_margin_seconds: int = Field(default=45, ge=0, le=300)
    enable_colab: bool = False
    enable_local_training: bool = True
    paper_trading_only: bool = True
    transaction_cost_bps: float = Field(default=15.0, ge=0.0)  # configurable IDR market cost

    @model_validator(mode="before")
    @classmethod
    def inject_env_live_trading_guard(cls, data: Any) -> Any:
        """
        Fail-closed: if LIVE_TRADING env is truthy, raise even when
        Settings() is constructed without explicit kwargs.
        """
        if not isinstance(data, dict):
            data = {}
        env_raw = os.environ.get("LIVE_TRADING")
        if env_raw is not None and env_raw.strip().lower() in ("1", "true", "yes", "on"):
            # Force the field so the field_validator fires
            data = {**data, "live_trading": True}
        return data

    @field_validator("live_trading", mode="before")
    @classmethod
    def force_live_trading_false(cls, v: object) -> bool:
        """
        Absolute safety: LIVE_TRADING must never be true in this project.
        Any attempt to set it true raises immediately. Fail-closed.
        """
        if v is True or (isinstance(v, str) and str(v).strip().lower() in ("true", "1", "yes", "on")):
            raise ValueError(
                "FATAL CONFIGURATION ERROR: LIVE_TRADING cannot be enabled. "
                "This project is signal-only + paper-trading. "
                "Live broker execution is permanently disabled."
            )
        return False

    @field_validator("paper_trading", mode="before")
    @classmethod
    def ensure_paper_trading(cls, v: object) -> bool:
        return True

    @field_validator("n_jobs", mode="before")
    @classmethod
    def force_n_jobs_one(cls, v: object) -> int:
        return 1

    @model_validator(mode="after")
    def safety_invariants(self) -> "Settings":
        if self.live_trading is True:
            raise ValueError(
                "FATAL CONFIGURATION ERROR: LIVE_TRADING=true detected after validation. "
                "System refuses to start."
            )
        if self.initial_balance < 0:
            raise ValueError("INITIAL_BALANCE must be non-negative.")
        if self.market != "IDX":
            raise ValueError("MARKET must be IDX.")
        if self.currency != "IDR":
            raise ValueError("CURRENCY must be IDR.")
        if self.n_jobs != 1:
            raise ValueError("n_jobs must be 1 (resource safety for GitHub Actions Free).")
        return self

    def assert_safe(self) -> None:
        """Explicit runtime guard."""
        if self.live_trading:
            raise RuntimeError(
                "SAFETY GUARD: LIVE_TRADING is true. Refusing to continue. "
                "This project does not support live execution."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance loaded from environment."""
    return Settings(
        market=_env_str("MARKET", "IDX"),  # type: ignore[arg-type]
        currency=_env_str("CURRENCY", "IDR"),  # type: ignore[arg-type]
        timezone=_env_str("TIMEZONE", "Asia/Jakarta"),
        paper_trading=_env_bool("PAPER_TRADING", True),
        live_trading=_env_bool("LIVE_TRADING", False),
        initial_balance=_env_int("INITIAL_BALANCE", 10_000_000),
        telegram_enabled=_env_bool("TELEGRAM_ENABLED", False),
        telegram_bot_token=_env_str("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=_env_str("TELEGRAM_CHAT_ID", ""),
        log_level=_env_str("LOG_LEVEL", "INFO"),
        data_cache_dir=_env_str("DATA_CACHE_DIR", ".cache/data"),
        state_dir=_env_str("STATE_DIR", ".state"),
        random_seed=_env_int("RANDOM_SEED", 42),
        max_models=_env_int("MAX_MODELS", 7),
        min_models=_env_int("MIN_MODELS", 1),
        n_jobs=1,
        max_training_minutes=_env_int("MAX_TRAINING_MINUTES", 19),
        budget_safety_margin_seconds=_env_int("BUDGET_SAFETY_MARGIN_SECONDS", 45),
        enable_colab=_env_bool("ENABLE_COLAB", False),
        enable_local_training=_env_bool("ENABLE_LOCAL_TRAINING", True),
        paper_trading_only=_env_bool("PAPER_TRADING_ONLY", True),
        transaction_cost_bps=float(os.environ.get("TRANSACTION_COST_BPS", "15") or "15"),
    )


def reload_settings() -> Settings:
    """Clear cache and reload (for tests)."""
    get_settings.cache_clear()
    return get_settings()
