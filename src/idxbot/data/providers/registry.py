"""
Provider registry with priority / fallback.

Never silently fabricates OHLCV. Fixture is explicit and non-production.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Optional, Sequence

from idxbot.data.providers.base import (
    HistoricalRequest,
    MarketDataProvider,
    PermanentProviderError,
    ProviderError,
    RateLimitError,
    TransientProviderError,
)
from idxbot.data.providers.fixture import FixtureProvider

logger = logging.getLogger(__name__)


class ProviderStatus(str, Enum):
    OK = "OK"
    NETWORK_ERROR = "NETWORK_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    EMPTY_DATA = "EMPTY_DATA"
    STALE_DATA = "STALE_DATA"
    SCHEMA_ERROR = "SCHEMA_ERROR"
    QUALITY_ERROR = "QUALITY_ERROR"
    NO_PROVIDER = "NO_PROVIDER"
    FIXTURE = "FIXTURE"


@dataclass
class ProviderResult:
    provider_name: str
    symbol: str
    data: list[dict[str, Any]] = field(default_factory=list)
    fetched_at: str = ""
    data_timestamp: str = ""
    status: ProviderStatus = ProviderStatus.NO_PROVIDER
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (ProviderStatus.OK, ProviderStatus.FIXTURE) and bool(self.data)


class ProviderRegistry:
    """
    Ordered fallback chain.

    Production: register real providers first.
    Fixture only when allow_fixture=True (tests / explicit env).
    """

    def __init__(self, providers: Optional[Sequence[MarketDataProvider]] = None, *, allow_fixture: bool = False) -> None:
        self._providers: list[MarketDataProvider] = list(providers or [])
        self.allow_fixture = allow_fixture
        if allow_fixture and not any(str(p.name()).startswith("fixture") for p in self._providers):
            self._providers.append(FixtureProvider())

    def register(self, provider: MarketDataProvider) -> None:
        self._providers.append(provider)

    def fetch_historical(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> ProviderResult:
        if not self._providers:
            return ProviderResult(
                provider_name="none",
                symbol=symbol,
                status=ProviderStatus.NO_PROVIDER,
                error="No providers registered",
                fetched_at=datetime.now(timezone.utc).isoformat() + "Z",
            )

        last_error = ""
        for prov in self._providers:
            name = prov.name()
            try:
                req = HistoricalRequest(symbols=[symbol], start=start, end=end)
                # FixtureProvider and real providers implement fetch differently;
                # use duck-typing for historical bars.
                bars = self._call_fetch(prov, req)
                if not bars:
                    last_error = f"{name}: empty"
                    continue
                # Normalize to list[dict]
                rows = [_bar_to_dict(b) for b in bars]
                # Fixture may return multi-symbol; filter
                rows = [r for r in rows if str(r.get("symbol", symbol)).upper() == symbol.upper()]
                if not rows:
                    last_error = f"{name}: empty after filter"
                    continue
                status = ProviderStatus.FIXTURE if name.startswith("fixture") else ProviderStatus.OK
                ts = rows[-1].get("timestamp", "") if rows else ""
                return ProviderResult(
                    provider_name=name,
                    symbol=symbol,
                    data=rows,
                    fetched_at=datetime.now(timezone.utc).isoformat() + "Z",
                    data_timestamp=str(ts),
                    status=status,
                )
            except RateLimitError as e:
                last_error = f"{name}: RATE_LIMIT {e}"
                logger.warning("provider_rate_limit", extra={"provider": name, "symbol": symbol})
                continue
            except TransientProviderError as e:
                last_error = f"{name}: NETWORK {e}"
                continue
            except (PermanentProviderError, ProviderError, Exception) as e:
                last_error = f"{name}: {type(e).__name__} {e}"
                logger.warning("provider_failed", extra={"provider": name, "error": str(e)})
                continue

        return ProviderResult(
            provider_name="fallback_exhausted",
            symbol=symbol,
            status=ProviderStatus.EMPTY_DATA if "empty" in last_error else ProviderStatus.NETWORK_ERROR,
            error=last_error or "all providers failed",
            fetched_at=datetime.now(timezone.utc).isoformat() + "Z",
        )

    def _call_fetch(self, prov: MarketDataProvider, req: HistoricalRequest) -> list[Any]:
        # Prefer explicit methods used by FixtureProvider
        if hasattr(prov, "get_historical"):
            return list(prov.get_historical(req))  # type: ignore[attr-defined]
        if hasattr(prov, "fetch_historical"):
            return list(prov.fetch_historical(req))  # type: ignore[attr-defined]
        raise PermanentProviderError(f"Provider {prov.name()} has no historical fetch method")


def _bar_to_dict(b: Any) -> dict[str, Any]:
    if isinstance(b, dict):
        d = dict(b)
    else:
        d = {}
        for k in (
            "symbol", "timestamp", "open", "high", "low", "close",
            "adjusted_close", "volume", "raw_close", "raw_open", "raw_high", "raw_low",
        ):
            if hasattr(b, k):
                v = getattr(b, k)
                d[k] = v.isoformat() if hasattr(v, "isoformat") else v
    if "open" not in d and "raw_open" in d:
        d["open"] = d["raw_open"]
    if "high" not in d and "raw_high" in d:
        d["high"] = d["raw_high"]
    if "low" not in d and "raw_low" in d:
        d["low"] = d["raw_low"]
    if "close" not in d and "raw_close" in d:
        d["close"] = d["raw_close"]
    if "adjusted_close" not in d:
        d["adjusted_close"] = d.get("close")
    return d


def build_default_registry(*, allow_fixture: bool = False) -> ProviderRegistry:
    """
    Production default priority:
      1. IDXBOT_DATA_PROVIDER_PRIORITY (comma list: yahoo_chart,fixture,...)
      2. yahoo_chart
      3. fixture only if allow_fixture / IDXBOT_USE_FIXTURE

    Never silently uses fixture in production without allow_fixture.
    """
    import os
    from idxbot.data.providers.yahoo import YahooFinanceProvider

    priority = os.environ.get("IDXBOT_DATA_PROVIDER_PRIORITY", "yahoo_chart").strip()
    names = [n.strip() for n in priority.split(",") if n.strip()]
    providers: list[MarketDataProvider] = []
    for name in names:
        if name in ("yahoo_chart", "yahoo"):
            providers.append(YahooFinanceProvider())
        elif name.startswith("fixture"):
            if allow_fixture or os.environ.get("IDXBOT_USE_FIXTURE", "").lower() in ("1", "true", "yes"):
                providers.append(FixtureProvider())
    # Always ensure yahoo if list empty of known providers
    if not providers:
        providers.append(YahooFinanceProvider())
    if allow_fixture and not any(str(p.name()).startswith("fixture") for p in providers):
        providers.append(FixtureProvider())
    return ProviderRegistry(providers=providers, allow_fixture=allow_fixture)
