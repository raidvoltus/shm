"""
Yahoo Finance chart API adapter for IDX equities.

Uses public chart endpoint (no API key). Symbols normalized to BBCA.JK.
Adjusted close preferred when present; never fabricates adjustment.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

import requests

from idxbot.data.providers.base import (
    AdjustmentMode,
    HistoricalRequest,
    IncrementalRequest,
    MarketDataProvider,
    PermanentProviderError,
    RateLimitError,
    SymbolInfo,
    TransientProviderError,
)

logger = logging.getLogger(__name__)
JAKARTA = ZoneInfo("Asia/Jakarta")
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
USER_AGENT = "Mozilla/5.0 (compatible; idxbot/0.10; research)"


def to_yahoo_symbol(symbol: str) -> str:
    """Canonical internal BBCA.JK → Yahoo BBCA.JK; bare BBCA → BBCA.JK."""
    s = symbol.strip().upper()
    if s.endswith(".JK"):
        return s
    if ":" in s:
        s = s.split(":")[-1]
    return f"{s}.JK"


def to_canonical_symbol(symbol: str) -> str:
    """Any provider form → internal canonical BBCA.JK."""
    return to_yahoo_symbol(symbol)


class YahooFinanceProvider(MarketDataProvider):
    """Primary production provider for IDX via Yahoo chart API."""

    def __init__(self, *, timeout: float = 20.0, session: Optional[requests.Session] = None) -> None:
        self.timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})

    def name(self) -> str:
        return "yahoo_chart"

    def get_symbols(self) -> list[SymbolInfo]:
        # Dynamic full universe not available without auth; return liquid core.
        core = ["BBCA", "BBRI", "BMRI", "TLKM", "ASII", "UNVR", "ICBP", "INDF", "BBNI", "BRIS"]
        return [SymbolInfo(symbol=to_canonical_symbol(s), exchange="IDX", status="ACTIVE") for s in core]

    def get_historical(self, request: HistoricalRequest) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for sym in request.symbols:
            rows = self._fetch_symbol(sym, request.start, request.end, request.adjustment)
            out.extend(rows)
        return out

    def get_latest_incremental(self, request: IncrementalRequest) -> list[dict[str, Any]]:
        end = date.today()
        start = end
        if request.since is not None:
            start = request.since.date() if hasattr(request.since, "date") else request.since  # type: ignore[assignment]
        hist = HistoricalRequest(
            symbols=request.symbols,
            start=start,  # type: ignore[arg-type]
            end=end,
            adjustment=request.adjustment,
        )
        return self.get_historical(hist)

    def _fetch_symbol(
        self,
        symbol: str,
        start: date,
        end: date,
        adjustment: AdjustmentMode,
    ) -> list[dict[str, Any]]:
        ysym = to_yahoo_symbol(symbol)
        canonical = to_canonical_symbol(symbol)
        period1 = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp())
        # Yahoo end is exclusive-ish; add one day
        end_dt = datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
        period2 = int(end_dt.timestamp()) + 86400

        params = {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "includePrePost": "false",
            "events": "div,splits",
        }
        url = CHART_URL.format(symbol=ysym)
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
        except requests.Timeout as e:
            raise TransientProviderError(f"yahoo timeout {ysym}: {e}") from e
        except requests.RequestException as e:
            raise TransientProviderError(f"yahoo network {ysym}: {e}") from e

        if resp.status_code == 429:
            raise RateLimitError(f"yahoo rate limit {ysym}")
        if resp.status_code >= 500:
            raise TransientProviderError(f"yahoo 5xx {ysym}: {resp.status_code}")
        if resp.status_code != 200:
            raise PermanentProviderError(f"yahoo HTTP {resp.status_code} for {ysym}")

        try:
            payload = resp.json()
        except ValueError as e:
            raise PermanentProviderError(f"yahoo invalid JSON {ysym}") from e

        chart = payload.get("chart") or {}
        if chart.get("error"):
            err = chart["error"]
            raise PermanentProviderError(f"yahoo error {ysym}: {err}")

        results = chart.get("result") or []
        if not results:
            return []

        result = results[0]
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        adj_block = ((result.get("indicators") or {}).get("adjclose") or [{}])
        adj = adj_block[0] if adj_block else {}

        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []
        adjcloses = adj.get("adjclose") or []

        rows: list[dict[str, Any]] = []
        n = len(timestamps)
        for i in range(n):
            ts = timestamps[i]
            # Yahoo timestamps are session open in exchange TZ approximation (UTC epoch)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(JAKARTA)
            o = opens[i] if i < len(opens) else None
            h = highs[i] if i < len(highs) else None
            l = lows[i] if i < len(lows) else None
            c = closes[i] if i < len(closes) else None
            v = volumes[i] if i < len(volumes) else 0
            ac = adjcloses[i] if i < len(adjcloses) else None

            if c is None or o is None or h is None or l is None:
                continue

            row: dict[str, Any] = {
                "symbol": canonical,
                "timestamp": dt.isoformat(),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": int(v or 0),
                "raw_open": float(o),
                "raw_high": float(h),
                "raw_low": float(l),
                "raw_close": float(c),
                "adjustment_mode": "RAW",
                "price_adjustment": 1.0,
                "adjustment_source": "yahoo",
                "adjustment_version": "v1",
                "source_timestamp": dt.isoformat(),
                "provider": self.name(),
            }
            if ac is not None:
                row["adjusted_close"] = float(ac)
                row["adjustment_mode"] = "ADJUSTED" if abs(float(ac) - float(c)) > 1e-9 else "RAW"
            else:
                row["adjusted_close"] = float(c)
                row["adjustment_note"] = "ADJUSTED_CLOSE_UNAVAILABLE"
            rows.append(row)

        # Sort ascending
        rows.sort(key=lambda r: r["timestamp"])
        return rows
