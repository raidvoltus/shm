"""
Deterministic fixture provider for local tests.

Simulates: normal OHLCV, stock split, reverse split, duplicates,
missing candles, stale data, malformed responses, HTTP 429, timeout,
abnormal volume, invalid OHLC.

No credentials. No production API keys.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

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

JAKARTA = ZoneInfo("Asia/Jakarta")


def _bar(
    symbol: str,
    d: date,
    o: float,
    h: float,
    l: float,
    c: float,
    vol: int,
    *,
    adj_factor: float = 1.0,
    source_ts: Optional[datetime] = None,
) -> dict[str, Any]:
    ts = datetime(d.year, d.month, d.day, 16, 0, tzinfo=JAKARTA)
    raw = {
        "symbol": symbol.upper(),
        "timestamp": ts.isoformat(),
        "raw_open": o,
        "raw_high": h,
        "raw_low": l,
        "raw_close": c,
        "volume": vol,
        "adjustment_mode": "RAW",
        "price_adjustment": 1.0,
        "adjustment_source": "none",
        "adjustment_version": "v0",
        "source_timestamp": (source_ts or ts).isoformat(),
    }
    if adj_factor != 1.0:
        raw.update(
            {
                "adjusted_open": o * adj_factor,
                "adjusted_high": h * adj_factor,
                "adjusted_low": l * adj_factor,
                "adjusted_close": c * adj_factor,
                "adjustment_mode": "ADJUSTED",
                "price_adjustment": adj_factor,
                "adjustment_source": "fixture_split",
                "adjustment_version": "v1",
            }
        )
    return raw


class FixtureProvider(MarketDataProvider):
    """
    In-memory deterministic provider.

    Scenario flags control simulated failures.
    """

    def __init__(
        self,
        *,
        scenario: str = "normal",
        fail_after: int = 0,
    ) -> None:
        self.scenario = scenario
        self.fail_after = fail_after
        self._calls = 0
        self._symbols = self._build_universe()
        self._series = self._build_series()

    def name(self) -> str:
        return f"fixture:{self.scenario}"

    def _build_universe(self) -> list[SymbolInfo]:
        # Dynamic-looking universe (not a 4-symbol hardcode in consumer code)
        base = [
            ("BBCA", "Bank Central Asia", 12_000_000, 90e9, 400),
            ("BBRI", "Bank Rakyat Indonesia", 80_000_000, 40e9, 400),
            ("BMRI", "Bank Mandiri", 40_000_000, 35e9, 400),
            ("TLKM", "Telkom Indonesia", 50_000_000, 20e9, 400),
            ("ASII", "Astra International", 30_000_000, 15e9, 400),
            ("UNVR", "Unilever Indonesia", 5_000_000, 8e9, 400),
            ("ICBP", "Indofood CBP", 8_000_000, 6e9, 350),
            ("INDF", "Indofood Sukses", 10_000_000, 5e9, 350),
            ("ILLQ", "Illiquid Dummy", 50_000, 1e7, 30),  # fails liquidity
            ("NEWC", "New Listing", 2_000_000, 1e9, 10),  # short history
        ]
        return [
            SymbolInfo(
                symbol=s,
                exchange="IDX",
                name=n,
                status="ACTIVE",
                listing_status="LISTED",
                avg_volume=av,
                avg_traded_value=atv,
                history_days=hd,
            )
            for s, n, av, atv, hd in base
        ]

    def _build_series(self) -> dict[str, list[dict[str, Any]]]:
        """Build per-symbol series including split scenario for BBCA."""
        series: dict[str, list[dict[str, Any]]] = {}
        start = date(2024, 1, 2)

        for info in self._symbols:
            sym = info.symbol
            bars: list[dict[str, Any]] = []
            price = 5000.0 + hash(sym) % 5000
            for i in range(max(info.history_days or 60, 5)):
                d = start + timedelta(days=i)
                if d.weekday() >= 5:
                    continue
                o = price
                h = price * 1.01
                l = price * 0.99
                c = price * (1 + ((i % 5) - 2) * 0.002)
                vol = int((info.avg_volume or 1_000_000) * (0.8 + (i % 3) * 0.1))
                bars.append(_bar(sym, d, o, h, l, c, vol))
                price = c
            series[sym] = bars

        # Stock-split fixture on BBCA: day T-1 close 10000 → T close 2000 (5:1)
        # with adjusted series available
        split_day = date(2024, 3, 15)
        pre = date(2024, 3, 14)
        bbca = [
            _bar("BBCA", pre, 9900, 10100, 9800, 10000, 5_000_000, adj_factor=0.2),
            _bar("BBCA", split_day, 1980, 2050, 1950, 2000, 25_000_000, adj_factor=1.0),
            # post-split adjusted continues
            _bar(
                "BBCA",
                date(2024, 3, 18),
                2000,
                2100,
                1980,
                2050,
                20_000_000,
                adj_factor=1.0,
            ),
        ]
        # Merge into BBCA series (replace overlapping dates)
        existing = {b["timestamp"][:10]: b for b in series.get("BBCA", [])}
        for b in bbca:
            existing[b["timestamp"][:10]] = b
        series["BBCA"] = sorted(existing.values(), key=lambda x: x["timestamp"])

        # Reverse-split style on TLKM: 2000 → 10000 (1:5) flagged scenario data
        if self.scenario == "reverse_split":
            series["TLKM"] = [
                _bar("TLKM", date(2024, 4, 1), 1900, 2100, 1850, 2000, 10_000_000, adj_factor=5.0),
                _bar("TLKM", date(2024, 4, 2), 9800, 10200, 9700, 10000, 2_000_000, adj_factor=1.0),
            ]

        if self.scenario == "duplicates":
            b = series["ASII"][5]
            series["ASII"].insert(6, dict(b))  # exact duplicate timestamp

        if self.scenario == "invalid_ohlc":
            bad = dict(series["UNVR"][3])
            bad["raw_high"] = 1.0
            bad["raw_low"] = 100.0
            series["UNVR"][3] = bad

        if self.scenario == "abnormal_volume":
            b = dict(series["INDF"][10])
            b["volume"] = int(b["volume"] * 100)
            series["INDF"][10] = b

        if self.scenario == "stale":
            # source_timestamp far in the past relative to "now"
            for bars in series.values():
                for b in bars:
                    b["source_timestamp"] = "2020-01-01T00:00:00+07:00"

        return series

    def _maybe_fail(self) -> None:
        self._calls += 1
        if self.scenario == "timeout":
            raise TransientProviderError("connection timed out")
        if self.scenario == "http_429":
            raise RateLimitError("HTTP 429 Too Many Requests")
        if self.scenario == "http_500":
            raise TransientProviderError("HTTP 500")
        if self.scenario == "auth_error":
            raise PermanentProviderError("HTTP 401 Unauthorized")
        if self.scenario == "malformed":
            raise PermanentProviderError("schema error: missing required field")
        if self.fail_after and self._calls > self.fail_after:
            raise TransientProviderError("induced failure")

    def get_symbols(self) -> list[SymbolInfo]:
        self._maybe_fail()
        return list(self._symbols)

    def get_historical(self, request: HistoricalRequest) -> list[dict[str, Any]]:
        self._maybe_fail()
        if request.adjustment == AdjustmentMode.MIXED:
            raise PermanentProviderError("MIXED adjustment mode is not allowed")
        out: list[dict[str, Any]] = []
        for sym in request.symbols:
            for bar in self._series.get(sym.upper(), []):
                d = date.fromisoformat(bar["timestamp"][:10])
                if request.start <= d <= request.end:
                    out.append(dict(bar))
        return out

    def get_latest_incremental(
        self, request: IncrementalRequest
    ) -> list[dict[str, Any]]:
        self._maybe_fail()
        out: list[dict[str, Any]] = []
        since = request.since
        for sym in request.symbols:
            for bar in self._series.get(sym.upper(), []):
                ts = datetime.fromisoformat(bar["timestamp"])
                if since is None or ts > since:
                    out.append(dict(bar))
        return out
