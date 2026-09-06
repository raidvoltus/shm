"""
Feature engineering engine — deterministic, causal, anti-lookahead.

Architecture:
  OHLCV bars → sort → validate timeframe → compute features per symbol
  → feature quality → ordered feature rows

Labels / future returns are NEVER computed here (PHASE 5).
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional, Sequence

from idxbot.data.normalize import NormalizedBar
from idxbot.features.indicators import (
    causal_percentile_rank,
    ema_series,
    rolling_std,
    rsi,
    sma,
    true_range,
)
from idxbot.features.schema import (
    CANONICAL_TIMEFRAME,
    FEATURE_COLUMNS,
    FEATURE_METADATA,
    empty_feature_row,
)


class FeatureEngineError(ValueError):
    pass


class FeatureEngine:
    """
    Compute technical features from NormalizedBar sequences.

    price_mode: 'adjusted' uses bar.close (canonical adjusted when available);
                'raw' uses bar.raw_close.
    """

    def __init__(self, *, price_mode: str = "adjusted", timeframe: str = CANONICAL_TIMEFRAME) -> None:
        if timeframe.lower() not in ("1d", "1D"):
            raise FeatureEngineError(
                f"Unsupported timeframe {timeframe!r}. Only 1D is accepted."
            )
        if price_mode not in ("adjusted", "raw"):
            raise FeatureEngineError("price_mode must be 'adjusted' or 'raw'")
        self.price_mode = price_mode
        self.timeframe = CANONICAL_TIMEFRAME

    def _price(self, bar: NormalizedBar) -> float:
        if self.price_mode == "raw":
            return float(bar.raw_close)
        return float(bar.close)

    def _open(self, bar: NormalizedBar) -> float:
        if self.price_mode == "raw":
            return float(bar.raw_open)
        return float(bar.open)

    def _high(self, bar: NormalizedBar) -> float:
        if self.price_mode == "raw":
            return float(bar.raw_high)
        return float(bar.high)

    def _low(self, bar: NormalizedBar) -> float:
        if self.price_mode == "raw":
            return float(bar.raw_low)
        return float(bar.low)

    def _validate_bars(self, bars: Sequence[NormalizedBar]) -> list[NormalizedBar]:
        if not bars:
            return []
        # reject wrong timeframe metadata if present
        for b in bars:
            tf = getattr(b, "timeframe", "1d") or "1d"
            if str(tf).lower() not in ("1d", "1D"):
                raise FeatureEngineError(f"Bar timeframe {tf!r} rejected; only 1D allowed")
        # sort canonical
        ordered = sorted(bars, key=lambda b: (b.symbol, b.timestamp))
        # dedupe: reject duplicates
        seen: set[tuple[str, datetime]] = set()
        unique: list[NormalizedBar] = []
        for b in ordered:
            key = (b.symbol, b.timestamp)
            if key in seen:
                raise FeatureEngineError(
                    f"Duplicate bar {b.symbol} @ {b.timestamp.isoformat()} — REJECT"
                )
            seen.add(key)
            unique.append(b)
        return unique

    def transform(self, bars: Sequence[NormalizedBar]) -> list[dict[str, Any]]:
        """Full historical feature calculation. Chronological output."""
        bars = self._validate_bars(bars)
        by_sym: dict[str, list[NormalizedBar]] = {}
        for b in bars:
            by_sym.setdefault(b.symbol, []).append(b)

        rows: list[dict[str, Any]] = []
        for symbol in sorted(by_sym.keys()):
            rows.extend(self._transform_symbol(by_sym[symbol]))
        # global chronological then symbol
        rows.sort(key=lambda r: (str(r["timestamp"]), r["symbol"]))
        return rows

    def transform_incremental(
        self,
        history: Sequence[NormalizedBar],
        new_bars: Sequence[NormalizedBar],
    ) -> list[dict[str, Any]]:
        """
        Incremental path: compute features for new_bars using history+new.
        Result for each new timestamp must equal full transform of history+new.
        """
        combined = list(history) + list(new_bars)
        full = self.transform(combined)
        new_keys = {(b.symbol, b.timestamp.isoformat()) for b in new_bars}
        return [
            r
            for r in full
            if (r["symbol"], str(r["timestamp"])) in new_keys
            or (r["symbol"], r["timestamp"]) in {(b.symbol, b.timestamp.isoformat()) for b in new_bars}
        ]

    def _transform_symbol(self, bars: list[NormalizedBar]) -> list[dict[str, Any]]:
        n = len(bars)
        closes = [self._price(b) for b in bars]
        opens = [self._open(b) for b in bars]
        highs = [self._high(b) for b in bars]
        lows = [self._low(b) for b in bars]
        volumes = [float(b.volume) for b in bars]

        ema10 = ema_series(closes, 10)
        ema20 = ema_series(closes, 20)
        ema50 = ema_series(closes, 50)
        ema12 = ema_series(closes, 12)
        ema26 = ema_series(closes, 26)

        macd_line = [
            (ema12[i] - ema26[i]) if not (math.isnan(ema12[i]) or math.isnan(ema26[i])) else float("nan")
            for i in range(n)
        ]
        # signal needs contiguous macd; treat nan as break — use only valid suffix seed
        macd_signal = self._ema_on_maybe_nan(macd_line, 9)
        macd_hist = [
            (macd_line[i] - macd_signal[i])
            if not (math.isnan(macd_line[i]) or math.isnan(macd_signal[i]))
            else float("nan")
            for i in range(n)
        ]

        trs: list[float] = []
        for i in range(n):
            pc = closes[i - 1] if i > 0 else None
            trs.append(true_range(highs[i], lows[i], pc))

        ret1 = [float("nan")] * n
        for i in range(1, n):
            if closes[i - 1] != 0:
                ret1[i] = closes[i] / closes[i - 1] - 1.0

        rows: list[dict[str, Any]] = []
        vol_hist: list[float] = []
        vol_ratio_hist: list[float] = []

        for i in range(n):
            bar = bars[i]
            ts = bar.timestamp.isoformat()
            row = empty_feature_row(bar.symbol, ts, self.timeframe)

            # returns
            row["return_1d"] = ret1[i]
            row["return_5d"] = (
                closes[i] / closes[i - 5] - 1.0 if i >= 5 and closes[i - 5] != 0 else float("nan")
            )
            row["return_20d"] = (
                closes[i] / closes[i - 20] - 1.0 if i >= 20 and closes[i - 20] != 0 else float("nan")
            )

            # SMA / EMA
            row["sma_5"] = sma(closes, 5, i)
            row["sma_10"] = sma(closes, 10, i)
            row["sma_20"] = sma(closes, 20, i)
            row["sma_50"] = sma(closes, 50, i)
            row["ema_10"] = ema10[i]
            row["ema_20"] = ema20[i]
            row["ema_50"] = ema50[i]

            s20, s50, s5 = row["sma_20"], row["sma_50"], row["sma_5"]
            row["price_vs_sma20"] = (
                closes[i] / s20 - 1.0 if s20 and not math.isnan(s20) and s20 != 0 else float("nan")
            )
            row["price_vs_sma50"] = (
                closes[i] / s50 - 1.0 if s50 and not math.isnan(s50) and s50 != 0 else float("nan")
            )
            row["sma5_vs_sma20"] = (
                s5 / s20 - 1.0
                if s5 is not None and s20 is not None and not math.isnan(s5) and not math.isnan(s20) and s20 != 0
                else float("nan")
            )
            row["sma20_vs_sma50"] = (
                s20 / s50 - 1.0
                if s20 is not None and s50 is not None and not math.isnan(s20) and not math.isnan(s50) and s50 != 0
                else float("nan")
            )

            row["rsi_14"] = rsi(closes, 14, i)
            row["macd"] = macd_line[i]
            row["macd_signal"] = macd_signal[i]
            row["macd_histogram"] = macd_hist[i]

            row["true_range"] = trs[i]
            row["atr_14"] = sma(trs, 14, i)
            row["rolling_volatility_20"] = rolling_std(ret1, 20, i)
            row["rolling_range_20"] = sma([highs[j] - lows[j] for j in range(n)], 20, i)

            row["volume_sma_20"] = sma(volumes, 20, i)
            vs = row["volume_sma_20"]
            row["volume_ratio_20"] = (
                volumes[i] / vs if vs and not math.isnan(vs) and vs != 0 else float("nan")
            )
            row["volume_change_1d"] = (
                volumes[i] / volumes[i - 1] - 1.0 if i >= 1 and volumes[i - 1] != 0 else float("nan")
            )

            # price action
            row["daily_range"] = highs[i] - lows[i]
            row["body_size"] = abs(closes[i] - opens[i])
            row["upper_wick"] = highs[i] - max(opens[i], closes[i])
            row["lower_wick"] = min(opens[i], closes[i]) - lows[i]
            dr = highs[i] - lows[i]
            row["close_position_in_range"] = (
                (closes[i] - lows[i]) / dr if dr > 0 else float("nan")
            )
            row["gap_from_previous_close"] = (
                opens[i] / closes[i - 1] - 1.0 if i >= 1 and closes[i - 1] != 0 else float("nan")
            )

            # regimes — causal percentile on expanding window ending at i
            pvs = row["price_vs_sma20"]
            if pvs is not None and not math.isnan(pvs):
                row["trend_regime"] = 1.0 if pvs > 0 else (-1.0 if pvs < 0 else 0.0)
            else:
                row["trend_regime"] = float("nan")

            rv = row["rolling_volatility_20"]
            if rv is not None and not math.isnan(rv):
                vol_hist.append(rv)
                row["volatility_regime"] = causal_percentile_rank(vol_hist, rv)
            else:
                row["volatility_regime"] = float("nan")

            vr = row["volume_ratio_20"]
            if vr is not None and not math.isnan(vr):
                vol_ratio_hist.append(vr)
                row["volume_regime"] = causal_percentile_rank(vol_ratio_hist, vr)
            else:
                row["volume_regime"] = float("nan")

            # warm-up: need at least sma_50 window for "fully valid"
            row["feature_valid"] = i >= 49 and not math.isnan(row["sma_50"])

            # sanitize inf
            for k, v in list(row.items()):
                if isinstance(v, float) and (math.isinf(v)):
                    row[k] = float("nan")

            rows.append(row)
        return rows

    @staticmethod
    def _ema_on_maybe_nan(values: Sequence[float], window: int) -> list[float]:
        n = len(values)
        out = [float("nan")] * n
        # find first index where we have `window` consecutive non-nan
        i = 0
        while i < n:
            if math.isnan(values[i]):
                i += 1
                continue
            # count forward
            j = i
            while j < n and not math.isnan(values[j]):
                j += 1
            segment = list(values[i:j])
            if len(segment) >= window:
                ema = ema_series(segment, window)
                for k, val in enumerate(ema):
                    out[i + k] = val
            i = j if j > i else i + 1
        return out

    def metadata(self) -> dict[str, Any]:
        return {k: v.to_dict() for k, v in FEATURE_METADATA.items()}

    @staticmethod
    def columns() -> list[str]:
        return list(FEATURE_COLUMNS)


class FeatureQualityEngine:
    """Validate feature rows — no silent fixes."""

    def check(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        seen: set[tuple[str, str]] = set()
        for r in rows:
            status = "VALID"
            reasons: list[str] = []
            key = (str(r.get("symbol")), str(r.get("timestamp")))
            if key in seen:
                status = "REJECTED"
                reasons.append("DUPLICATE_FEATURE_ROW")
            seen.add(key)
            if r.get("timeframe") not in ("1d", "1D", CANONICAL_TIMEFRAME):
                status = "REJECTED"
                reasons.append("WRONG_TIMEFRAME")
            if not r.get("symbol"):
                status = "REJECTED"
                reasons.append("INVALID_SYMBOL")
            for k, v in r.items():
                if isinstance(v, float) and math.isinf(v):
                    status = "FLAGGED" if status == "VALID" else status
                    reasons.append("INF_VALUE")
                    break
            if not r.get("feature_valid"):
                reasons.append("INSUFFICIENT_HISTORY")
                if status == "VALID":
                    status = "FLAGGED"
            results.append({"status": status, "reasons": reasons or ["OK"], "row": r})
        return results


class TrainOnlyNormalizer:
    """
    Fit statistics on TRAIN only; transform any split.
    Prevents global mean/std leakage across train/validation.
    """

    def __init__(self, columns: Optional[list[str]] = None) -> None:
        self.columns = columns or [
            c
            for c in FEATURE_COLUMNS
            if c not in ("symbol", "timestamp", "timeframe", "feature_valid")
            and not c.endswith("_regime")
        ]
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.fitted = False

    def fit(self, rows: Sequence[dict[str, Any]]) -> "TrainOnlyNormalizer":
        for col in self.columns:
            vals = [
                float(r[col])
                for r in rows
                if r.get(col) is not None and not math.isnan(float(r[col]))
            ]
            if not vals:
                self.means[col] = 0.0
                self.stds[col] = 1.0
                continue
            m = sum(vals) / len(vals)
            var = sum((x - m) ** 2 for x in vals) / len(vals)
            self.means[col] = m
            self.stds[col] = math.sqrt(var) if var > 0 else 1.0
        self.fitted = True
        return self

    def transform(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.fitted:
            raise FeatureEngineError("Normalizer not fitted — fit on TRAIN only")
        out = []
        for r in rows:
            nr = dict(r)
            for col in self.columns:
                v = r.get(col)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    nr[col] = float("nan")
                else:
                    nr[col] = (float(v) - self.means[col]) / self.stds[col]
            out.append(nr)
        return out
