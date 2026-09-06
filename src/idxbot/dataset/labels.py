"""
Multi-horizon label generator.

Future data MAY be used for labels (outcomes).
Future data must NOT influence thresholds (backward-looking only).
Incomplete horizons → PENDING (excluded from training).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence


class LabelClass(str, Enum):
    DOWN = "DOWN"
    FLAT = "FLAT"
    UP = "UP"
    PENDING = "PENDING"


@dataclass
class LabelResult:
    symbol: str
    timestamp: str
    horizon: int
    future_return: Optional[float]
    threshold: Optional[float]
    label: LabelClass
    label_start_idx: int
    label_end_idx: int  # exclusive end in bar index space (T+horizon)
    observation_idx: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "horizon": self.horizon,
            "future_return": self.future_return,
            "threshold": self.threshold,
            "label": self.label.value,
            "label_start_idx": self.label_start_idx,
            "label_end_idx": self.label_end_idx,
            "observation_idx": self.observation_idx,
        }


class LabelGenerator:
    """
    Extensible multi-horizon labeler.

    threshold_mode:
      - 'fixed': use absolute threshold
      - 'vol': threshold = vol_multiplier * rolling_vol(T)  [backward-looking]
    """

    def __init__(
        self,
        horizons: Sequence[int] = (1, 5, 20),
        *,
        fixed_threshold: float = 0.01,
        threshold_mode: str = "vol",
        vol_window: int = 20,
        vol_multiplier: float = 1.0,
        label_version: str = "1",
    ) -> None:
        if not horizons or any(h < 1 for h in horizons):
            raise ValueError("horizons must be positive integers")
        self.horizons = tuple(sorted(set(int(h) for h in horizons)))
        self.fixed_threshold = fixed_threshold
        self.threshold_mode = threshold_mode
        self.vol_window = vol_window
        self.vol_multiplier = vol_multiplier
        self.label_version = label_version

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    def embargo_period(self) -> int:
        """Dynamic embargo >= max horizon + 1."""
        return self.max_horizon + 1

    def _backward_vol(self, closes: Sequence[float], idx: int) -> float:
        """Rolling volatility of 1d returns using only indices ≤ idx."""
        if idx < self.vol_window:
            return float("nan")
        rets = []
        start = idx - self.vol_window + 1
        for i in range(start, idx + 1):
            if i < 1:
                continue
            prev = closes[i - 1]
            if prev == 0:
                continue
            rets.append(closes[i] / prev - 1.0)
        if len(rets) < max(5, self.vol_window // 2):
            return float("nan")
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        return math.sqrt(var)

    def threshold_at(self, closes: Sequence[float], idx: int) -> float:
        if self.threshold_mode == "fixed":
            return self.fixed_threshold
        vol = self._backward_vol(closes, idx)
        if math.isnan(vol):
            return self.fixed_threshold  # fallback when insufficient history
        return max(self.fixed_threshold * 0.1, self.vol_multiplier * vol)

    def label_symbol(
        self,
        symbol: str,
        timestamps: Sequence[str],
        closes: Sequence[float],
    ) -> list[LabelResult]:
        """
        Generate labels for one symbol series (oldest → newest).
        closes[i] is price at timestamps[i].
        """
        n = len(closes)
        assert n == len(timestamps)
        results: list[LabelResult] = []
        for h in self.horizons:
            for i in range(n):
                thr = self.threshold_at(closes, i)
                end_idx = i + h
                if end_idx >= n:
                    results.append(
                        LabelResult(
                            symbol=symbol,
                            timestamp=timestamps[i],
                            horizon=h,
                            future_return=None,
                            threshold=thr if not math.isnan(thr) else None,
                            label=LabelClass.PENDING,
                            label_start_idx=i,
                            label_end_idx=end_idx,
                            observation_idx=i,
                        )
                    )
                    continue
                c0, c1 = closes[i], closes[end_idx]
                if c0 == 0:
                    fut = float("nan")
                else:
                    fut = c1 / c0 - 1.0
                if math.isnan(fut) or math.isnan(thr):
                    lab = LabelClass.PENDING
                elif fut > thr:
                    lab = LabelClass.UP
                elif fut < -thr:
                    lab = LabelClass.DOWN
                else:
                    lab = LabelClass.FLAT
                results.append(
                    LabelResult(
                        symbol=symbol,
                        timestamp=timestamps[i],
                        horizon=h,
                        future_return=fut,
                        threshold=thr,
                        label=lab,
                        label_start_idx=i,
                        label_end_idx=end_idx,
                        observation_idx=i,
                    )
                )
        return results
