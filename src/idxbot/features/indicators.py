"""
Pure technical indicator calculations — causal only (no lookahead).

All functions operate on sequences ordered oldest→newest and only use
indices ≤ current position.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence


def _nan() -> float:
    return float("nan")


def sma(values: Sequence[float], window: int, idx: int) -> float:
    if window <= 0 or idx + 1 < window:
        return _nan()
    chunk = values[idx - window + 1 : idx + 1]
    if any(math.isnan(x) for x in chunk):
        return _nan()
    return sum(chunk) / window


def ema_series(values: Sequence[float], window: int) -> list[float]:
    """Full EMA series; positions before warm-up are NaN."""
    n = len(values)
    out = [_nan()] * n
    if n < window or window <= 0:
        return out
    alpha = 2.0 / (window + 1)
    # seed with SMA
    seed = sum(values[:window]) / window
    out[window - 1] = seed
    prev = seed
    for i in range(window, n):
        prev = alpha * values[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def rsi(values: Sequence[float], window: int, idx: int) -> float:
    """Wilder RSI using only data up to idx."""
    if idx < window:
        return _nan()
    gains = []
    losses = []
    for i in range(idx - window + 1, idx + 1):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = sum(gains) / window
    avg_loss = sum(losses) / window
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def true_range(high: float, low: float, prev_close: Optional[float]) -> float:
    if prev_close is None or math.isnan(prev_close):
        return high - low
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def rolling_std(values: Sequence[float], window: int, idx: int) -> float:
    if idx + 1 < window:
        return _nan()
    chunk = values[idx - window + 1 : idx + 1]
    if any(math.isnan(x) for x in chunk):
        return _nan()
    mean = sum(chunk) / window
    var = sum((x - mean) ** 2 for x in chunk) / window
    return math.sqrt(var)


def causal_percentile_rank(history: Sequence[float], value: float) -> float:
    """Percentile rank of value among history (including value). Causal only."""
    valid = [x for x in history if not math.isnan(x)]
    if not valid:
        return _nan()
    below = sum(1 for x in valid if x < value)
    return below / len(valid)
