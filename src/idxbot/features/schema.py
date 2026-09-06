"""Canonical feature schema, metadata, and field ordering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# Deterministic column order for feature rows
FEATURE_COLUMNS: list[str] = [
    "symbol",
    "timestamp",
    "timeframe",
    "return_1d",
    "return_5d",
    "return_20d",
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "ema_10",
    "ema_20",
    "ema_50",
    "price_vs_sma20",
    "price_vs_sma50",
    "sma5_vs_sma20",
    "sma20_vs_sma50",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "true_range",
    "atr_14",
    "rolling_volatility_20",
    "rolling_range_20",
    "volume_sma_20",
    "volume_ratio_20",
    "volume_change_1d",
    "daily_range",
    "body_size",
    "upper_wick",
    "lower_wick",
    "close_position_in_range",
    "gap_from_previous_close",
    "trend_regime",
    "volatility_regime",
    "volume_regime",
    "feature_valid",
]

CANONICAL_TIMEFRAME = "1d"


@dataclass(frozen=True)
class FeatureMeta:
    feature_name: str
    source_fields: tuple[str, ...]
    formula: str
    window: int
    timeframe: str = CANONICAL_TIMEFRAME
    price_mode: str = "adjusted"  # adjusted | raw
    adjustment_mode: str = "ADJUSTED"
    version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "source_fields": list(self.source_fields),
            "formula": self.formula,
            "window": self.window,
            "timeframe": self.timeframe,
            "price_mode": self.price_mode,
            "adjustment_mode": self.adjustment_mode,
            "version": self.version,
        }


FEATURE_METADATA: dict[str, FeatureMeta] = {
    "return_1d": FeatureMeta("return_1d", ("close",), "close(T)/close(T-1)-1", 1),
    "return_5d": FeatureMeta("return_5d", ("close",), "close(T)/close(T-5)-1", 5),
    "return_20d": FeatureMeta("return_20d", ("close",), "close(T)/close(T-20)-1", 20),
    "sma_5": FeatureMeta("sma_5", ("close",), "mean(close[T-4:T])", 5),
    "sma_10": FeatureMeta("sma_10", ("close",), "mean(close[T-9:T])", 10),
    "sma_20": FeatureMeta("sma_20", ("close",), "mean(close[T-19:T])", 20),
    "sma_50": FeatureMeta("sma_50", ("close",), "mean(close[T-49:T])", 50),
    "ema_10": FeatureMeta("ema_10", ("close",), "EMA(close,10)", 10),
    "ema_20": FeatureMeta("ema_20", ("close",), "EMA(close,20)", 20),
    "ema_50": FeatureMeta("ema_50", ("close",), "EMA(close,50)", 50),
    "price_vs_sma20": FeatureMeta("price_vs_sma20", ("close", "sma_20"), "close/sma20-1", 20),
    "price_vs_sma50": FeatureMeta("price_vs_sma50", ("close", "sma_50"), "close/sma50-1", 50),
    "sma5_vs_sma20": FeatureMeta("sma5_vs_sma20", ("sma_5", "sma_20"), "sma5/sma20-1", 20),
    "sma20_vs_sma50": FeatureMeta("sma20_vs_sma50", ("sma_20", "sma_50"), "sma20/sma50-1", 50),
    "rsi_14": FeatureMeta("rsi_14", ("close",), "RSI(close,14)", 14),
    "macd": FeatureMeta("macd", ("close",), "EMA12-EMA26", 26),
    "macd_signal": FeatureMeta("macd_signal", ("macd",), "EMA(macd,9)", 35),
    "macd_histogram": FeatureMeta("macd_histogram", ("macd", "macd_signal"), "macd-signal", 35),
    "true_range": FeatureMeta("true_range", ("high", "low", "close"), "max(h-l,|h-pc|,|l-pc|)", 1),
    "atr_14": FeatureMeta("atr_14", ("true_range",), "mean(TR,14)", 14),
    "rolling_volatility_20": FeatureMeta(
        "rolling_volatility_20", ("return_1d",), "std(return_1d,20)", 20
    ),
    "rolling_range_20": FeatureMeta(
        "rolling_range_20", ("high", "low"), "mean(high-low,20)", 20
    ),
    "volume_sma_20": FeatureMeta("volume_sma_20", ("volume",), "mean(volume,20)", 20),
    "volume_ratio_20": FeatureMeta(
        "volume_ratio_20", ("volume", "volume_sma_20"), "volume/vol_sma20", 20
    ),
    "volume_change_1d": FeatureMeta(
        "volume_change_1d", ("volume",), "volume(T)/volume(T-1)-1", 1
    ),
    "daily_range": FeatureMeta("daily_range", ("high", "low"), "high-low", 1),
    "body_size": FeatureMeta("body_size", ("open", "close"), "abs(close-open)", 1),
    "upper_wick": FeatureMeta(
        "upper_wick", ("high", "open", "close"), "high-max(open,close)", 1
    ),
    "lower_wick": FeatureMeta(
        "lower_wick", ("low", "open", "close"), "min(open,close)-low", 1
    ),
    "close_position_in_range": FeatureMeta(
        "close_position_in_range",
        ("close", "high", "low"),
        "(close-low)/(high-low)",
        1,
    ),
    "gap_from_previous_close": FeatureMeta(
        "gap_from_previous_close", ("open", "close"), "open(T)/close(T-1)-1", 1
    ),
    "trend_regime": FeatureMeta(
        "trend_regime", ("price_vs_sma20",), "sign(price_vs_sma20)", 20
    ),
    "volatility_regime": FeatureMeta(
        "volatility_regime",
        ("rolling_volatility_20",),
        "percentile_rank of vol using past only",
        20,
    ),
    "volume_regime": FeatureMeta(
        "volume_regime", ("volume_ratio_20",), "percentile_rank of vol_ratio past only", 20
    ),
}


def empty_feature_row(symbol: str, timestamp: str, timeframe: str = CANONICAL_TIMEFRAME) -> dict[str, Any]:
    row: dict[str, Any] = {c: None for c in FEATURE_COLUMNS}
    row["symbol"] = symbol
    row["timestamp"] = timestamp
    row["timeframe"] = timeframe
    row["feature_valid"] = False
    return row
