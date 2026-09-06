# PHASE 4/15 — Feature Engineering & Technical Feature Pipeline

## Delivered

- Isolated `src/idxbot/features/` engine
- Canonical timeframe **1D** only
- Price mode: adjusted (default) or raw
- Features: returns, SMA, EMA, RSI, MACD, ATR, volatility, volume, price-action, regimes
- Warm-up → NaN + `feature_valid=false` (no fillna(0))
- Feature metadata registry
- Incremental calculation (`transform_incremental` == full recompute)
- `TrainOnlyNormalizer` (fit TRAIN → transform any split)
- Hard anti-lookahead / anti-leakage tests

## Non-goals

Labels, ML training, signals, Telegram, live trading.
