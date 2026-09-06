# PHASE 3/15 — IDX Market Data Engine & Data Normalization

## Delivered

- `MarketDataProvider` abstraction (vendor-swappable)
- `FixtureProvider` deterministic scenarios (split, reverse-split, 429, timeout, stale, …)
- Dynamic IDX universe + `LiquidityFilter`
- Canonical `NormalizedBar` with explicit **raw_*** and **adjusted_*** fields
- MIXED adjustment rejected
- `DataQualityEngine`: VALID / FLAGGED / REJECTED + reason codes
- Abnormal gap → CORPORATE_ACTION_REVIEW (no silent correction)
- Partitioned Parquet store + delta files + `DataCompactor`
- Provenance + last_successful cursor
- Bounded retry / exponential backoff (`ProviderRetry`)
- Calendar integration (PHASE 2)

## Non-goals (still deferred)

ML prediction, signals, Telegram production, broker, live trading.
