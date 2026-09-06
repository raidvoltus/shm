# PHASE 5/15 — Dataset Construction, Labeling & Experience Store

## Label methodology

- Horizons: 1D, 5D, 20D (configurable)
- Classes: UP / FLAT / DOWN / PENDING
- Threshold: fixed or volatility-adjusted (**backward-looking only** at T)
- Incomplete future horizon → **PENDING** (excluded from training)

## Embargo / purge

- `embargo_period = max(horizons) + 1` (dynamic)
- Label interval `[observation, observation+horizon)` purged across train/val
- Walk-forward expanding windows

## Experience Store

```
experiences/year=YYYY/month=MM/experiences.parquet
```

- Deterministic `experience_id`
- Idempotent append (dedup)
- Independent of paper account state/reset

## Guarantees

- Features at T ignore data after T
- Threshold at T ignores data after T
- Train class distribution ignores val/test labels
- Duplicate append does not increase unique count
