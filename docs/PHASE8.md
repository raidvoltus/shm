# PHASE 8/15 — Model Validation & Walk-Forward Evaluation

## Rolling walk-forward

- Bounded `max_train_lookback` (not unbounded expanding)
- Dynamic embargo = max(horizons)+1
- Purge gap ≥ embargo; no train/val overlap

## Final Test

- Locked via `FinalTestGuard`
- Training access → `FinalTestViolation`
- One-way OOS report only after model lock

## Statistics

- Wilcoxon signed-rank on paired fold losses
- Holm correction for multiple comparisons
- Stability: mean/median/std/min/max/worst/best fold
- Champion decision: WIN / HOLD / FAIL (no live promotion)

## Constraints

n_jobs=1, serial folds, Experience Store untouched, LIVE_TRADING=false
