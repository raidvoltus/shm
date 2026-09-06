# PHASE 7/15 — Ensemble & Meta-Learner

## Pipeline

```
7 base models → purged WF OOF (disk parquet) → NNLS meta weights (≥0, Σ=1)
  → time-decay (causal) → weighted ensemble → independent calibration → registry
```

## Constraints

- n_jobs=1, no GPU, serial OOF, float32 probs
- Final test never used for fit/tune/calibration
- Dynamic subset: 3 / 5 / 7 models supported
- Governor (PHASE 9) decides active model count — not here
- RiskPolicy / LIVE_TRADING untouched
