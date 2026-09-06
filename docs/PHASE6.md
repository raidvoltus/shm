# PHASE 6/15 — ML Model Pool

## Candidates (7)

| Algorithm | Notes |
|-----------|--------|
| logistic_regression | Locked baseline, deterministic, n_jobs=1 |
| random_forest | n_estimators=50, max_depth=6, n_jobs=1 |
| extra_trees | n_jobs=1 |
| hist_gradient_boosting | RESOURCE_HEAVY |
| gradient_boosting | RESOURCE_HEAVY |
| ridge_classifier | Regularized linear |
| regime_aware_logreg | Causal regime features from Phase 4 |

Active count at runtime is **not** fixed to 7 — Computational Governor (PHASE 9) decides.

## Protocol

```
TRAIN → fit model + preprocessor
VALIDATION → probability calibration (prefit sigmoid)
TEST → holdout metrics only (no fit, no tune, no promotion)
```

## Registry

```
models/algorithm=<name>/version=<ver>/model.joblib + meta.json
```

SHA256 verified before joblib load. Feature/dataset/env handshake. Immutable versions.

## Champion / Challenger

Margin of Equivalency default **2%** on `balanced_accuracy`. Below margin → NOT_PROMOTED.

## Constraints

- n_jobs=1 everywhere
- No GPU / CUDA
- float32 features
- GitHub Free oriented
