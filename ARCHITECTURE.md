# Architecture — IDX Autonomous Signal Intelligence Bot

**THIS SYSTEM DOES NOT EXECUTE LIVE TRADES.**

`LIVE_TRADING = False` permanently. No broker execution module exists.
Output is `OrderIntent` only: BUY / SELL / HOLD.

## Pipeline (ASCII)

```
DATA
 ↓
VALIDATION (schema, quality, stale, future-ts reject)
 ↓
FEATURES (deterministic, causal, versioned)
 ↓
DATASET (labels, purged windows)
 ↓
PURGED WALK FORWARD + DYNAMIC EMBARGO
 ↓
MODEL POOL (max 7, n_jobs=1, no GPU, no multiprocessing)
 ↓
OOF (purged out-of-fold)
 ↓
NNLS ENSEMBLE (non-negative weights, sum=1) + CALIBRATION
 ↓
VALIDATION GATE (anti-leakage, adversarial, baseline, stability)
 ↓
COMPUTATIONAL GOVERNOR (7 → 5 → 3 → 1 → SAFE_EXIT)
 ↓
PORTFOLIO GOVERNOR (paper only, exposure / cash / concentration)
 ↓
SIGNAL ENGINE (multi-horizon + hysteresis)
 ↓
ORDER INTENT (immutable, SHA-256 idempotent signal_id)
 ↓
TELEGRAM (secrets from env, limited retry, no secret logging)
```

## Core Safety Invariants

1. LIVE_TRADING forced false at config load; any attempt to set true raises.
2. No broker / execution path in codebase.
3. All ML uses `n_jobs=1`.
4. Test set never used for training, tuning, stacking, or feature selection.
5. Purged + embargo splits; adversarial tests raise on leakage.
6. Self-learning requires promotion gate; no auto-replace of production model.
7. Governor SAFE_EXIT → no ML, HOLD only.
8. Determinism: same input + seed + config → same OrderIntent.

## Resource Governor States

| State       | Active models | Behaviour                     |
|-------------|---------------|-------------------------------|
| FULL_7      | 7             | Full ensemble                 |
| DEGRADED_5  | 5             | Weighted fallback             |
| DEGRADED_3  | 3             | Weighted fallback             |
| DEGRADED_1  | 1             | Single model                  |
| SAFE_EXIT   | 0             | No ML, HOLD, health report    |

## Data Storage

Parquet partitioned:

```
data/market/symbol=BBCA.JK/year=2026/month=09/data.parquet
```

Atomic writes only. Historical data never silently overwritten.

## Timezone

Canonical: `Asia/Jakarta`. Internal timestamps documented and consistent.
GitHub Actions cron is UTC; conversion is explicit in docs/workflows.

## Persistence

GitHub Actions runners are ephemeral. `StateStore` abstraction with
`LocalAtomicStateStore` (temp → fsync → rename). Ephemeral mode is reported,
never pretended to be durable.
