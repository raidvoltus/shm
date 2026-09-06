# PHASE 9/15 — Computational Governor

## Role

Controls **computational budget only**:
- how many models to load (7→5→3→1→SAFE_EXIT)
- META vs WEIGHTED_FALLBACK vs SINGLE_MODEL
- retrain deferral

Does **not** control RiskPolicy, sizing, SL/TP, exposure, or paper balance.

## Resource detection

1. cgroup v2 `memory.max` / `memory.current`
2. cgroup v1 limit/usage
3. psutil / `/proc/meminfo` fallback

## Lazy health

Registry `meta.json` only — SHA256, versions, estimated_memory — **no** joblib load until subset chosen.

## Ensemble

| Active | Mode |
|--------|------|
| 7 + meta weights | META |
| 5 / 3 | WEIGHTED_FALLBACK (Σ weights = 1) |
| 1 | SINGLE_MODEL |
| 0 | SAFE_EXIT |
