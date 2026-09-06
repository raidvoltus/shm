# PHASE 2/15 — GitHub Actions Scheduler & Autonomous Runtime

## Summary

Autonomous batch runtime with:

- `python -m idxbot` entry point (`--help`, `--health`, `--dry-run`)
- Asia/Jakarta market clock + 30-minute scheduler latency tolerance
- Market calendar interface (SESSION_1 / SESSION_2 / BREAK / CLOSED / POST_MARKET)
- Fail-closed calendar (provider failure → CLOSED)
- Atomic LocalStorageBackend (temp → fsync → validate → os.replace)
- State version / optimistic concurrency
- Idempotency store
- Bounded retry for transient errors; no retry for fatal
- Structured logging with secret redaction
- Workflows: `market_runtime.yml`, `healthcheck.yml` (concurrency cancel-in-progress: false)

## Non-goals (still deferred)

ML, signals, Telegram production, broker, live trading.
