# Cross-run signal delivery idempotency

## Guarantee

**AT-MOST-ONCE** notification for BUY/SELL when the ledger is required.

Prefer a missed Telegram message over a duplicate BUY/SELL.

Not exactly-once: a crash between Telegram HTTP success and ledger finalize can still allow a later retry after PENDING lease expiry.

## SignalID

`SHA256(symbol | timestamp_bucket | feature_version | model_version | intent)`

Deterministic identity; ledger provides persistent idempotency.

## Backend

- Branch: `idxbot-state`
- Path: `idempotency/ledger.json`
- API: GitHub Contents API + `GITHUB_TOKEN` (`permissions.contents: write`)
- Alternatives: `IDXBOT_LEDGER_BACKEND=file|memory` for tests/local

**Not used:** GitHub Actions artifacts (not a database; cannot restore across runs reliably).

## TTL / bounds

- TTL: 24 hours
- MAX_ENTRIES: 500
- PENDING lease: 30 minutes

## Delivery states

| Status | Effect |
|--------|--------|
| PENDING | Reservation; blocks duplicate while lease valid |
| SUCCESS | Blocks re-send for TTL |
| UNKNOWN_DELIVERY_STATE | Blocks re-send (timeout may have delivered) |
| PERMANENT_FAILURE | No auto cross-run retry |
| TRANSIENT_FAILURE | In-process retry only; may finalize after exhausted |

## Failure policy

`IDXBOT_LEDGER_REQUIRED=true` (default on GitHub Actions):

Ledger unavailable → **do not send** BUY/SELL (fail-closed).

## Concurrency

`signal.yml` concurrency group `idx-signal` with `cancel-in-progress: false` serializes writers.
