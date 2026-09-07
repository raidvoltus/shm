# Cross-run signal delivery idempotency

## Guarantee

**AT-MOST-ONCE** (conditional / best-effort), **not** exactly-once.

Prefer a missed Telegram message over a duplicate BUY/SELL.

Crash window remains between Telegram HTTP completion and ledger finalize.

## SignalID

`SHA256(symbol | timestamp_bucket | feature_version | model_version | intent)`

## Backend

- Branch: `idxbot-state` / `idempotency/ledger.json`
- GitHub Contents API + `GITHUB_TOKEN` (`permissions.contents: write`)
- Not used: GitHub Actions artifacts

## State machine

| From | To | Trigger |
|------|-----|---------|
| ABSENT | PENDING | reserve() before send |
| PENDING | SUCCESS | Telegram 2xx |
| PENDING | UNKNOWN_DELIVERY_STATE | Timeout / ambiguous response |
| PENDING | PERMANENT_FAILURE | HTTP 4xx (not 429) |
| PENDING | TRANSIENT_FAILURE | Exhausted in-process retries |
| PENDING (lease expired) | UNKNOWN_DELIVERY_STATE | Safety: may have delivered |
| SUCCESS / UNKNOWN / PERMANENT | terminal (TTL block) | — |

## PENDING expiry

**Policy:** promote to `UNKNOWN_DELIVERY_STATE`, remain blocking for TTL (24h).

Do **not** allow automatic resend after 30m lease — Telegram may already have accepted.

## Failure policy

`IDXBOT_LEDGER_REQUIRED=true` (default on GHA):

- Ledger unavailable / SHA conflict / corrupt → **do not send** BUY/SELL.

Corrupt ledger is **never** silently reinitialized.

## Concurrency

`signal.yml`: `concurrency.group: idx-signal`, `cancel-in-progress: false`.
