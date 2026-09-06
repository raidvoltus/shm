# Security Policy — IDX Signal Bot

## Principles

1. **No live trading** — `LIVE_TRADING` is forced `false`. Any attempt to enable it raises a fatal configuration error (fail-closed).
2. **No secrets in source** — Telegram tokens, API keys, passwords, private keys must never appear in the repository.
3. **Secrets via environment / GitHub Secrets only**.
4. **Ephemeral runners** — Do not rely on local filesystem of GitHub Actions for production state as durable database.

## Protected paths (see `.gitignore`)

- `.env`, `*.pem`, `*.key`
- `credentials/`, `secrets/`
- `.state/`, `.cache/`, model artifacts

## Reporting

If you discover a credential or security issue in this repository, do **not** open a public issue. Contact the repository owner privately and rotate the credential immediately.

## PHASE 1 scope

This phase implements configuration guards, storage abstraction, and data contracts only. No broker APIs, no Telegram production tokens, no cloud credentials are present.

## CI trustworthiness

Mandatory pipeline steps must **not** use `|| true` or `continue-on-error: true`.

- Install, tests, safety guards, SIGNAL/LEARNING runs fail the job on non-zero exit.
- `continue-on-error` is only allowed on optional first-run artifact restore (no prior artifact).

False-success patterns are treated as security defects: a red X must mean a real failure.

## Storage

S3 / object storage is **not used**. State is local atomic filesystem only.
Do not add IDXBOT_S3_* secrets.
