# IDX Signal Bot (Indonesian Stock Exchange)

**Signal-only system for IDX equities.**  
Paper trading with virtual balance **Rp 10.000.000**.  
**Live trading is permanently disabled.**

---

## Purpose

This project produces trading **signals** for Indonesian stocks (IDX) and can simulate them via paper trading. It does **not** place real orders, connect to brokers for execution, or move real money.

| Mode            | Status                          |
|-----------------|---------------------------------|
| Signal generation | Target (later phases)         |
| Paper trading   | Foundation ready (PHASE 1–2)    |
| Live trading    | **DISABLED forever**            |
| Telegram alerts | Flag present; off by default    |

---

## Architecture (target)

```
MARKET DATA
    ↓
DATA NORMALIZATION
    ↓
FEATURE ENGINEERING
    ↓
ML ENSEMBLE
    ↓
SIGNAL ENGINE
    ↓
RISK ENGINE
    ↓
PAPER TRADING
    ↓
PORTFOLIO STATE
    ↓
TELEGRAM
```

GitHub Actions will orchestrate the pipeline in later phases.  
PHASE 1 delivered contracts, configuration, safety, storage abstraction, and tests.
PHASE 2 adds autonomous batch runtime, market clock, calendar, scheduler workflows,
idempotency, atomic state, healthcheck, and structured logging.

---

## Safety

- `LIVE_TRADING = false` by default and **forced false** by validation.
- Setting `LIVE_TRADING=true` (env or code) raises a **fatal configuration error** (fail-closed).
- No broker execution module exists.
- Secrets (Telegram token, API keys, etc.) must come from environment variables / GitHub Secrets — never committed.

---

## Paper account

Initial virtual balance:

```text
cash            = 10_000_000 IDR
equity          = 10_000_000 IDR
realized_pnl    = 0
unrealized_pnl  = 0
positions       = {}
```

---

## Configuration

Key defaults (see `src/idxbot/config/settings.py`):

| Key               | Default        |
|-------------------|----------------|
| MARKET            | IDX            |
| CURRENCY          | IDR            |
| PAPER_TRADING     | true           |
| LIVE_TRADING      | false          |
| INITIAL_BALANCE   | 10_000_000     |
| TELEGRAM_ENABLED  | false          |
| TIMEZONE          | Asia/Jakarta   |

Copy `.env.example` → `.env` for local overrides. Never commit `.env`.

---

## Storage strategy

GitHub Actions runners are **ephemeral**. Production state must use an object-storage backend (S3-compatible).  

PHASE 1 provides:

- `StorageBackend` abstract interface (`load_state` / `save_state`)
- `LocalStorageBackend` for local tests
- `ObjectStorageBackend` contract (implementation later)

No cloud credentials are hardcoded.

---


---

## CLI (PHASE 2)

```bash
python -m idxbot --help
python -m idxbot --health
python -m idxbot --dry-run
python -m idxbot --scheduled-at 2024-06-17T09:00:00+07:00 --dry-run --json
```

## Local development

```bash
# Python 3.11+
python -m venv .venv
source .venv/bin/activate   # or Windows equivalent
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/ -v
```

---

## Testing

```bash
pytest tests/ -v --tb=short
```

Coverage includes:

1. Configuration defaults  
2. LIVE_TRADING safety guard  
3. Paper balance initialization  
4. MarketData validation  
5. Timezone-aware timestamp enforcement  
6. Storage save/load  
7. State serialization  
8. Invalid configuration rejection  

---

## Security

See [SECURITY.md](SECURITY.md) and `.gitignore`.  
Simple secret scan should be run before any push.

---

## 15-phase roadmap (high level)

| Phase | Focus                                      |
|-------|--------------------------------------------|
| 1     | Foundation & architecture (this phase)     |
| 2     | GitHub Actions scheduler & autonomous runtime |
| 3–5   | Market data, normalization, features       |
| 6–7   | ML ensemble                                |
| 8–9   | Signal & risk engines                      |
| 10–11 | Paper trading engine & portfolio           |
| 12    | Telegram notifications                     |
| 13–14 | Hardening, metrics, observability          |
| 15    | Final packaging & ZIP for user push        |

**This system is not intended for live trading.** Live trading is out of scope by design.

---

## License

MIT (see project metadata).

---

## Explicit Safety Statement

**THIS SYSTEM DOES NOT EXECUTE LIVE TRADES.**

- `LIVE_TRADING` is hard-disabled and cannot be turned on.
- There is no broker execution module.
- The only outputs are `OrderIntent` objects (BUY / SELL / HOLD) and paper-portfolio simulation.
- Telegram receives signal notifications; it does not place orders.

## Local run

```bash
export PYTHONPATH=src
python -m idxbot --help
python -m idxbot --health
python -m idxbot --dry-run
```

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

Adversarial leakage tests live under `tests/adversarial/` and **raise** on detected leakage.

## GitHub Secrets (required for Telegram)

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Never commit secrets. Never log token values.

## Known limitations

- Full ML training requires scikit-learn (and optionally pyarrow for Parquet).
- GitHub Actions Free has strict CPU/RAM; governor enforces degradation.
- Market data providers may be rate-limited or incomplete; system fails safe.
- No profitability claims. Evaluation prioritizes correctness, anti-leakage, and stability over raw return.

---

## Production orchestration (SIGNAL / LEARNING)

| Mode | When (Asia/Jakarta) | Entry | Heavy training |
|------|---------------------|-------|----------------|
| **SIGNAL** | Mon–Fri | `python -m idxbot --mode SIGNAL` | No |
| **LEARNING** | Sat–Sun | `python -m idxbot --mode LEARNING` | Yes, ≤19 min budget |
| **auto** | Detect weekday | `python -m idxbot --mode auto` | Per day |

### Paper trading only

```text
PAPER_TRADING_ONLY=true
LIVE_TRADING=false   # forced; any true value → fatal config error
Initial capital    = Rp 10.000.000
```

There is **no broker API** and **no real order path**.

### Reset paper account (preserves ML)

```bash
python -m idxbot --reset-paper --force
# or
python -m idxbot.trading.reset --force
```

Resets only:

- `paper/portfolio.json`, `positions.json`, `trades.json`
- `.state` portfolio snapshot

Does **not** delete:

- `.models/`, `.experience/`, `ml/models/`, `ml/experience/`, `ml/metadata/`
- champion registry, training history

### Compute providers

- **LocalComputeProvider** — default, always available, respects 19-minute budget.
- **ColabComputeProvider** — optional (`ENABLE_COLAB=true` + endpoint). If unavailable, Governor falls back to local. Invalid Colab artifacts are rejected (checksum / schema / no arbitrary pickle).

### GitHub Actions

- `.github/workflows/signal.yml` — weekdays, SIGNAL mode
- `.github/workflows/learning.yml` — Saturday, LEARNING mode

Secrets (never commit):

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

### Exact commands

```bash
# Install
pip install -r requirements.txt
pip install -e .

# Health
PYTHONPATH=src python -m idxbot --health

# Signal (dry-run)
PYTHONPATH=src python -m idxbot --mode SIGNAL --dry-run --json

# Learning
PYTHONPATH=src python -m idxbot --mode LEARNING

# Paper reset (ML preserved)
PYTHONPATH=src python -m idxbot --reset-paper --force

# Tests
PYTHONPATH=src python -m pytest tests/ -q
```

### Safety invariants

1. `LIVE_TRADING` cannot be enabled (fail-closed).
2. No broker execution module.
3. Training uses `n_jobs=1`.
4. Temporal / purged splits; leakage adversarial tests.
5. Champion promotion only via promotion gate.
6. Telegram failures do not roll back paper accounting.
7. Colab failure → local fallback; never blocks SIGNAL mode.

**THIS SYSTEM IS PAPER TRADING ONLY. NO REAL BROKER EXECUTION.**
