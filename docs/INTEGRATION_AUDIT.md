# INTEGRATION AUDIT — IDX Autonomous Signal Intelligence Bot

Date: 2026-09-01
Auditor stance: production SRE / quant engineer, evidence-only.

## RUNTIME PATH (current)

```
python -m idxbot
  → MarketRuntime.__init__
  → MarketRuntime.run()
       assert_no_live_trading
       MarketClock.build_context + latency gate
       StaticIDXCalendar session check
       IdempotencyStore.already_executed
       load PortfolioState (LocalStorageBackend)
       [PHASE-2 STUB] only heartbeat metrics write
       save PortfolioState
       mark idempotency completed
  → RuntimeResult(OK|SKIP|FATAL|CONFLICT)
```

**Telegram, SignalEngine, Governor, FeatureEngine, ModelPool, Ensemble, ExperienceStore are NOT called from MarketRuntime.run().**

Health path (`--health`) only checks config/safety/storage/calendar.

## MODULE STATUS

| MODULE | STATUS | NOTES |
|--------|--------|-------|
| idxbot.config.settings | WIRED | Used by runtime, safety, governor |
| idxbot.core.safety | WIRED | assert_no_live_trading in run() |
| idxbot.runtime.runner (MarketRuntime) | PARTIALLY_WIRED | Orchestration skeleton only; no ML/data/signal |
| idxbot.runtime.clock | WIRED | latency + context |
| idxbot.runtime.idempotency | WIRED | identity key |
| idxbot.runtime.retry | WIRED | retry policy |
| idxbot.calendar.static | WIRED | trading day / session |
| idxbot.storage.backend / state | WIRED | portfolio persistence (local atomic) |
| idxbot.logging.structured | WIRED | structured logger |
| idxbot.data.providers.base | PARTIALLY_WIRED | Interface exists; only FixtureProvider implemented |
| idxbot.data.providers.fixture | WIRED (tests) | Deterministic test provider; NOT production |
| idxbot.data.normalize | PARTIALLY_WIRED | Exists; not called from runtime |
| idxbot.data.quality.engine | PARTIALLY_WIRED | Exists; not called from runtime |
| idxbot.data.store.parquet_store | PARTIALLY_WIRED | Exists; not called from runtime |
| idxbot.data.universe | PARTIALLY_WIRED | Exists; not called from runtime |
| idxbot.data.rate_limit | DEAD | No incoming production calls observed |
| idxbot.features.engine | PARTIALLY_WIRED | Full causal engine; not in runtime path |
| idxbot.features.indicators / schema | PARTIALLY_WIRED | Used by FeatureEngine only |
| idxbot.dataset.builder / labels / split | PARTIALLY_WIRED | Purged split + labels exist; not runtime-wired |
| idxbot.ml.candidates / trainer / registry / promotion | PARTIALLY_WIRED | 7-candidate pool, n_jobs=1 design; not runtime-wired |
| idxbot.ensemble.* | PARTIALLY_WIRED | OOF, NNLS meta, calibrate, weighted; not runtime-wired |
| idxbot.governor.* | PARTIALLY_WIRED | FULL_7→SAFE_EXIT ladder exists; not called from MarketRuntime |
| idxbot.validation.* | PARTIALLY_WIRED | Walk-forward / final_test; used by tests/scripts, not daily runtime |
| idxbot.experience.store | PARTIALLY_WIRED | Append-only store; not called from runtime |
| idxbot.signals.order_intent / hysteresis / multi_horizon / signal_engine | PARTIALLY_WIRED | New contracts present; **not called from MarketRuntime** |
| idxbot.telegram.notifier | PARTIALLY_WIRED | Implemented; **not called from MarketRuntime** |
| idxbot.risk | DEAD | Empty package (__init__ only) |
| idxbot.trading | DEAD | Empty package (__init__ only) |

## DUPLICATE / DUAL SOURCE OF TRUTH

- `configs/` (empty-ish top-level) vs `config/` package — minor.
- No competing OrderIntent definitions found.
- Governor vs Risk: risk package is empty; portfolio checks live only in storage/state so far.

## DEPENDENCY CYCLES

None detected in static import graph for core path. ML/ensemble/governor form a clean DAG when used together.

## ACTION REQUIRED (priority order)

1. **Wire single Autonomous pipeline inside MarketRuntime.run()** (or extract AutonomousSignalRuntime and call it).
2. Introduce real-or-fail-safe provider registry (Fixture only for tests; production path must report NO_DATA / DEGRADED, never fabricate).
3. Call Data Quality gate → skip symbol on failure.
4. Call Governor.decide() → respect SAFE_EXIT (no ML).
5. Call FeatureEngine + Model inference / ensemble under governor budget.
6. Call SignalEngine (multi-horizon + hysteresis) with persistent previous_intent.
7. Call Portfolio constraint checks (cash / exposure) → may force HOLD.
8. Persist OrderIntent + ExperienceStore.
9. Telegram after persistence; HOLD default silent.
10. Explicit persistence_mode = EPHEMERAL | ARTIFACT | EXTERNAL; health DEGRADED when ephemeral.
11. `python -m idxbot run` alias (keep existing flags).
12. Integration tests that execute the full path on synthetic deterministic data.
13. Remove or clearly mark DEAD packages (risk, trading) or implement minimal portfolio governor there.

## PERSISTENCE REALITY

LocalStorageBackend writes under `.state/`. GitHub Actions runners are ephemeral → state lost between runs unless artifacts / external store / state branch is used. Current health does not yet surface `persistence_mode`.

## CONCLUSION OF PHASE A

Foundation modules exist and many unit/adversarial tests pass.
**End-to-end autonomous path is not wired.** Daily scheduler currently produces only portfolio heartbeat + SKIP/OK, never OrderIntent or Telegram signal.

Next: PHASE B — implement explicit sequential pipeline in runtime without rewriting already-correct components.

---

## POST-WIRING UPDATE (2026-09-01)

### Modules newly WIRED into MarketRuntime

| MODULE | STATUS AFTER |
|--------|--------------|
| idxbot.runtime.pipeline.AutonomousPipeline | WIRED |
| idxbot.data.providers.registry | WIRED |
| idxbot.governor.ComputationalGovernor | WIRED |
| idxbot.signals.SignalEngine / OrderIntent | WIRED |
| idxbot.portfolio.governor | WIRED |
| idxbot.telegram.notifier | WIRED (after intents) |
| Fixture provider path | WIRED with DEGRADED health |

### Runtime execution graph (current)

```
python -m idxbot [--dry-run]
  → MarketRuntime.run
       safety / latency / calendar / idempotency
       load PortfolioState
       AutonomousPipeline.run
            Governor.decide → FULL_7…SAFE_EXIT
            ProviderRegistry.fetch (fixture|real)
            quality gate per symbol
            multi-horizon momentum proxy (causal)
            SignalEngine (hysteresis + horizons)
            PortfolioGovernor (cash/exposure → may HOLD)
            OrderIntent (SHA-256 idempotent)
            Telegram (BUY/SELL only)
       save previous_intents into portfolio metrics
       mark idempotency
```

### Still PARTIALLY_WIRED / deferred

- Full sklearn model load from ModelRegistry + OOF ensemble on every daily run (resource-heavy; currently causal momentum proxy under governor budget; registry path ready).
- Real production market-data provider (only Fixture registered; health=DEGRADED when fixture).
- Persistent hysteresis across GH Actions runs relies on artifact upload/download (EPHEMERAL if artifact missing).
- risk/ and trading/ packages still empty (portfolio governor lives under portfolio/).

### Honest limitations

1. Without `IDXBOT_USE_FIXTURE=true` and no real provider → FAILED / no BUY/SELL.
2. persistence_mode reported as EPHEMERAL; artifact best-effort only.
3. Momentum proxy is deterministic and causal but is **not** a claim of ML edge.
4. LIVE_TRADING remains impossible to enable.
