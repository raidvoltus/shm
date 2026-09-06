"""
Central batch runtime.

Lifecycle: START → LOAD → PROCESS → SAVE → EXIT
No infinite loops, no daemons. Optimized for GitHub Actions free runners.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any, Optional

from idxbot.calendar.static import StaticIDXCalendar
from idxbot.calendar.base import MarketSession
from idxbot.config import get_settings
from idxbot.core.safety import assert_no_live_trading
from idxbot.logging import get_logger
from idxbot.runtime.clock import MarketClock, ExecutionContext, LatencyDecision
from idxbot.runtime.idempotency import ExecutionIdentity, IdempotencyStore
from idxbot.runtime.retry import RetryPolicy, FatalError, StateCorruption, ConfigurationError
from idxbot.storage import build_storage_backend
from idxbot.storage.backend import (
    LocalStorageBackend,
    StorageBackend,
    StateCorruptionError,
    VersionConflictError,
)
from idxbot.storage.state import PortfolioState, AccountState
from idxbot.runtime.pipeline import AutonomousPipeline

PORTFOLIO_KEY = "portfolio"
logger = get_logger()


class RuntimeResult:
    def __init__(
        self,
        status: str,
        message: str,
        ctx: Optional[ExecutionContext] = None,
        duration_ms: float = 0.0,
    ) -> None:
        self.status = status
        self.message = message
        self.ctx = ctx
        self.duration_ms = duration_ms

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "status": self.status,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "context": self.ctx.to_dict() if self.ctx else None,
        }
        pipe = getattr(self, "pipeline", None)
        if pipe is not None and hasattr(pipe, "to_dict"):
            d["pipeline"] = pipe.to_dict()
        return d


class MarketRuntime:
    """
    Autonomous batch runtime for IDX signal bot.

    Orchestrates full autonomous pipeline: data → quality → governor → signal → OrderIntent.
    Paper only. LIVE_TRADING permanently disabled.
    """

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        calendar: Optional[StaticIDXCalendar] = None,
        clock: Optional[MarketClock] = None,
    ) -> None:
        self.settings = get_settings()
        self.storage = storage or build_storage_backend()
        self.calendar = calendar or StaticIDXCalendar()
        self.clock = clock or MarketClock()
        self.idempotency = IdempotencyStore(self.storage)
        self.retry = RetryPolicy(max_attempts=3, base_delay_sec=0.05)

    def health(self) -> dict[str, Any]:
        """Non-mutating health check."""
        checks: dict[str, Any] = {"status": "Healthy", "checks": {}}
        try:
            assert_no_live_trading()
            checks["checks"]["safety"] = "ok"
            checks["checks"]["live_trading"] = self.settings.live_trading
            checks["checks"]["paper_trading"] = self.settings.paper_trading
            checks["checks"]["initial_balance"] = self.settings.initial_balance

            # storage
            self.storage.exists("__health__")
            checks["checks"]["storage"] = "ok"

            # state load (read-only)
            try:
                st = self.storage.load_state(PORTFOLIO_KEY)
                checks["checks"]["state"] = "present" if st else "absent"
            except StateCorruptionError as e:
                checks["checks"]["state"] = f"corrupt: {e}"
                checks["status"] = "Unhealthy"

            # calendar
            today = self.clock.market_date()
            td = self.calendar.is_trading_day(today)
            checks["checks"]["calendar"] = "ok"
            checks["checks"]["today_trading_day"] = td

            checks["checks"]["python"] = "ok"
            checks["checks"]["config"] = "ok"
        except Exception as exc:  # noqa: BLE001
            checks["status"] = "Unhealthy"
            checks["error"] = str(exc)
        return checks

    def _load_portfolio(self) -> tuple[PortfolioState, int]:
        """Load portfolio; initialize only if truly absent. Never reset on corrupt."""
        try:
            raw = self.storage.load_state(PORTFOLIO_KEY)
        except StateCorruptionError:
            raise StateCorruption(
                "Portfolio state corrupt. FAIL CLOSED — will not reset balance."
            )

        if raw is None:
            ps = PortfolioState.create_initial(self.settings.initial_balance)
            return ps, 0

        version = int(raw.get("state_version", 0))
        # strip internal meta
        clean = {k: v for k, v in raw.items() if not k.startswith("_")}
        if "account" in clean:
            ps = PortfolioState.from_dict(clean)
        else:
            # legacy / partial
            ps = PortfolioState.create_initial(self.settings.initial_balance)
        return ps, version

    def _save_portfolio(self, ps: PortfolioState, expected_version: int) -> int:
        data = ps.to_dict()
        data["state_version"] = expected_version + 1
        self.storage.save_state(
            PORTFOLIO_KEY, data, expected_version=expected_version
        )
        return expected_version + 1

    def run(
        self,
        *,
        run_id: Optional[str] = None,
        scheduled_at: Optional[datetime] = None,
        dry_run: bool = False,
    ) -> RuntimeResult:
        t0 = time.perf_counter()
        run_id = run_id or str(uuid.uuid4())
        scheduled_at = scheduled_at or self.clock.now()

        try:
            assert_no_live_trading()
            if self.settings.live_trading:
                raise ConfigurationError("LIVE_TRADING must be false")
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000
            logger.error("safety_violation", status="FATAL", run_id=run_id, error=str(exc))
            return RuntimeResult("FATAL", str(exc), duration_ms=ms)

        ctx = self.clock.build_context(
            run_id=run_id, scheduled_at=scheduled_at, dry_run=dry_run
        )
        logger.info(
            "run_start",
            run_id=run_id,
            market_date=ctx.market_date.isoformat(),
            status="START",
            dry_run=dry_run,
            latency_decision=ctx.latency_decision.value,
        )

        # Latency gate always applies (window validity). dry_run only affects
        # persistence and market-closed soft-path, not late-arrival skips.
        ok, reason = self.clock.should_process(ctx)
        if not ok:
            ms = (time.perf_counter() - t0) * 1000
            logger.warning(
                "skip_late",
                run_id=run_id,
                market_date=ctx.market_date.isoformat(),
                status="SKIP",
                reason=reason,
                duration_ms=ms,
            )
            return RuntimeResult("SKIP", reason, ctx=ctx, duration_ms=ms)

        # Calendar
        session = self.calendar.get_market_session(ctx.execution_at)
        if self.calendar.last_error:
            logger.warning(
                "calendar_provider_error",
                run_id=run_id,
                status="WARNING",
                error=self.calendar.last_error,
            )
        if session == MarketSession.CLOSED and not dry_run:
            # Still allow dry-run to exercise path
            is_td = self.calendar.is_trading_day(ctx.market_date)
            if not is_td:
                ms = (time.perf_counter() - t0) * 1000
                msg = "MARKET CLOSED (non-trading day or calendar fail-closed)"
                logger.info(
                    "market_closed",
                    run_id=run_id,
                    market_date=ctx.market_date.isoformat(),
                    market_session=session.value,
                    status="SKIP",
                    duration_ms=ms,
                )
                return RuntimeResult("SKIP", msg, ctx=ctx, duration_ms=ms)

        # Idempotency
        identity = ExecutionIdentity(
            run_id=run_id,
            scheduled_at=ctx.scheduled_at,
            market_window=f"{ctx.window_start.isoformat()}/{ctx.window_end.isoformat()}",
            state_version=0,
        )
        if self.idempotency.already_executed(identity):
            ms = (time.perf_counter() - t0) * 1000
            msg = f"DUPLICATE: identity {identity.key()} already completed"
            logger.info(
                "duplicate_skip",
                run_id=run_id,
                status="SKIP",
                duration_ms=ms,
            )
            return RuntimeResult("SKIP", msg, ctx=ctx, duration_ms=ms)

        # Load state
        try:
            portfolio, version = self.retry.run(lambda: self._load_portfolio())
        except (StateCorruption, StateCorruptionError) as exc:
            ms = (time.perf_counter() - t0) * 1000
            logger.error(
                "state_corrupt",
                run_id=run_id,
                status="FATAL",
                error=str(exc),
                duration_ms=ms,
            )
            return RuntimeResult("FATAL", str(exc), ctx=ctx, duration_ms=ms)

        identity = ExecutionIdentity(
            run_id=identity.run_id,
            scheduled_at=identity.scheduled_at,
            market_window=identity.market_window,
            state_version=version,
        )

        # PROCESS — full autonomous pipeline (data → signal → OrderIntent)
        account_dict = portfolio.account.model_dump() if hasattr(portfolio, "account") else {
            "cash": getattr(portfolio, "cash", 10_000_000),
            "equity": getattr(portfolio, "equity", 10_000_000),
            "positions": {},
        }
        if hasattr(portfolio, "account") and hasattr(portfolio.account, "positions"):
            pos = portfolio.account.positions
            if isinstance(pos, dict):
                account_dict["positions"] = {
                    k: (v.model_dump() if hasattr(v, "model_dump") else v)
                    for k, v in pos.items()
                }

        pipeline = AutonomousPipeline()  # fixture only if IDXBOT_USE_FIXTURE
        # restore hysteresis state if present
        prev = (portfolio.metrics or {}).get("previous_intents") or {}
        if isinstance(prev, dict):
            pipeline.load_previous_intents(prev)

        pipe_result = pipeline.run(
            run_id=run_id,
            scheduled_at=ctx.scheduled_at,
            account=account_dict,
            dry_run=dry_run,
        )

        portfolio.metrics = {
            **(portfolio.metrics or {}),
            "last_run_id": run_id,
            "last_scheduled_at": ctx.scheduled_at.isoformat(),
            "last_session": session.value,
            "last_pipeline_status": pipe_result.status,
            "last_governor": pipe_result.governor_state,
            "last_intent_count": len(pipe_result.intents),
            "previous_intents": pipeline.previous_intents_snapshot(),
            "persistence_mode": pipe_result.persistence_mode,
        }

        state_version = version
        if not dry_run:
            try:
                new_version = self.retry.run(
                    lambda: self._save_portfolio(portfolio, version)
                )
                state_version = new_version
            except VersionConflictError as exc:
                ms = (time.perf_counter() - t0) * 1000
                logger.error(
                    "version_conflict",
                    run_id=run_id,
                    status="CONFLICT",
                    error=str(exc),
                    duration_ms=ms,
                )
                return RuntimeResult("CONFLICT", str(exc), ctx=ctx, duration_ms=ms)
            except Exception as exc:
                ms = (time.perf_counter() - t0) * 1000
                logger.error(
                    "save_failed",
                    run_id=run_id,
                    status="FATAL",
                    error=str(exc),
                    duration_ms=ms,
                )
                return RuntimeResult("FATAL", f"save failed: {exc}", ctx=ctx, duration_ms=ms)

            self.idempotency.mark_completed(identity)

        ms = (time.perf_counter() - t0) * 1000
        status = "OK"
        if pipe_result.status in ("FAILED", "SAFE_EXIT"):
            status = pipe_result.status if pipe_result.status != "SAFE_EXIT" else "OK"
        logger.info(
            "run_complete",
            run_id=run_id,
            market_date=ctx.market_date.isoformat(),
            market_session=session.value,
            state_version=state_version,
            duration_ms=ms,
            status=status,
            pipeline_status=pipe_result.status,
            intents=len(pipe_result.intents),
            governor=pipe_result.governor_state,
        )
        # Attach pipeline summary onto result message
        msg = f"{pipe_result.status}: {pipe_result.message}; intents={len(pipe_result.intents)}"
        result = RuntimeResult(status, msg, ctx=ctx, duration_ms=ms)
        result.pipeline = pipe_result  # type: ignore[attr-defined]
        return result
