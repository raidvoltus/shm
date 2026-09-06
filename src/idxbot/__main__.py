"""
Entry point: python -m idxbot

Orchestrator only — no indicators, training, Telegram formatting,
or portfolio accounting implemented here.

Modes:
  SIGNAL   — weekday: market data → features → inference → signal → paper → Telegram
  LEARNING — weekend: data → governor → train → backtest → promotion → experience
  auto     — detect from Asia/Jakarta weekday
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import logging

from idxbot.config import get_settings
from idxbot.core.safety import assert_no_live_trading
from idxbot.runtime.runner import MarketRuntime

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("idxbot")
JAKARTA = ZoneInfo("Asia/Jakarta")


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=JAKARTA)
    return dt


def detect_mode(now: datetime | None = None) -> str:
    """Weekday = SIGNAL, weekend = LEARNING (Asia/Jakarta)."""
    now = now or datetime.now(JAKARTA)
    if now.tzinfo is None:
        now = now.replace(tzinfo=JAKARTA)
    else:
        now = now.astimezone(JAKARTA)
    # Monday=0 … Sunday=6
    return "LEARNING" if now.weekday() >= 5 else "SIGNAL"


def _run_learning(args: argparse.Namespace) -> int:
    """
    Weekend learning path — real end-to-end pipeline.
    Hard budget: max_training_minutes (default 19).
    Colab optional; Local mandatory fallback.
    """
    import os
    from idxbot.ml.learning_pipeline import LearningPipeline

    settings = get_settings()
    assert_no_live_trading()
    budget = settings.max_training_minutes
    use_fixture = os.environ.get("IDXBOT_USE_FIXTURE", "true").lower() in ("1", "true", "yes")
    logger.info("RUN_START MODE=LEARNING budget_minutes=%s fixture=%s", budget, use_fixture)

    pipeline = LearningPipeline(
        budget_seconds=budget * 60,
        use_fixture=use_fixture,
        random_seed=settings.random_seed,
    )
    report = pipeline.run(run_id=getattr(args, "run_id", None))
    if args.json:
        import json as _json
        print(_json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(
            f"status={report.status} provider={report.provider} "
            f"trained={len(report.trained)} promoted={bool((report.promotion or {}).get('promoted'))} "
            f"elapsed={report.elapsed_seconds:.1f}s message={report.message}"
        )
    logger.info("RUN_END MODE=LEARNING status=%s", report.status)
    return 0 if report.status in ("OK", "DEGRADED", "BUDGET_EXHAUSTED") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="idxbot",
        description=(
            "IDX Signal Bot — paper trading only. "
            "No live broker execution. Modes: SIGNAL | LEARNING | auto"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("SIGNAL", "LEARNING", "auto"),
        default="auto",
        help="Pipeline mode (default: auto-detect from Asia/Jakarta weekday)",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Run health check and exit (no trading, no state mutation)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute pipeline without persisting portfolio changes",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit run identifier (for idempotency)",
    )
    parser.add_argument(
        "--scheduled-at",
        type=str,
        default=None,
        help="ISO scheduled time (e.g. 2024-06-15T09:00:00+07:00)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON result",
    )
    parser.add_argument(
        "--reset-paper",
        action="store_true",
        help="Reset paper account only (requires --force). Preserves ML.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Confirm destructive paper reset",
    )

    args = parser.parse_args(argv)
    settings = get_settings()
    assert_no_live_trading()
    settings.assert_safe()

    if args.reset_paper:
        from idxbot.trading.reset import run_reset

        try:
            run_reset(force=args.force)
            return 0
        except SystemExit as e:
            print(str(e))
            return 2

    runtime = MarketRuntime()

    if args.health:
        result = runtime.health()
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"Health: {result['status']}")
            for k, v in result.get("checks", {}).items():
                print(f"  {k}: {v}")
            if "error" in result:
                print(f"  error: {result['error']}")
        return 0 if result.get("status") == "Healthy" else 1

    mode = args.mode
    if mode == "auto":
        mode = detect_mode(
            _parse_dt(args.scheduled_at) if args.scheduled_at else None
        )
    logger.info("RUN START MODE=%s PAPER_TRADING_ONLY=%s", mode, settings.paper_trading_only)

    if mode == "LEARNING":
        return _run_learning(args)

    # SIGNAL mode — weekday pipeline via existing MarketRuntime
    scheduled_at = _parse_dt(args.scheduled_at) if args.scheduled_at else None
    result = runtime.run(
        run_id=args.run_id,
        scheduled_at=scheduled_at,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(
            f"status={result.status} message={result.message} "
            f"duration_ms={result.duration_ms:.1f}"
        )

    logger.info("RUN END MODE=SIGNAL status=%s", result.status)
    if result.status in ("FATAL", "CONFLICT"):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
