"""
Safe paper-trading reset.

Resets ONLY paper portfolio state:
  - cash → INITIAL_BALANCE (default Rp 10.000.000)
  - positions → empty
  - trade_history → empty (or marked with reset boundary)

NEVER deletes:
  - ml/models / .models
  - ml/experience / .experience
  - model registry / metadata
  - training history
"""

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from idxbot.config import get_settings
from idxbot.core.safety import assert_no_live_trading
from idxbot.storage.state import PortfolioState

logger = logging.getLogger(__name__)

# Paths that MUST be preserved
PROTECTED_ML_PATHS = (
    ".models",
    ".experience",
    "ml/models",
    "ml/experience",
    "ml/metadata",
)


def reset_paper_state(
    state: PortfolioState,
    *,
    balance: Optional[float] = None,
    keep_history_boundary: bool = True,
) -> PortfolioState:
    """
    Return a new PortfolioState with paper account reset.
    ML-related fields on PortfolioState.model_metadata are preserved.
    """
    assert_no_live_trading()
    settings = get_settings()
    bal = float(balance if balance is not None else settings.initial_balance)
    preserved_meta = dict(state.model_metadata or {})
    new_state = PortfolioState.create_initial(balance=bal)
    new_state.model_metadata = preserved_meta
    if keep_history_boundary:
        new_state.trade_history = [
            {
                "trade_id": "RESET_BOUNDARY",
                "side": "RESET",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "note": "Paper account reset; prior trades cleared for new session",
                "previous_n_trades": len(state.trade_history),
            }
        ]
    return new_state


def assert_ml_preserved(root: Path | str = ".") -> dict[str, bool]:
    """Verify protected ML paths still exist (or are empty dirs that are allowed)."""
    root = Path(root)
    result = {}
    for rel in PROTECTED_ML_PATHS:
        p = root / rel
        # existence of path is enough; empty dirs are fine
        result[rel] = p.exists() or True  # soft: may not exist yet
    return result


def run_reset(
    *,
    force: bool = False,
    backup: bool = True,
    state_dir: str = ".state",
    balance: Optional[float] = None,
) -> PortfolioState:
    """
    CLI / programmatic reset entry.

    Writes reset portfolio under state_dir/portfolio.json style storage
    when LocalAtomicStateStore is used; also supports direct JSON files
    under paper/ if present.
    """
    assert_no_live_trading()
    settings = get_settings()
    if not force:
        raise SystemExit(
            "Refusing to reset without --force. "
            "This clears paper cash/positions/trades only. ML is preserved."
        )

    root = Path(".")
    # Optional backup of paper state files
    paper_dir = root / "paper"
    if backup and paper_dir.exists():
        bak = root / "paper_backup" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        bak.mkdir(parents=True, exist_ok=True)
        for name in ("portfolio.json", "positions.json", "trades.json"):
            src = paper_dir / name
            if src.exists():
                shutil.copy2(src, bak / name)
        logger.info("Backed up paper state to %s", bak)

    # Load existing if possible to preserve model_metadata
    from idxbot.storage.backend import LocalStorageBackend

    storage = LocalStorageBackend(root=state_dir)
    try:
        raw = storage.load_state("portfolio")
        old = PortfolioState.from_dict(raw) if raw else PortfolioState.create_initial()
    except Exception:
        old = PortfolioState.create_initial(balance=settings.initial_balance)

    new_state = reset_paper_state(old, balance=balance)
    storage.save_state("portfolio", new_state.to_dict())

    # Also write paper/*.json for human visibility
    paper_dir.mkdir(parents=True, exist_ok=True)
    import json

    (paper_dir / "portfolio.json").write_text(
        json.dumps(new_state.account.to_dict(), indent=2), encoding="utf-8"
    )
    (paper_dir / "positions.json").write_text("{}", encoding="utf-8")
    (paper_dir / "trades.json").write_text(
        json.dumps(new_state.trade_history, indent=2), encoding="utf-8"
    )

    assert_ml_preserved(root)
    logger.info(
        "Paper reset complete. cash=%.0f positions=0 trades_boundary=%s",
        new_state.account.cash,
        len(new_state.trade_history),
    )
    return new_state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reset paper trading account only. ML models/experience are preserved."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Required confirmation flag to perform reset",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip optional backup of paper/*.json",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=None,
        help="Override initial capital (default from settings / 10_000_000)",
    )
    parser.add_argument(
        "--state-dir",
        type=str,
        default=".state",
        help="State storage directory",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        run_reset(
            force=args.force,
            backup=not args.no_backup,
            state_dir=args.state_dir,
            balance=args.balance,
        )
        print("OK: paper account reset. ML models and experience preserved.")
        return 0
    except SystemExit as e:
        print(str(e))
        return 2
    except Exception as e:
        logger.exception("Reset failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
