"""
Final Test isolation contract.

Any attempt to use Final Test rows for training/fitting raises FinalTestViolation.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Set


class FinalTestViolation(RuntimeError):
    """Raised when training/fit pipeline attempts to access Final Test data."""


class FinalTestGuard:
    """
    Holds the immutable Final Test key set (symbol, timestamp, horizon).

    Call assert_not_in_training() from trainers / fit paths.
    """

    def __init__(self) -> None:
        self._keys: Set[tuple] = set()
        self._locked = False
        self._final_evaluated = False

    def lock(self, rows: Sequence[dict[str, Any]]) -> None:
        self._keys = {
            (str(r.get("symbol")), str(r.get("timestamp")), int(r.get("horizon", 1)))
            for r in rows
        }
        self._locked = True

    def clear(self) -> None:
        self._keys.clear()
        self._locked = False
        self._final_evaluated = False

    @property
    def locked(self) -> bool:
        return self._locked

    def contains(self, row: dict[str, Any]) -> bool:
        key = (str(row.get("symbol")), str(row.get("timestamp")), int(row.get("horizon", 1)))
        return key in self._keys

    def assert_not_in_training(self, rows: Sequence[dict[str, Any]], context: str = "train") -> None:
        if not self._locked:
            return
        for r in rows:
            if self.contains(r):
                raise FinalTestViolation(
                    f"Final Test isolation violation during {context}: "
                    f"{r.get('symbol')} @ {r.get('timestamp')} cannot be used for training/fitting"
                )

    def mark_final_evaluated(self) -> None:
        """Call once after model lock for one-way OOS report only."""
        self._final_evaluated = True

    @property
    def final_evaluated(self) -> bool:
        return self._final_evaluated


# Process-wide default guard (tests may construct their own)
DEFAULT_FINAL_TEST_GUARD = FinalTestGuard()
