"""
Rolling walk-forward validation with bounded lookback, purge, and dynamic embargo.

NOT expanding without bound — oldest data drops out of the train window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence


def dynamic_embargo(horizons: Sequence[int]) -> int:
    """embargo >= max(horizons) + 1 — never hardcoded to 5."""
    if not horizons:
        return 2
    return max(int(h) for h in horizons) + 1


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    purge_size: int
    embargo_size: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]

    @property
    def train_rows(self) -> int:
        return len(self.train_indices)

    @property
    def validation_rows(self) -> int:
        return len(self.validation_indices)


class RollingWalkForward:
    """
    Rolling (sliding) train window + validation with purge/embargo.

    max_train_lookback: max number of training rows (bounded RAM).
    """

    def __init__(
        self,
        *,
        max_train_lookback: int = 252 * 3,  # ~3 years trading days
        val_size: int = 40,
        step: int = 20,
        embargo: int = 21,
        min_train: int = 50,
    ) -> None:
        if max_train_lookback < min_train:
            raise ValueError("max_train_lookback must be >= min_train")
        self.max_train_lookback = max_train_lookback
        self.val_size = val_size
        self.step = step
        self.embargo = embargo
        self.min_train = min_train

    def split(self, rows: Sequence[dict[str, Any]]) -> Iterator[WalkForwardFold]:
        n = len(rows)
        if n < self.min_train + self.embargo + self.val_size:
            return
        # validation windows advance from earliest possible to end
        fold_id = 0
        # first val starts after min_train + embargo
        val_start = self.min_train + self.embargo
        while val_start + self.val_size <= n:
            val_end = min(n, val_start + self.val_size)
            train_end = val_start - self.embargo
            train_start_idx = max(0, train_end - self.max_train_lookback)
            if train_end - train_start_idx < self.min_train:
                val_start += self.step
                continue
            train_idx = tuple(range(train_start_idx, train_end))
            val_idx = tuple(range(val_start, val_end))
            # enforce no overlap
            if set(train_idx) & set(val_idx):
                raise RuntimeError("fold overlap detected")
            # purge: gap between train_end and val_start is embargo
            purge_size = val_start - train_end
            assert purge_size >= self.embargo

            yield WalkForwardFold(
                fold_id=fold_id,
                train_start=str(rows[train_idx[0]]["timestamp"]),
                train_end=str(rows[train_idx[-1]]["timestamp"]),
                validation_start=str(rows[val_idx[0]]["timestamp"]),
                validation_end=str(rows[val_idx[-1]]["timestamp"]),
                purge_size=purge_size,
                embargo_size=self.embargo,
                train_indices=train_idx,
                validation_indices=val_idx,
            )
            fold_id += 1
            val_start += self.step
