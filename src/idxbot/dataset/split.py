"""
Purged time-series split + walk-forward with dynamic embargo.

NO random shuffle. NO sklearn train_test_split.
Embargo = max(horizons) + 1 (dynamic).
Label intervals must not overlap across train/val/test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence

from idxbot.dataset.labels import LabelClass


@dataclass(frozen=True)
class SplitFold:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    purge_size: int
    embargo_size: int
    fold_id: int = 0


def _intervals_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Half-open [start, end) overlap."""
    return a_start < b_end and b_start < a_end


class PurgedTimeSeriesSplit:
    """
    Single chronological train/validation split with purge + embargo.

    Rows must be sorted by (timestamp, symbol, horizon).
    Uses label_start_idx / label_end_idx when present for interval purge.
    """

    def __init__(
        self,
        *,
        embargo: int,
        val_ratio: float = 0.2,
    ) -> None:
        if embargo < 1:
            raise ValueError("embargo must be >= 1")
        if not 0 < val_ratio < 1:
            raise ValueError("val_ratio must be in (0,1)")
        self.embargo = embargo
        self.val_ratio = val_ratio

    def split(self, rows: Sequence[dict[str, Any]]) -> SplitFold:
        n = len(rows)
        if n < 10:
            raise ValueError("need at least 10 rows to split")
        # Use unique observation timestamps order
        # Split by observation index within each symbol would be ideal;
        # for simplicity split by row position after sort, then purge by label intervals.
        cut = int(n * (1 - self.val_ratio))
        cut = max(1, min(n - 1, cut))

        val_candidates = list(range(cut, n))
        train_candidates = list(range(0, cut))

        # Build validation set first
        val_set = set(val_candidates)
        # Purge train samples whose label interval overlaps any val observation window
        # Approximate: use observation_idx and label_end_idx relative within series
        train_kept: list[int] = []
        for ti in train_candidates:
            tr = rows[ti]
            t_start = int(tr.get("label_start_idx", tr.get("observation_idx", ti)))
            t_end = int(tr.get("label_end_idx", t_start + 1))
            # Also apply embargo: drop train rows near the cut by embargo bars
            # Using positional embargo as secondary safety
            if cut - ti <= self.embargo and cut - ti > 0:
                # near boundary — skip
                continue
            overlap = False
            for vi in val_set:
                vr = rows[vi]
                v_start = int(vr.get("label_start_idx", vr.get("observation_idx", vi)))
                v_end = int(vr.get("label_end_idx", v_start + 1))
                # Same symbol only for interval purge
                if tr.get("symbol") != vr.get("symbol"):
                    continue
                if _intervals_overlap(t_start, t_end, v_start, v_end):
                    overlap = True
                    break
                # embargo after train label end into val start
                if t_end + self.embargo > v_start and t_end <= v_start:
                    overlap = True
                    break
            if not overlap:
                train_kept.append(ti)

        train_idx = tuple(train_kept)
        val_idx = tuple(sorted(val_set))
        return SplitFold(
            train_indices=train_idx,
            validation_indices=val_idx,
            train_start=str(rows[train_idx[0]]["timestamp"]) if train_idx else "",
            train_end=str(rows[train_idx[-1]]["timestamp"]) if train_idx else "",
            validation_start=str(rows[val_idx[0]]["timestamp"]) if val_idx else "",
            validation_end=str(rows[val_idx[-1]]["timestamp"]) if val_idx else "",
            purge_size=len(train_candidates) - len(train_kept),
            embargo_size=self.embargo,
        )


class WalkForwardSplitter:
    """Expanding-window walk-forward: TRAIN → VAL, TRAIN → VAL, ..."""

    def __init__(
        self,
        *,
        embargo: int,
        n_folds: int = 3,
        val_size: int = 50,
        min_train: int = 50,
    ) -> None:
        self.embargo = embargo
        self.n_folds = n_folds
        self.val_size = val_size
        self.min_train = min_train

    def split(self, rows: Sequence[dict[str, Any]]) -> Iterator[SplitFold]:
        n = len(rows)
        # Place validation windows from the end backwards
        folds = []
        end = n
        for fold_id in range(self.n_folds):
            val_end = end
            val_start = max(self.min_train + self.embargo, val_end - self.val_size)
            if val_start >= val_end:
                break
            train_end = max(0, val_start - self.embargo)
            if train_end < self.min_train:
                break
            train_idx = list(range(0, train_end))
            val_idx = list(range(val_start, val_end))

            # Interval purge
            train_kept = []
            for ti in train_idx:
                tr = rows[ti]
                t_start = int(tr.get("label_start_idx", ti))
                t_end = int(tr.get("label_end_idx", t_start + 1))
                bad = False
                for vi in val_idx:
                    vr = rows[vi]
                    if tr.get("symbol") != vr.get("symbol"):
                        continue
                    v_start = int(vr.get("label_start_idx", vi))
                    v_end = int(vr.get("label_end_idx", v_start + 1))
                    if _intervals_overlap(t_start, t_end, v_start, v_end):
                        bad = True
                        break
                if not bad:
                    train_kept.append(ti)

            fold = SplitFold(
                train_indices=tuple(train_kept),
                validation_indices=tuple(val_idx),
                train_start=str(rows[train_kept[0]]["timestamp"]) if train_kept else "",
                train_end=str(rows[train_kept[-1]]["timestamp"]) if train_kept else "",
                validation_start=str(rows[val_idx[0]]["timestamp"]) if val_idx else "",
                validation_end=str(rows[val_idx[-1]]["timestamp"]) if val_idx else "",
                purge_size=len(train_idx) - len(train_kept),
                embargo_size=self.embargo,
                fold_id=fold_id,
            )
            folds.append(fold)
            end = val_start
        # yield chronological (earliest fold first)
        for f in reversed(folds):
            yield f


def get_train_class_distribution(
    rows: Sequence[dict[str, Any]], train_indices: Sequence[int], *, horizon: Optional[int] = None
) -> dict[str, int]:
    """In-sample only class counts (TRAIN split). Validation must not affect this."""
    counts = {LabelClass.UP.value: 0, LabelClass.FLAT.value: 0, LabelClass.DOWN.value: 0}
    for i in train_indices:
        r = rows[i]
        if horizon is not None and r.get("horizon") != horizon:
            continue
        lab = r.get("label")
        if lab in counts:
            counts[lab] += 1
    return counts


def assert_no_interval_overlap(
    rows: Sequence[dict[str, Any]], a_idx: Sequence[int], b_idx: Sequence[int]
) -> None:
    for ai in a_idx:
        ar = rows[ai]
        for bi in b_idx:
            br = rows[bi]
            if ar.get("symbol") != br.get("symbol"):
                continue
            if ar.get("horizon") != br.get("horizon"):
                continue
            if _intervals_overlap(
                int(ar.get("label_start_idx", 0)),
                int(ar.get("label_end_idx", 0)),
                int(br.get("label_start_idx", 0)),
                int(br.get("label_end_idx", 0)),
            ):
                raise AssertionError(
                    f"label interval overlap {ar['symbol']} h={ar.get('horizon')} "
                    f"{ar.get('label_start_idx')}-{ar.get('label_end_idx')} vs "
                    f"{br.get('label_start_idx')}-{br.get('label_end_idx')}"
                )
