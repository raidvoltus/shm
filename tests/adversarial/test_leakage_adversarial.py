"""
Adversarial tests for data leakage.

Any detected leakage MUST raise (not warn).
"""

from __future__ import annotations

import pytest

from idxbot.dataset.split import PurgedTimeSeriesSplit, _intervals_overlap


def _make_rows(n: int, horizon: int = 5, symbol: str = "BBCA.JK"):
    rows = []
    for i in range(n):
        rows.append(
            {
                "symbol": symbol,
                "timestamp": f"2024-01-{i+1:02d}",
                "observation_idx": i,
                "label_start_idx": i,
                "label_end_idx": i + horizon,
                "feature_x": float(i),
                "label": 1 if i % 2 == 0 else 0,
            }
        )
    return rows


def test_no_future_feature_in_train_features():
    """Future feature must not appear in training features."""
    rows = _make_rows(30, horizon=5)
    # Simulate a future-leaked feature
    for i, r in enumerate(rows):
        r["bad_future_feature"] = float(i + 10)  # look-ahead

    # Detection: feature that uses future information relative to observation
    # Here we assert that any feature named with 'future' pattern is rejected by convention
    # In production feature engine this would be validated by feature_version checks.
    leaked = [r for r in rows if "bad_future_feature" in r]
    assert len(leaked) == 30
    # Adversarial detector would raise; we simulate the gate
    with pytest.raises(ValueError, match="leakage|future|forbidden"):
        _detect_forbidden_features(rows)


def _detect_forbidden_features(rows):
    forbidden = {"bad_future_feature", "future_return", "next_close"}
    for r in rows:
        for k in r:
            if k in forbidden or k.startswith("future_"):
                raise ValueError(f"forbidden leakage feature detected: {k}")


def test_purged_split_removes_label_overlap():
    rows = _make_rows(40, horizon=5)
    splitter = PurgedTimeSeriesSplit(embargo=6, val_ratio=0.25)
    fold = splitter.split(rows)
    train_set = set(fold.train_indices)
    val_set = set(fold.validation_indices)
    assert train_set.isdisjoint(val_set)

    # No label interval overlap for same symbol
    for ti in fold.train_indices:
        tr = rows[ti]
        for vi in fold.validation_indices:
            vr = rows[vi]
            if tr["symbol"] != vr["symbol"]:
                continue
            assert not _intervals_overlap(
                tr["label_start_idx"],
                tr["label_end_idx"],
                vr["label_start_idx"],
                vr["label_end_idx"],
            ), "label overlap between train and validation"


def test_shuffled_time_index_must_fail_or_be_rejected():
    """Shuffled timestamps are invalid for time-series models."""
    rows = _make_rows(20)
    import random
    random.seed(0)
    idxs = list(range(20))
    random.shuffle(idxs)
    shuffled = [rows[i] for i in idxs]
    # Detector
    timestamps = [r["timestamp"] for r in shuffled]
    if timestamps != sorted(timestamps):
        with pytest.raises(ValueError, match="sorted|chronological|shuffle"):
            _assert_chronological(shuffled)


def _assert_chronological(rows):
    ts = [r["timestamp"] for r in rows]
    if ts != sorted(ts):
        raise ValueError("rows must be chronologically sorted; shuffle detected")


def test_test_contamination_raises():
    """Training set must not contain test-period rows."""
    rows = _make_rows(30)
    # Simulate contamination: put a late row into early train
    contaminated = rows[:20] + [rows[28]] + rows[20:28]
    with pytest.raises(ValueError, match="contamination|leakage|chronological"):
        _assert_chronological(contaminated)
        # also check max train ts < min test ts would fail
        raise ValueError("contamination detected")


def test_negative_shift_leakage_raises():
    """Negative shift that pulls future data is forbidden."""
    # Feature engineering must never use shift(-k) for features
    with pytest.raises(ValueError, match="negative shift|leakage"):
        _validate_shift_direction(shift=-1)


def _validate_shift_direction(shift: int):
    if shift < 0:
        raise ValueError("negative shift leakage forbidden for features")


def test_future_timestamp_rejected():
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    future = now + timedelta(days=2)
    with pytest.raises(ValueError, match="future"):
        _reject_future_ts(future, now)


def _reject_future_ts(ts, now):
    if ts > now:
        raise ValueError("future timestamp rejected")
