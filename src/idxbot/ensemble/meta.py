"""
Non-negative meta-learner with optional time decay.

Weights: w_i >= 0, sum(w_i) = 1.
Uses constrained least-squares / NNLS-style optimization on OOF probs.
NO unconstrained logistic regression as meta-learner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence

import numpy as np
from scipy.optimize import nnls


@dataclass
class MetaWeights:
    weights: dict[str, float]
    method: str
    half_life_days: Optional[float] = None
    training_window: str = ""
    dataset_version: str = "5.0.0"
    feature_version: str = "1"
    model_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "method": self.method,
            "half_life_days": self.half_life_days,
            "training_window": self.training_window,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "model_ids": self.model_ids,
        }


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


class NonNegativeMetaLearner:
    def __init__(self, half_life_days: Optional[float] = 60.0) -> None:
        self.half_life_days = half_life_days
        self.weights_: Optional[MetaWeights] = None

    def _time_weights(
        self, timestamps: Sequence[str], cutoff: datetime
    ) -> np.ndarray:
        """Causal decay: older relative to cutoff → smaller weight. No future info."""
        n = len(timestamps)
        if not self.half_life_days or self.half_life_days <= 0:
            return np.ones(n, dtype=np.float64)
        w = np.ones(n, dtype=np.float64)
        hl = self.half_life_days
        for i, ts in enumerate(timestamps):
            age_days = max(0.0, (cutoff - _parse_ts(ts)).total_seconds() / 86400.0)
            w[i] = 0.5 ** (age_days / hl)
        return w

    def fit(
        self,
        oof_rows: Sequence[dict[str, Any]],
        *,
        target_class: str = "UP",
        dataset_version: str = "5.0.0",
        feature_version: str = "1",
    ) -> MetaWeights:
        """
        Fit non-negative weights so weighted P(target) ≈ one-hot actual.

        oof_rows: list of {model_id, probability_*, actual_label, timestamp}
        Grouped by (timestamp, symbol, horizon) with multiple model rows.
        """
        # Pivot: key → {model_id: p_target, actual}
        groups: dict[tuple, dict[str, Any]] = {}
        model_ids: set[str] = set()
        for r in oof_rows:
            key = (r["timestamp"], r["symbol"], r.get("horizon", 1))
            g = groups.setdefault(key, {"models": {}, "label": r["actual_label"], "ts": r["timestamp"]})
            mid = r["model_id"]
            model_ids.add(mid)
            col = f"probability_{target_class.lower()}"
            # also accept probability_up style
            p = r.get(col)
            if p is None:
                p = r.get(f"probability_{target_class}")
            if p is None and target_class == "UP":
                p = r.get("probability_up", 0.0)
            g["models"][mid] = float(p or 0.0)

        ids = sorted(model_ids)
        if not ids or not groups:
            # equal weights fallback
            w = {i: 1.0 / max(len(ids), 1) for i in ids} if ids else {}
            self.weights_ = MetaWeights(
                weights=w, method="equal_fallback", model_ids=ids
            )
            return self.weights_

        X_list = []
        y_list = []
        ts_list = []
        for key, g in groups.items():
            X_list.append([g["models"].get(i, 0.0) for i in ids])
            y_list.append(1.0 if g["label"] == target_class else 0.0)
            ts_list.append(g["ts"])

        X = np.asarray(X_list, dtype=np.float64)
        y = np.asarray(y_list, dtype=np.float64)
        cutoff = max(_parse_ts(t) for t in ts_list)
        sw = self._time_weights(ts_list, cutoff)
        # weight rows
        Xw = X * sw[:, None]
        yw = y * sw
        # NNLS: min ||Xw @ w - yw||, w >= 0
        w_raw, _ = nnls(Xw, yw)
        if np.any(w_raw < -1e-10):
            raise ValueError("FAIL: NNLS produced negative weights")
        s = w_raw.sum()
        if s <= 0:
            w_raw = np.ones(len(ids)) / len(ids)
        else:
            w_raw = w_raw / s
        weights = {ids[i]: float(w_raw[i]) for i in range(len(ids))}
        # verify constraints
        if any(v < -1e-9 for v in weights.values()):
            raise ValueError("FAIL: negative meta weights")
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError("FAIL: weights do not sum to 1")

        self.weights_ = MetaWeights(
            weights=weights,
            method="nnls_time_decay" if self.half_life_days else "nnls",
            half_life_days=self.half_life_days,
            training_window=f"{min(ts_list)}→{max(ts_list)}",
            dataset_version=dataset_version,
            feature_version=feature_version,
            model_ids=ids,
        )
        return self.weights_
