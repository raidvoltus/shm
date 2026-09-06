"""Independent ensemble calibration on temporal holdout (not meta-train, not test)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from idxbot.ml.candidates import CLASS_ORDER, CLASS_TO_INT


@dataclass
class CalibrationMeta:
    method: str
    calibration_dataset_version: str
    calibration_window: str
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "calibration_dataset_version": self.calibration_dataset_version,
            "calibration_window": self.calibration_window,
            "n_samples": self.n_samples,
        }


class EnsembleCalibrator:
    """
    Platt-style calibration on ensemble scores using holdout only.
    Fits 3 binary logistic models (one-vs-rest style on predicted probs).
    """

    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self.meta: Optional[CalibrationMeta] = None
        self.fitted = False

    def fit(
        self,
        probs: Sequence[dict[str, float]],
        labels: Sequence[str],
        *,
        timestamps: Optional[Sequence[str]] = None,
        dataset_version: str = "5.0.0",
    ) -> CalibrationMeta:
        if len(probs) < 10:
            # too small — identity calibration
            self.fitted = False
            self.meta = CalibrationMeta(
                method="identity",
                calibration_dataset_version=dataset_version,
                calibration_window="",
                n_samples=len(probs),
            )
            return self.meta
        X = np.array([[p[c] for c in CLASS_ORDER] for p in probs], dtype=np.float64)
        for c in CLASS_ORDER:
            y = np.array([1 if lab == c else 0 for lab in labels])
            if y.sum() < 2 or y.sum() > len(y) - 2:
                continue
            lr = LogisticRegression(max_iter=500, random_state=42)
            lr.fit(X, y)
            self.models[c] = lr
        window = ""
        if timestamps:
            window = f"{min(timestamps)}→{max(timestamps)}"
        self.fitted = True
        self.meta = CalibrationMeta(
            method="platt_ovr",
            calibration_dataset_version=dataset_version,
            calibration_window=window,
            n_samples=len(probs),
        )
        return self.meta

    def transform(self, probs: Sequence[dict[str, float]]) -> list[dict[str, float]]:
        if not self.fitted or not self.models:
            return [dict(p) for p in probs]
        X = np.array([[p[c] for c in CLASS_ORDER] for p in probs], dtype=np.float64)
        out = []
        for i in range(len(probs)):
            scores = {}
            for c in CLASS_ORDER:
                if c in self.models:
                    scores[c] = float(self.models[c].predict_proba(X[i : i + 1])[0, 1])
                else:
                    scores[c] = probs[i][c]
            s = sum(scores.values()) or 1.0
            out.append({c: scores[c] / s for c in CLASS_ORDER})
        return out
