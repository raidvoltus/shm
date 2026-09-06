"""
Model training — temporal TRAIN → calibrate on VALIDATION → never touch TEST.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone

from idxbot.ml.candidates import (
    CLASS_ORDER,
    INT_TO_CLASS,
    ModelCandidate,
    get_candidate,
)
from idxbot.ml.preprocess import FeatureMatrixBuilder


@dataclass
class TrainResult:
    algorithm: str
    model: Any
    preprocessor: FeatureMatrixBuilder
    calibrated: bool
    training_time_ms: float
    n_train: int
    n_calibration: int
    random_seed: int
    feature_schema: dict[str, Any]
    class_order: tuple[str, ...] = CLASS_ORDER
    resource_heavy: bool = False

    def predict_proba_dict(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, float]]:
        X = self.preprocessor.transform(rows)
        raw = self._proba(X)
        out = []
        for row in raw:
            d = {CLASS_ORDER[i]: float(row[i]) for i in range(len(CLASS_ORDER))}
            self._validate_proba(d)
            out.append(d)
        return out

    def predict(self, rows: Sequence[dict[str, Any]]) -> list[str]:
        probs = self.predict_proba_dict(rows)
        return [max(p, key=p.get) for p in probs]  # type: ignore

    def _proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            P = self.model.predict_proba(X)
            # align columns to CLASS_ORDER
            classes = list(getattr(self.model, "classes_", range(P.shape[1])))
            aligned = np.zeros((P.shape[0], len(CLASS_ORDER)), dtype=np.float64)
            for j, c in enumerate(classes):
                # c may be int label
                name = INT_TO_CLASS.get(int(c), str(c))
                if name in CLASS_ORDER:
                    aligned[:, CLASS_ORDER.index(name)] = P[:, j]
            # if some class missing, renormalize
            s = aligned.sum(axis=1, keepdims=True)
            s[s == 0] = 1.0
            return aligned / s
        # RidgeClassifier etc: decision_function → softmax
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            if scores.ndim == 1:
                # binary — expand
                scores = np.column_stack([-scores, scores])
            # map to 3 classes if needed
            exp = np.exp(scores - scores.max(axis=1, keepdims=True))
            P = exp / exp.sum(axis=1, keepdims=True)
            classes = list(getattr(self.model, "classes_", range(P.shape[1])))
            aligned = np.zeros((P.shape[0], len(CLASS_ORDER)), dtype=np.float64)
            for j, c in enumerate(classes):
                name = INT_TO_CLASS.get(int(c), str(c))
                if name in CLASS_ORDER:
                    aligned[:, CLASS_ORDER.index(name)] = P[:, j]
            s = aligned.sum(axis=1, keepdims=True)
            s[s == 0] = 1.0
            return aligned / s
        raise RuntimeError(f"Model {type(self.model)} has no probability interface")

    @staticmethod
    def _validate_proba(d: dict[str, float]) -> None:
        vals = [d[c] for c in CLASS_ORDER]
        if any(not np.isfinite(v) for v in vals):
            raise ValueError("Non-finite probability")
        if any(v < -1e-9 or v > 1 + 1e-9 for v in vals):
            raise ValueError("Probability out of [0,1]")
        s = sum(vals)
        if abs(s - 1.0) > 0.05:
            raise ValueError(f"Probabilities sum to {s}, not ~1")


class ModelTrainer:
    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def train(
        self,
        algorithm: str,
        train_rows: Sequence[dict[str, Any]],
        calibration_rows: Optional[Sequence[dict[str, Any]]] = None,
        *,
        calibrate: bool = True,
        feature_version: str = "1",
    ) -> TrainResult:
        if len(train_rows) < 10:
            raise ValueError("insufficient training rows")
        cand = get_candidate(algorithm)
        prep = FeatureMatrixBuilder(use_regime=cand.uses_regime, feature_version=feature_version)
        prep.fit(train_rows)
        X_train = prep.transform(train_rows)
        y_train = prep.labels(train_rows)

        est = cand.create(self.random_seed)
        t0 = time.perf_counter()
        est.fit(X_train, y_train)
        calibrated = False
        n_cal = 0

        if calibrate and calibration_rows and len(calibration_rows) >= 10:
            # Fit calibrator on validation holdout ONLY (not train, not test)
            X_cal = prep.transform(calibration_rows)
            y_cal = prep.labels(calibration_rows)
            try:
                # sklearn CalibratedClassifierCV with cv='prefit' uses provided model
                cal = CalibratedClassifierCV(est, method="sigmoid", cv="prefit")
                cal.fit(X_cal, y_cal)
                est = cal
                calibrated = True
                n_cal = len(calibration_rows)
            except Exception:
                # keep uncalibrated model on failure
                calibrated = False

        elapsed = (time.perf_counter() - t0) * 1000
        return TrainResult(
            algorithm=algorithm,
            model=est,
            preprocessor=prep,
            calibrated=calibrated,
            training_time_ms=elapsed,
            n_train=len(train_rows),
            n_calibration=n_cal,
            random_seed=self.random_seed,
            feature_schema=prep.schema(),
            resource_heavy=cand.resource_heavy,
        )
