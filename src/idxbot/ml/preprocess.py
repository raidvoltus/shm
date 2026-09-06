"""
Train-only feature matrix preparation.

Normalizer / imputer fit on TRAIN only — never on validation/test.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

from idxbot.ml.candidates import (
    CLASS_ORDER,
    CLASS_TO_INT,
    DEFAULT_FEATURE_COLS,
    REGIME_COLS,
)


class FeatureMatrixBuilder:
    def __init__(
        self,
        feature_cols: Optional[list[str]] = None,
        *,
        use_regime: bool = False,
        feature_version: str = "1",
    ) -> None:
        cols = list(feature_cols or DEFAULT_FEATURE_COLS)
        if use_regime:
            for c in REGIME_COLS:
                if c not in cols:
                    cols.append(c)
        self.feature_cols = cols
        self.feature_version = feature_version
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.fitted = False

    def _row_values(self, row: dict[str, Any]) -> list[float]:
        vals = []
        for c in self.feature_cols:
            v = row.get(c)
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                vals.append(float("nan"))
            else:
                vals.append(float(v))
        return vals

    def fit(self, rows: Sequence[dict[str, Any]]) -> "FeatureMatrixBuilder":
        cols_data: dict[str, list[float]] = {c: [] for c in self.feature_cols}
        for r in rows:
            for c, v in zip(self.feature_cols, self._row_values(r)):
                if not math.isnan(v):
                    cols_data[c].append(v)
        for c in self.feature_cols:
            vals = cols_data[c]
            if not vals:
                self.means[c] = 0.0
                self.stds[c] = 1.0
            else:
                m = sum(vals) / len(vals)
                var = sum((x - m) ** 2 for x in vals) / len(vals)
                self.means[c] = m
                self.stds[c] = math.sqrt(var) if var > 1e-12 else 1.0
        self.fitted = True
        return self

    def transform(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FeatureMatrixBuilder not fitted — fit on TRAIN only")
        X = np.zeros((len(rows), len(self.feature_cols)), dtype=np.float32)
        for i, r in enumerate(rows):
            for j, c in enumerate(self.feature_cols):
                v = r.get(c)
                if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    # impute with train mean
                    X[i, j] = np.float32(self.means[c])
                else:
                    X[i, j] = np.float32((float(v) - self.means[c]) / self.stds[c])
        return X

    def labels(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        y = []
        for r in rows:
            lab = r.get("label")
            if lab not in CLASS_TO_INT:
                raise ValueError(f"Invalid/missing label: {lab}")
            y.append(CLASS_TO_INT[lab])
        return np.array(y, dtype=np.int32)

    def schema(self) -> dict[str, Any]:
        return {
            "feature_cols": list(self.feature_cols),
            "feature_version": self.feature_version,
            "n_features": len(self.feature_cols),
            "dtype": "float32",
        }
