"""Model diversity metrics — correlation / disagreement (no accuracy-only view)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from idxbot.ensemble.contracts import ModelPrediction
from idxbot.ml.candidates import CLASS_ORDER, CLASS_TO_INT


class DiversityMetrics:
    @staticmethod
    def disagreement_rate(preds_a: Sequence[str], preds_b: Sequence[str]) -> float:
        if not preds_a or len(preds_a) != len(preds_b):
            return float("nan")
        return sum(1 for a, b in zip(preds_a, preds_b) if a != b) / len(preds_a)

    @staticmethod
    def prediction_correlation(
        proba_a: Sequence[float], proba_b: Sequence[float]
    ) -> float:
        a = np.asarray(proba_a, dtype=np.float64)
        b = np.asarray(proba_b, dtype=np.float64)
        if len(a) < 2:
            return float("nan")
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    @staticmethod
    def pairwise_disagreement(
        model_preds: dict[str, list[str]],
    ) -> dict[str, float]:
        """model_id_a|model_id_b → disagreement rate."""
        ids = sorted(model_preds.keys())
        out: dict[str, float] = {}
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                out[f"{a}|{b}"] = DiversityMetrics.disagreement_rate(
                    model_preds[a], model_preds[b]
                )
        return out

    @staticmethod
    def error_correlation(
        preds_a: Sequence[str], preds_b: Sequence[str], y_true: Sequence[str]
    ) -> float:
        ea = np.array([1.0 if p != y else 0.0 for p, y in zip(preds_a, y_true)])
        eb = np.array([1.0 if p != y else 0.0 for p, y in zip(preds_b, y_true)])
        if ea.std() < 1e-12 or eb.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(ea, eb)[0, 1])
