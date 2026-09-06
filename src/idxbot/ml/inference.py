"""
Champion model inference for daily autonomous runs.

Policy:
  - Load champion from ModelRegistry only.
  - Feature version must match.
  - No silent momentum fallback as production ML.
  - Missing/corrupt champion → MODEL_UNAVAILABLE (caller issues HOLD).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from idxbot.ml.candidates import DEFAULT_FEATURE_COLS, CLASS_ORDER
from idxbot.ml.registry import ModelRegistry, ModelRegistryError

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    status: str  # OK | MODEL_UNAVAILABLE | FEATURE_MISMATCH | INFERENCE_FAILED
    probability_up: float = 0.5
    probabilities: dict[str, float] = field(default_factory=dict)
    model_id: str = ""
    feature_version: str = ""
    model_version: str = ""
    algorithm: str = ""
    error: str = ""
    active_models: int = 0


class ChampionInferencer:
    """
    Daily inference path. Does not train. Does not fit ensemble weights.
    """

    def __init__(
        self,
        registry_root: str | Path,
        *,
        expected_feature_version: str = "features-v1",
        feature_columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.registry = ModelRegistry(registry_root)
        self.expected_feature_version = expected_feature_version
        self.feature_columns = list(feature_columns or DEFAULT_FEATURE_COLS)

    def predict_row(self, feature_row: Mapping[str, Any]) -> InferenceResult:
        champ = self.registry.get_champion()
        if not champ:
            return InferenceResult(
                status="MODEL_UNAVAILABLE",
                error="no champion registered",
            )
        algorithm = champ["algorithm"]
        version = champ["version"]
        try:
            payload, meta = self.registry.load(
                algorithm,
                version,
                expected_feature_version=self.expected_feature_version,
            )
        except ModelRegistryError as e:
            msg = str(e)
            status = "FEATURE_MISMATCH" if "Feature version" in msg else "INFERENCE_FAILED"
            return InferenceResult(status=status, error=msg)

        model = payload.get("model") if isinstance(payload, dict) else payload
        if model is None:
            return InferenceResult(status="INFERENCE_FAILED", error="empty model payload")

        try:
            x = self._vectorize(feature_row)
            proba = self._predict_proba(model, x)
        except Exception as e:
            logger.warning("inference_failed", extra={"error": type(e).__name__})
            return InferenceResult(
                status="INFERENCE_FAILED",
                error=f"{type(e).__name__}: {e}",
                model_id=f"{algorithm}:{version}",
            )

        # Map to UP probability (binary or multiclass)
        p_up = float(proba.get("UP", proba.get("1", 0.5)))
        return InferenceResult(
            status="OK",
            probability_up=p_up,
            probabilities=proba,
            model_id=f"{algorithm}:{version}",
            feature_version=meta.feature_version,
            model_version=meta.model_version,
            algorithm=algorithm,
            active_models=1,
        )

    def _vectorize(self, row: Mapping[str, Any]) -> np.ndarray:
        vals = []
        for col in self.feature_columns:
            v = row.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vals.append(0.0)
            else:
                vals.append(float(v))
        return np.asarray([vals], dtype=np.float64)

    def _predict_proba(self, model: Any, x: np.ndarray) -> dict[str, float]:
        if hasattr(model, "predict_proba"):
            raw = model.predict_proba(x)[0]
            classes = list(getattr(model, "classes_", range(len(raw))))
            out: dict[str, float] = {}
            for c, p in zip(classes, raw):
                # normalize class labels
                if isinstance(c, (int, np.integer)):
                    name = CLASS_ORDER[int(c)] if int(c) < len(CLASS_ORDER) else str(c)
                else:
                    name = str(c)
                out[name] = float(p)
            # binary 0/1 → DOWN/UP
            if set(out.keys()) <= {"0", "1"} or set(out.keys()) <= {0, 1}:
                out = {
                    "DOWN": float(out.get("0", out.get(0, 0.0))),
                    "UP": float(out.get("1", out.get(1, 0.0))),
                }
            return out
        if hasattr(model, "decision_function"):
            score = float(model.decision_function(x)[0])
            # logistic squash
            p = 1.0 / (1.0 + np.exp(-score))
            return {"DOWN": 1.0 - p, "UP": p}
        pred = model.predict(x)[0]
        return {"UP": 1.0 if pred in (1, "UP", "1") else 0.0, "DOWN": 0.0 if pred in (1, "UP", "1") else 1.0}
