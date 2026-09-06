"""
EnsembleEngine — wire base predictions → weighted/meta → calibrated output.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from idxbot.ensemble.calibrate import EnsembleCalibrator
from idxbot.ensemble.contracts import EnsemblePrediction, ModelPrediction
from idxbot.ensemble.meta import MetaWeights, NonNegativeMetaLearner
from idxbot.ensemble.weighted import WeightedAverageEnsemble
from idxbot.ml.candidates import CLASS_ORDER
from idxbot.ml.trainer import TrainResult


class EnsembleEngine:
    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        *,
        ensemble_version: str = "ens-v1",
    ) -> None:
        self.weights = weights
        self.ensemble_version = ensemble_version
        self.calibrator: Optional[EnsembleCalibrator] = None
        self.aggregator = WeightedAverageEnsemble(weights)

    def set_meta_weights(self, meta: MetaWeights) -> None:
        self.weights = meta.weights
        self.aggregator = WeightedAverageEnsemble(meta.weights)

    def set_calibrator(self, cal: EnsembleCalibrator) -> None:
        self.calibrator = cal

    def predict_from_model_preds(
        self, preds: Sequence[ModelPrediction]
    ) -> EnsemblePrediction:
        ep = self.aggregator.aggregate(preds, ensemble_version=self.ensemble_version)
        if self.calibrator and self.calibrator.fitted:
            calibrated = self.calibrator.transform([ep.probs()])[0]
            pred_class = max(calibrated, key=calibrated.get)  # type: ignore
            ep = EnsemblePrediction(
                symbol=ep.symbol,
                timestamp=ep.timestamp,
                horizon=ep.horizon,
                probability_down=calibrated["DOWN"],
                probability_flat=calibrated["FLAT"],
                probability_up=calibrated["UP"],
                predicted_class=pred_class,
                confidence=calibrated[pred_class],
                model_count=ep.model_count,
                active_models=ep.active_models,
                max_model_probability=ep.max_model_probability,
                min_model_probability=ep.min_model_probability,
                disagreement_score=ep.disagreement_score,
                ensemble_version=ep.ensemble_version,
                weights=ep.weights,
            )
            ep.validate()
        return ep

    def predict_from_train_results(
        self,
        results: dict[str, TrainResult],
        rows: Sequence[dict[str, Any]],
        *,
        model_version: str = "live",
    ) -> list[EnsemblePrediction]:
        """For each row, collect base probs → ensemble."""
        out: list[EnsemblePrediction] = []
        # precompute all model probs
        model_probs: dict[str, list[dict[str, float]]] = {}
        available = []
        for mid, tr in results.items():
            try:
                model_probs[mid] = tr.predict_proba_dict(rows)
                available.append(mid)
            except Exception:
                continue
        if not available:
            raise ValueError("SAFE FAILURE: no valid models for ensemble")
        for i, r in enumerate(rows):
            preds = []
            for mid in available:
                p = model_probs[mid][i]
                pred_class = max(p, key=p.get)  # type: ignore
                preds.append(
                    ModelPrediction(
                        model_id=mid,
                        model_version=model_version,
                        symbol=str(r["symbol"]),
                        timestamp=str(r["timestamp"]),
                        horizon=int(r.get("horizon", 1)),
                        probability_down=p["DOWN"],
                        probability_flat=p["FLAT"],
                        probability_up=p["UP"],
                        predicted_class=pred_class,
                        confidence=p[pred_class],
                        feature_version=str(r.get("feature_version", "1")),
                        dataset_version=str(r.get("dataset_version", "5.0.0")),
                    )
                )
            out.append(self.predict_from_model_preds(preds))
        return out
