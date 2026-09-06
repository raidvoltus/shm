"""Weighted probability averaging baseline — non-negative weights summing to 1."""

from __future__ import annotations

from typing import Optional, Sequence

from idxbot.ensemble.contracts import EnsemblePrediction, ModelPrediction
from idxbot.ml.candidates import CLASS_ORDER


class WeightedAverageEnsemble:
    def __init__(self, weights: Optional[dict[str, float]] = None) -> None:
        self.weights = weights  # model_id → weight; None = equal

    def _resolve_weights(self, preds: Sequence[ModelPrediction]) -> dict[str, float]:
        ids = [p.model_id for p in preds]
        if not ids:
            raise ValueError("no predictions")
        if self.weights:
            w = {i: max(0.0, float(self.weights.get(i, 0.0))) for i in ids}
            s = sum(w.values())
            if s <= 0:
                # equal fallback
                return {i: 1.0 / len(ids) for i in ids}
            return {i: v / s for i, v in w.items()}
        return {i: 1.0 / len(ids) for i in ids}

    def aggregate(
        self,
        preds: Sequence[ModelPrediction],
        *,
        ensemble_version: str = "",
    ) -> EnsemblePrediction:
        if not preds:
            raise ValueError("SAFE FAILURE: no valid model predictions")
        for p in preds:
            p.validate()
        # same symbol/timestamp/horizon required
        sym = preds[0].symbol
        ts = preds[0].timestamp
        h = preds[0].horizon
        weights = self._resolve_weights(preds)
        agg = {c: 0.0 for c in CLASS_ORDER}
        up_probs = []
        for p in preds:
            w = weights[p.model_id]
            agg["DOWN"] += w * p.probability_down
            agg["FLAT"] += w * p.probability_flat
            agg["UP"] += w * p.probability_up
            up_probs.append(p.probability_up)
        total = sum(agg.values()) or 1.0
        for c in CLASS_ORDER:
            agg[c] /= total
        pred_class = max(agg, key=agg.get)  # type: ignore
        confidence = agg[pred_class]
        disagreement = float(max(up_probs) - min(up_probs)) if up_probs else 0.0
        ep = EnsemblePrediction(
            symbol=sym,
            timestamp=ts,
            horizon=h,
            probability_down=agg["DOWN"],
            probability_flat=agg["FLAT"],
            probability_up=agg["UP"],
            predicted_class=pred_class,
            confidence=confidence,
            model_count=len(preds),
            active_models=[p.model_id for p in preds],
            max_model_probability=max(up_probs) if up_probs else 0.0,
            min_model_probability=min(up_probs) if up_probs else 0.0,
            disagreement_score=disagreement,
            ensemble_version=ensemble_version,
            weights=weights,
        )
        ep.validate()
        return ep
