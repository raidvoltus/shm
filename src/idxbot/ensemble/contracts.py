"""Standard prediction contracts for base models and ensemble."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


PROB_TOL = 0.05  # sum ≈ 1.0


@dataclass
class ModelPrediction:
    model_id: str
    model_version: str
    symbol: str
    timestamp: str
    horizon: int
    probability_down: float
    probability_flat: float
    probability_up: float
    predicted_class: str
    confidence: float
    feature_version: str = "1"
    dataset_version: str = "5.0.0"

    def probs(self) -> dict[str, float]:
        return {
            "DOWN": self.probability_down,
            "FLAT": self.probability_flat,
            "UP": self.probability_up,
        }

    def validate(self) -> None:
        vals = [self.probability_down, self.probability_flat, self.probability_up]
        if any(v < -1e-9 or v > 1 + 1e-9 for v in vals):
            raise ValueError("probability out of [0,1]")
        if abs(sum(vals) - 1.0) > PROB_TOL:
            raise ValueError(f"probabilities sum to {sum(vals)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnsemblePrediction:
    symbol: str
    timestamp: str
    horizon: int
    probability_down: float
    probability_flat: float
    probability_up: float
    predicted_class: str
    confidence: float
    model_count: int
    active_models: list[str]
    max_model_probability: float
    min_model_probability: float
    disagreement_score: float
    ensemble_version: str = ""
    weights: dict[str, float] = field(default_factory=dict)

    def probs(self) -> dict[str, float]:
        return {
            "DOWN": self.probability_down,
            "FLAT": self.probability_flat,
            "UP": self.probability_up,
        }

    def validate(self) -> None:
        vals = [self.probability_down, self.probability_flat, self.probability_up]
        if any(v < -1e-9 or v > 1 + 1e-9 for v in vals):
            raise ValueError("ensemble probability out of [0,1]")
        if abs(sum(vals) - 1.0) > PROB_TOL:
            raise ValueError(f"ensemble probs sum to {sum(vals)}")
        if self.model_count < 1:
            raise ValueError("no active models")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
