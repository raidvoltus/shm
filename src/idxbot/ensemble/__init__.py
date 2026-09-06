"""Ensemble & Meta-Learner (PHASE 7)."""

from idxbot.ensemble.contracts import ModelPrediction, EnsemblePrediction
from idxbot.ensemble.weighted import WeightedAverageEnsemble
from idxbot.ensemble.diversity import DiversityMetrics
from idxbot.ensemble.oof import OOFGenerator
from idxbot.ensemble.meta import NonNegativeMetaLearner
from idxbot.ensemble.calibrate import EnsembleCalibrator
from idxbot.ensemble.registry import EnsembleRegistry
from idxbot.ensemble.engine import EnsembleEngine

__all__ = [
    "ModelPrediction",
    "EnsemblePrediction",
    "WeightedAverageEnsemble",
    "DiversityMetrics",
    "OOFGenerator",
    "NonNegativeMetaLearner",
    "EnsembleCalibrator",
    "EnsembleRegistry",
    "EnsembleEngine",
]
