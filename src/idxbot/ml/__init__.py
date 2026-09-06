"""ML Model Pool — candidates, training, registry, promotion, inference."""

from idxbot.ml.candidates import (
    CLASS_ORDER,
    ModelCandidate,
    list_candidates,
    get_candidate,
)

__all__ = [
    "CLASS_ORDER",
    "ModelCandidate",
    "list_candidates",
    "get_candidate",
]


def __getattr__(name: str):
    if name in ("ModelTrainer", "TrainResult"):
        from idxbot.ml.trainer import ModelTrainer, TrainResult
        return ModelTrainer if name == "ModelTrainer" else TrainResult
    if name in ("ModelEvaluator", "EvalMetrics"):
        from idxbot.ml.evaluator import ModelEvaluator, EvalMetrics
        return ModelEvaluator if name == "ModelEvaluator" else EvalMetrics
    if name in ("ModelRegistry", "ModelArtifactMeta"):
        from idxbot.ml.registry import ModelRegistry, ModelArtifactMeta
        return ModelRegistry if name == "ModelRegistry" else ModelArtifactMeta
    if name in ("ChampionChallenger", "PromotionDecision"):
        from idxbot.ml.promotion import ChampionChallenger, PromotionDecision
        return ChampionChallenger if name == "ChampionChallenger" else PromotionDecision
    raise AttributeError(name)
