"""Model validation & walk-forward evaluation (PHASE 8)."""

from idxbot.validation.final_test import FinalTestGuard, FinalTestViolation
from idxbot.validation.walkforward import (
    RollingWalkForward,
    WalkForwardFold,
    dynamic_embargo,
)
from idxbot.validation.stats import (
    wilcoxon_signed_rank,
    holm_correction,
    StabilityReport,
    compute_stability,
)
from idxbot.validation.evaluator import (
    WalkForwardEvaluator,
    FoldMetrics,
    ChampionDecision,
)
from idxbot.validation.artifacts import ValidationArtifactStore

__all__ = [
    "FinalTestGuard",
    "FinalTestViolation",
    "RollingWalkForward",
    "WalkForwardFold",
    "dynamic_embargo",
    "wilcoxon_signed_rank",
    "holm_correction",
    "StabilityReport",
    "compute_stability",
    "WalkForwardEvaluator",
    "FoldMetrics",
    "ChampionDecision",
    "ValidationArtifactStore",
]
