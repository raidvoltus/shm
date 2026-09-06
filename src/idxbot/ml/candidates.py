"""
Seven candidate model abstractions.

All estimators: n_jobs=1, explicit random_seed, no parallel search.
Regime-aware candidate uses Phase 4 causal regime features only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

def _sklearn():
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "RidgeClassifier": RidgeClassifier,
    }

CLASS_ORDER = ("DOWN", "FLAT", "UP")
CLASS_TO_INT = {c: i for i, c in enumerate(CLASS_ORDER)}
INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}

# Feature columns usable for ML (exclude identity / labels / regimes optional)
DEFAULT_FEATURE_COLS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "ema_10",
    "ema_20",
    "ema_50",
    "price_vs_sma20",
    "price_vs_sma50",
    "sma5_vs_sma20",
    "sma20_vs_sma50",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "true_range",
    "atr_14",
    "rolling_volatility_20",
    "rolling_range_20",
    "volume_sma_20",
    "volume_ratio_20",
    "volume_change_1d",
    "daily_range",
    "body_size",
    "upper_wick",
    "lower_wick",
    "close_position_in_range",
    "gap_from_previous_close",
]

REGIME_COLS = ["trend_regime", "volatility_regime", "volume_regime"]


@dataclass
class ModelCandidate:
    algorithm: str
    display_name: str
    factory: Callable[[int], Any]
    resource_heavy: bool = False
    uses_regime: bool = False
    notes: str = ""
    hyperparams: dict[str, Any] = field(default_factory=dict)

    def create(self, random_seed: int = 42) -> Any:
        return self.factory(random_seed)


def _logreg(seed: int):
    LogisticRegression = _sklearn()["LogisticRegression"]
    # n_jobs omitted: linear solvers do not use it (FutureWarning on sklearn>=1.5)
    return LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
    )


def _rf(seed: int):
    RandomForestClassifier = _sklearn()["RandomForestClassifier"]
    return RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=1,
    )


def _et(seed: int):
    ExtraTreesClassifier = _sklearn()["ExtraTreesClassifier"]
    return ExtraTreesClassifier(
        n_estimators=50,
        max_depth=6,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=1,
    )


def _hgb(seed: int):
    HistGradientBoostingClassifier = _sklearn()["HistGradientBoostingClassifier"]
    return HistGradientBoostingClassifier(
        max_iter=50,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=10,
        random_state=seed,
        # no n_jobs in older API; single-threaded by default for small data
    )


def _gb(seed: int):
    GradientBoostingClassifier = _sklearn()["GradientBoostingClassifier"]
    return GradientBoostingClassifier(
        n_estimators=40,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=5,
        random_state=seed,
    )


def _ridge(seed: int):
    RidgeClassifier = _sklearn()["RidgeClassifier"]
    return RidgeClassifier(alpha=1.0, random_state=seed)


def _regime_logreg(seed: int):
    LogisticRegression = _sklearn()["LogisticRegression"]
    return LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
        C=0.5,
    )


CANDIDATES: dict[str, ModelCandidate] = {
    "logistic_regression": ModelCandidate(
        algorithm="logistic_regression",
        display_name="Logistic Regression",
        factory=_logreg,
        notes="Absolute baseline — always available, deterministic",
        hyperparams={"max_iter": 500, "solver": "lbfgs"},
    ),
    "random_forest": ModelCandidate(
        algorithm="random_forest",
        display_name="Random Forest",
        factory=_rf,
        hyperparams={"n_estimators": 50, "max_depth": 6, "n_jobs": 1},
    ),
    "extra_trees": ModelCandidate(
        algorithm="extra_trees",
        display_name="Extra Trees",
        factory=_et,
        hyperparams={"n_estimators": 50, "max_depth": 6, "n_jobs": 1},
    ),
    "hist_gradient_boosting": ModelCandidate(
        algorithm="hist_gradient_boosting",
        display_name="HistGradientBoosting",
        factory=_hgb,
        resource_heavy=True,
        notes="RESOURCE_HEAVY on large universes — Governor may disable",
        hyperparams={"max_iter": 50, "max_depth": 4},
    ),
    "gradient_boosting": ModelCandidate(
        algorithm="gradient_boosting",
        display_name="Gradient Boosting",
        factory=_gb,
        resource_heavy=True,
        notes="RESOURCE_HEAVY — sklearn GB sequential, bounded estimators",
        hyperparams={"n_estimators": 40, "max_depth": 3},
    ),
    "ridge_classifier": ModelCandidate(
        algorithm="ridge_classifier",
        display_name="Ridge Classifier",
        factory=_ridge,
        notes="Regularized linear alternative",
        hyperparams={"alpha": 1.0},
    ),
    "regime_aware_logreg": ModelCandidate(
        algorithm="regime_aware_logreg",
        display_name="Regime-Aware Logistic Regression",
        factory=_regime_logreg,
        uses_regime=True,
        notes="Uses causal trend/volatility/volume regime features from Phase 4",
        hyperparams={"C": 0.5},
    ),
}


def list_candidates() -> list[ModelCandidate]:
    return list(CANDIDATES.values())


def get_candidate(algorithm: str) -> ModelCandidate:
    if algorithm not in CANDIDATES:
        raise KeyError(f"Unknown candidate: {algorithm}. Available: {list(CANDIDATES)}")
    return CANDIDATES[algorithm]
