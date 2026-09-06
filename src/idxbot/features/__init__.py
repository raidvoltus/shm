"""Feature engineering pipeline (PHASE 4). No labels, no ML training."""

from idxbot.features.engine import (
    FeatureEngine,
    FeatureEngineError,
    FeatureQualityEngine,
    TrainOnlyNormalizer,
)
from idxbot.features.schema import (
    CANONICAL_TIMEFRAME,
    FEATURE_COLUMNS,
    FEATURE_METADATA,
)

__all__ = [
    "FeatureEngine",
    "FeatureEngineError",
    "FeatureQualityEngine",
    "TrainOnlyNormalizer",
    "CANONICAL_TIMEFRAME",
    "FEATURE_COLUMNS",
    "FEATURE_METADATA",
]
