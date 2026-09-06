"""Dataset construction, labeling, and purged time-series splits (PHASE 5)."""

from idxbot.dataset.builder import DatasetBuilder, DatasetManifest
from idxbot.dataset.labels import LabelGenerator, LabelClass, LabelResult
from idxbot.dataset.split import (
    PurgedTimeSeriesSplit,
    WalkForwardSplitter,
    SplitFold,
    get_train_class_distribution,
)

__all__ = [
    "DatasetBuilder",
    "DatasetManifest",
    "LabelGenerator",
    "LabelClass",
    "LabelResult",
    "PurgedTimeSeriesSplit",
    "WalkForwardSplitter",
    "SplitFold",
    "get_train_class_distribution",
]
