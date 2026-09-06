"""Compute providers for ML training (local + optional Colab)."""

from idxbot.ml.providers.base import (
    ComputeCapabilities,
    ComputeProvider,
    JobStatus,
    TrainingJobSpec,
    TrainingResult,
)
from idxbot.ml.providers.colab import ColabArtifactValidationError, ColabComputeProvider, validate_colab_artifact
from idxbot.ml.providers.local import LocalComputeProvider

__all__ = [
    "ComputeCapabilities",
    "ComputeProvider",
    "JobStatus",
    "TrainingJobSpec",
    "TrainingResult",
    "LocalComputeProvider",
    "ColabComputeProvider",
    "ColabArtifactValidationError",
    "validate_colab_artifact",
]
