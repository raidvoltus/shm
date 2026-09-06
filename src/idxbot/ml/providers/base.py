"""
Compute provider abstraction for ML training.

Providers:
  - LocalComputeProvider (always available)
  - ColabComputeProvider (optional, fail-safe)

Signal mode must NEVER depend on Colab.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclass
class ComputeCapabilities:
    cpu: bool = True
    gpu: bool = False
    max_ram_gb: float = 4.0
    max_runtime_minutes: float = 19.0
    name: str = "base"


@dataclass
class TrainingJobSpec:
    job_id: str
    algorithm: str
    feature_version: str
    seed: int = 42
    max_minutes: float = 19.0
    hyperparams: dict[str, Any] = field(default_factory=dict)
    dataset_meta: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    job_id: str
    status: JobStatus
    algorithm: str
    metrics: dict[str, Any] = field(default_factory=dict)
    model_artifact_path: Optional[str] = None
    checksum_sha256: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    training_seconds: float = 0.0


class ComputeProvider(ABC):
    """Minimal interface for local / Colab compute."""

    @abstractmethod
    def availability(self) -> bool:
        ...

    @abstractmethod
    def capabilities(self) -> ComputeCapabilities:
        ...

    @abstractmethod
    def submit_job(self, spec: TrainingJobSpec) -> str:
        """Return job_id."""
        ...

    @abstractmethod
    def get_status(self, job_id: str) -> JobStatus:
        ...

    @abstractmethod
    def collect_result(self, job_id: str) -> TrainingResult:
        ...

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        ...
