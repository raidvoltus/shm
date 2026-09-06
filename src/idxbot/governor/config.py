"""GovernorConfig — explicit computational thresholds (no magic numbers)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class GovernorConfig:
    max_models: int = 7
    degradation_levels: tuple[int, ...] = (7, 5, 3, 1)
    # Memory (bytes) — headroom kept free; minimum_safe is floor before SAFE_EXIT
    minimum_safe_memory_bytes: int = 256 * 1024 * 1024  # 256 MiB free required
    memory_headroom_bytes: int = 128 * 1024 * 1024  # extra buffer when estimating loads
    # Per-model default estimate when metadata missing
    default_model_memory_bytes: int = 64 * 1024 * 1024  # 64 MiB
    max_cpu_load: float = 0.95  # fraction of cores
    max_runtime_seconds: float = 600.0
    minimum_disk_free_bytes: int = 256 * 1024 * 1024
    allow_gpu: bool = False
    allow_parallelism: bool = False
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if self.n_jobs != 1:
            raise ValueError("n_jobs must be 1")
        if self.allow_gpu:
            raise ValueError("GPU not allowed")
        if self.allow_parallelism:
            raise ValueError("parallelism not allowed")
        for lvl in self.degradation_levels:
            if lvl not in (0, 1, 3, 5, 7) and lvl != 0:
                # only allow ladder rungs
                if lvl not in self.degradation_levels:
                    pass
        if sorted(self.degradation_levels, reverse=True) != list(self.degradation_levels):
            raise ValueError("degradation_levels must be descending")
