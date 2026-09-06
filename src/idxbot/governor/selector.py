"""
Deterministic model subset selection under resource budget.

Ladder: 7 → 5 → 3 → 1 → 0 (SAFE_EXIT)
Never selects 2,4,6,8,...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from idxbot.governor.config import GovernorConfig
from idxbot.governor.health import ModelHealth
from idxbot.governor.resources import ResourceSnapshot


# Deterministic preference order (baseline first, heavy last)
DEFAULT_RANK = (
    "logistic_regression",
    "ridge_classifier",
    "regime_aware_logreg",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "gradient_boosting",
)


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[str, ...]
    disabled: tuple[str, ...]
    degradation_level: str  # FULL|DEGRADED|LOW|MINIMAL|UNSAFE
    target_count: int
    estimated_memory_bytes: int
    reason: str
    decision: str  # OK | SAFE_EXIT | RETRAIN_DEFERRED

    def to_dict(self) -> dict:
        return {
            "selected_models": list(self.selected),
            "disabled_models": list(self.disabled),
            "degradation_level": self.degradation_level,
            "target_count": self.target_count,
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "reason": self.reason,
            "decision": self.decision,
        }


def _level_name(n: int) -> str:
    return {7: "FULL", 5: "DEGRADED", 3: "LOW", 1: "MINIMAL", 0: "UNSAFE"}.get(n, f"N{n}")


def select_models(
    health: Sequence[ModelHealth],
    snapshot: ResourceSnapshot,
    config: GovernorConfig,
    *,
    rank: Sequence[str] = DEFAULT_RANK,
    force_retrain: bool = False,
) -> SelectionResult:
    healthy = [h for h in health if h.healthy]
    unhealthy = [h.algorithm for h in health if not h.healthy]
    # deterministic order
    rank_index = {a: i for i, a in enumerate(rank)}
    healthy.sort(key=lambda h: (rank_index.get(h.algorithm, 999), h.algorithm))

    # Hard safety: available memory
    avail = snapshot.memory_available_bytes
    if avail is not None and avail < config.minimum_safe_memory_bytes:
        return SelectionResult(
            selected=(),
            disabled=tuple(h.algorithm for h in health),
            degradation_level="UNSAFE",
            target_count=0,
            estimated_memory_bytes=0,
            reason="insufficient_memory_below_minimum_safe",
            decision="SAFE_EXIT",
        )

    # Disk
    if (
        snapshot.disk_available_bytes is not None
        and snapshot.disk_available_bytes < config.minimum_disk_free_bytes
    ):
        return SelectionResult(
            selected=(),
            disabled=tuple(h.algorithm for h in health),
            degradation_level="UNSAFE",
            target_count=0,
            estimated_memory_bytes=0,
            reason="disk_pressure",
            decision="SAFE_EXIT",
        )

    # CPU overload → defer retrain but may still run inference with min models
    cpu_pressure = (
        snapshot.cpu_load is not None and snapshot.cpu_load > config.max_cpu_load
    )

    if not healthy:
        return SelectionResult(
            selected=(),
            disabled=tuple(unhealthy),
            degradation_level="UNSAFE",
            target_count=0,
            estimated_memory_bytes=0,
            reason="no_healthy_models",
            decision="SAFE_EXIT",
        )

    # Budget for model loads = available - headroom
    budget = None
    if avail is not None:
        budget = max(0, avail - config.memory_headroom_bytes)

    # Try ladder rungs
    for target in config.degradation_levels:
        if target > len(healthy):
            continue
        candidates = healthy[:target]
        est = sum(h.estimated_memory_bytes for h in candidates)
        if budget is not None and est > budget:
            continue  # try smaller rung
        selected = tuple(h.algorithm for h in candidates)
        disabled = tuple(
            h.algorithm for h in health if h.algorithm not in selected
        )
        decision = "OK"
        reason = f"selected_{target}_within_budget"
        if cpu_pressure and force_retrain:
            decision = "RETRAIN_DEFERRED"
            reason = "excessive_cpu"
        return SelectionResult(
            selected=selected,
            disabled=disabled,
            degradation_level=_level_name(target),
            target_count=target,
            estimated_memory_bytes=est,
            reason=reason,
            decision=decision,
        )

    # Nothing fits → SAFE_EXIT
    return SelectionResult(
        selected=(),
        disabled=tuple(h.algorithm for h in health),
        degradation_level="UNSAFE",
        target_count=0,
        estimated_memory_bytes=0,
        reason="no_rung_fits_memory_budget",
        decision="SAFE_EXIT",
    )
