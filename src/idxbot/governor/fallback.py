"""
Ensemble mode under degradation.

7 models + compatible meta weights → META
5/3/1 → WEIGHTED_FALLBACK (renormalize available weights)
0 → SAFE_EXIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class EnsemblePlan:
    ensemble_mode: str  # META | WEIGHTED_FALLBACK | SINGLE_MODEL | SAFE_EXIT
    active_models: tuple[str, ...]
    weights: dict[str, float]
    reason: str

    def to_dict(self) -> dict:
        return {
            "ensemble_mode": self.ensemble_mode,
            "active_models": list(self.active_models),
            "weights": self.weights,
            "reason": self.reason,
        }


def plan_ensemble(
    selected: Sequence[str],
    *,
    meta_weights: Optional[Mapping[str, float]] = None,
    require_meta_dim: int = 7,
) -> EnsemblePlan:
    selected = tuple(selected)
    if not selected:
        return EnsemblePlan("SAFE_EXIT", (), {}, "no_active_models")

    if len(selected) == 1:
        m = selected[0]
        return EnsemblePlan("SINGLE_MODEL", selected, {m: 1.0}, "single_model")

    # Meta only if full 7-model set and weights cover them
    if (
        len(selected) == require_meta_dim
        and meta_weights is not None
        and all(m in meta_weights for m in selected)
    ):
        w = {m: float(meta_weights[m]) for m in selected}
        s = sum(w.values())
        if s <= 0:
            w = {m: 1.0 / len(selected) for m in selected}
        else:
            w = {m: v / s for m, v in w.items()}
        if any(v < 0 for v in w.values()):
            # fall through to weighted
            pass
        else:
            return EnsemblePlan("META", selected, w, "full_pool_meta")

    # Weighted fallback — renormalize available weights
    if meta_weights:
        raw = {m: max(0.0, float(meta_weights.get(m, 0.0))) for m in selected}
        s = sum(raw.values())
        if s > 0:
            w = {m: v / s for m, v in raw.items()}
        else:
            w = {m: 1.0 / len(selected) for m in selected}
    else:
        w = {m: 1.0 / len(selected) for m in selected}

    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(0 <= v <= 1 for v in w.values())
    return EnsemblePlan("WEIGHTED_FALLBACK", selected, w, "degraded_renormalized")
