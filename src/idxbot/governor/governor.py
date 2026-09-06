"""
Computational Governor — resource-aware model selection only.

Does NOT modify RiskPolicy, paper balance, Experience Store, or Model Registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from idxbot.governor.audit import build_audit_event
from idxbot.governor.config import GovernorConfig
from idxbot.governor.fallback import EnsemblePlan, plan_ensemble
from idxbot.governor.health import ModelHealth, health_from_metadata, scan_registry_metadata
from idxbot.governor.resources import ResourceSnapshot
from idxbot.governor.selector import SelectionResult, select_models
from idxbot.ml.candidates import list_candidates


@dataclass(frozen=True)
class GovernorDecision:
    selection: SelectionResult
    ensemble: EnsemblePlan
    snapshot: ResourceSnapshot
    audit: dict[str, Any]
    health: tuple[ModelHealth, ...]

    @property
    def is_safe_exit(self) -> bool:
        return self.selection.decision == "SAFE_EXIT"


class ComputationalGovernor:
    """
    Orchestrates: snapshot → metadata health → select ladder → ensemble plan → audit.

    NEVER loads model.joblib here. NEVER touches RiskPolicy.
    """

    def __init__(self, config: Optional[GovernorConfig] = None) -> None:
        self.config = config or GovernorConfig()
        if self.config.n_jobs != 1:
            raise ValueError("n_jobs must remain 1")

    def decide(
        self,
        *,
        health: Optional[Sequence[ModelHealth]] = None,
        registry_root: Optional[str] = None,
        algorithms: Optional[Sequence[str]] = None,
        resource_inject: Optional[dict] = None,
        meta_weights: Optional[Mapping[str, float]] = None,
        expected_feature_version: Optional[str] = None,
        force_retrain: bool = False,
    ) -> GovernorDecision:
        snapshot = ResourceSnapshot.capture(inject=resource_inject)
        if health is None:
            algos = list(algorithms) if algorithms else [c.algorithm for c in list_candidates()]
            if registry_root:
                health = scan_registry_metadata(
                    registry_root,
                    algos,
                    expected_feature_version=expected_feature_version,
                    default_memory=self.config.default_model_memory_bytes,
                )
            else:
                # synthetic healthy metadata for candidates (no registry)
                health = [
                    health_from_metadata(
                        a,
                        {
                            "artifact_sha256": "a" * 32,
                            "estimated_memory_bytes": self.config.default_model_memory_bytes,
                            "artifact_size_bytes": 1_000_000,
                            "feature_version": expected_feature_version or "1",
                            "dataset_version": "5.0.0",
                            "model_version": "v1",
                        },
                        expected_feature_version=expected_feature_version,
                        default_memory=self.config.default_model_memory_bytes,
                    )
                    for a in algos
                ]

        selection = select_models(
            health,
            snapshot,
            self.config,
            force_retrain=force_retrain,
        )
        ensemble = plan_ensemble(
            selection.selected,
            meta_weights=meta_weights,
            require_meta_dim=self.config.max_models,
        )
        retraining = "RETRAIN_DEFERRED" if selection.decision == "RETRAIN_DEFERRED" else "n/a"
        if selection.decision == "SAFE_EXIT":
            retraining = "RETRAIN_DEFERRED"
        audit = build_audit_event(
            snapshot.to_dict(),
            selection.to_dict(),
            ensemble.to_dict(),
            retraining_status=retraining,
        )
        return GovernorDecision(
            selection=selection,
            ensemble=ensemble,
            snapshot=snapshot,
            audit=audit,
            health=tuple(health),
        )
