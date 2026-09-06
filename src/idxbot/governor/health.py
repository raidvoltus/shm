"""Lazy metadata-only model health check — NO full artifact load."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class ModelHealth:
    algorithm: str
    healthy: bool
    reason: str
    estimated_memory_bytes: int
    artifact_size_bytes: int
    feature_version: str
    dataset_version: str
    artifact_sha256: str
    version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "healthy": self.healthy,
            "reason": self.reason,
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "artifact_size_bytes": self.artifact_size_bytes,
            "feature_version": self.feature_version,
            "dataset_version": self.dataset_version,
            "artifact_sha256": self.artifact_sha256,
            "version": self.version,
        }


def health_from_metadata(
    algorithm: str,
    meta: dict[str, Any],
    *,
    expected_feature_version: Optional[str] = None,
    default_memory: int = 64 * 1024 * 1024,
) -> ModelHealth:
    """Inspect registry metadata dict only — never joblib.load here."""
    if not meta:
        return ModelHealth(
            algorithm=algorithm,
            healthy=False,
            reason="missing_metadata",
            estimated_memory_bytes=default_memory,
            artifact_size_bytes=0,
            feature_version="",
            dataset_version="",
            artifact_sha256="",
        )
    sha = str(meta.get("artifact_sha256", ""))
    if not sha or len(sha) < 16:
        return ModelHealth(
            algorithm=algorithm,
            healthy=False,
            reason="missing_sha256",
            estimated_memory_bytes=int(meta.get("estimated_memory_bytes") or default_memory),
            artifact_size_bytes=int(meta.get("artifact_size_bytes") or 0),
            feature_version=str(meta.get("feature_version", "")),
            dataset_version=str(meta.get("dataset_version", "")),
            artifact_sha256=sha,
            version=str(meta.get("model_version", "")),
        )
    fv = str(meta.get("feature_version", "1"))
    if expected_feature_version and fv != expected_feature_version:
        return ModelHealth(
            algorithm=algorithm,
            healthy=False,
            reason=f"feature_version_mismatch:{fv}!={expected_feature_version}",
            estimated_memory_bytes=int(meta.get("estimated_memory_bytes") or default_memory),
            artifact_size_bytes=int(meta.get("artifact_size_bytes") or 0),
            feature_version=fv,
            dataset_version=str(meta.get("dataset_version", "")),
            artifact_sha256=sha,
            version=str(meta.get("model_version", "")),
        )
    est = int(meta.get("estimated_memory_bytes") or meta.get("artifact_size_bytes") or default_memory)
    # rough: estimated memory at least artifact size * 2 if only size known
    if meta.get("estimated_memory_bytes") is None and meta.get("artifact_size_bytes"):
        est = max(est, int(meta["artifact_size_bytes"]) * 2)
    return ModelHealth(
        algorithm=algorithm,
        healthy=True,
        reason="ok",
        estimated_memory_bytes=est,
        artifact_size_bytes=int(meta.get("artifact_size_bytes") or 0),
        feature_version=fv,
        dataset_version=str(meta.get("dataset_version", "")),
        artifact_sha256=sha,
        version=str(meta.get("model_version", "")),
    )


def scan_registry_metadata(
    registry_root: str | Path,
    algorithms: Sequence[str],
    *,
    expected_feature_version: Optional[str] = None,
    default_memory: int = 64 * 1024 * 1024,
) -> list[ModelHealth]:
    """
    Read meta.json for latest version of each algorithm — no model.joblib load.
    """
    root = Path(registry_root)
    results: list[ModelHealth] = []
    for algo in algorithms:
        base = root / f"algorithm={algo}"
        if not base.exists():
            results.append(
                health_from_metadata(algo, {}, default_memory=default_memory)
            )
            continue
        versions = sorted(base.glob("version=*"), reverse=True)
        if not versions:
            results.append(health_from_metadata(algo, {}, default_memory=default_memory))
            continue
        meta_path = versions[0] / "meta.json"
        if not meta_path.exists():
            results.append(
                ModelHealth(
                    algorithm=algo,
                    healthy=False,
                    reason="missing_meta_json",
                    estimated_memory_bytes=default_memory,
                    artifact_size_bytes=0,
                    feature_version="",
                    dataset_version="",
                    artifact_sha256="",
                )
            )
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            results.append(
                ModelHealth(
                    algorithm=algo,
                    healthy=False,
                    reason="corrupt_meta_json",
                    estimated_memory_bytes=default_memory,
                    artifact_size_bytes=0,
                    feature_version="",
                    dataset_version="",
                    artifact_sha256="",
                )
            )
            continue
        results.append(
            health_from_metadata(
                algo,
                meta,
                expected_feature_version=expected_feature_version,
                default_memory=default_memory,
            )
        )
    return results
