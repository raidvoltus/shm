"""Immutable ensemble registry with checksums."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class EnsembleRegistryError(Exception):
    pass


class EnsembleRegistry:
    """
    ensembles/
      ensemble_version=<ver>/
        ensemble.json
        weights.json
        metadata.json
        checksum.json
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, version: str) -> Path:
        return self.root / f"ensemble_version={version}"

    def save(
        self,
        version: str,
        *,
        base_models: list[str],
        weights: dict[str, float],
        meta_learner_type: str,
        feature_version: str = "1",
        dataset_version: str = "5.0.0",
        label_version: str = "1",
        calibration_version: str = "0",
        training_window: str = "",
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        d = self._dir(version)
        if d.exists() and (d / "ensemble.json").exists():
            raise EnsembleRegistryError(f"Immutable ensemble version exists: {version}")
        d.mkdir(parents=True, exist_ok=True)
        ensemble = {
            "ensemble_version": version,
            "base_models": base_models,
            "meta_learner_type": meta_learner_type,
        }
        weights_doc = {"weights": weights}
        metadata = {
            "ensemble_version": version,
            "base_models": base_models,
            "model_versions": {m: "oof" for m in base_models},
            "weights": weights,
            "meta_learner_type": meta_learner_type,
            "training_dataset_version": dataset_version,
            "feature_version": feature_version,
            "label_version": label_version,
            "calibration_version": calibration_version,
            "training_window": training_window,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": "7.0",
            **(extra or {}),
        }
        (d / "ensemble.json").write_text(json.dumps(ensemble, indent=2), encoding="utf-8")
        (d / "weights.json").write_text(json.dumps(weights_doc, indent=2), encoding="utf-8")
        (d / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        payload = json.dumps(metadata, sort_keys=True, default=str)
        checksum = {
            "artifact_hash": hashlib.sha256(payload.encode()).hexdigest(),
            "algorithm": "sha256",
        }
        (d / "checksum.json").write_text(json.dumps(checksum, indent=2), encoding="utf-8")
        metadata["artifact_hash"] = checksum["artifact_hash"]
        return metadata

    def load(self, version: str) -> dict[str, Any]:
        d = self._dir(version)
        meta_path = d / "metadata.json"
        ck_path = d / "checksum.json"
        if not meta_path.exists():
            raise EnsembleRegistryError(f"Ensemble not found: {version}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if ck_path.exists():
            ck = json.loads(ck_path.read_text(encoding="utf-8"))
            payload = json.dumps(
                {k: v for k, v in meta.items() if k != "artifact_hash"},
                sort_keys=True,
                default=str,
            )
            # recompute without artifact_hash field if present
            meta_copy = {k: v for k, v in meta.items() if k != "artifact_hash"}
            payload = json.dumps(meta_copy, sort_keys=True, default=str)
            h = hashlib.sha256(payload.encode()).hexdigest()
            if h != ck.get("artifact_hash"):
                raise EnsembleRegistryError("Ensemble checksum mismatch")
        return meta
