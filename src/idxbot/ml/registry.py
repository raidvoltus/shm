"""
Versioned model registry with SHA256 + dependency handshake.

Corrupt / mismatched artifacts → SAFE FAILURE (no silent load).
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from idxbot.ml.trainer import TrainResult


def _env_meta() -> dict[str, str]:
    meta = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import sklearn
        import pandas
        import numpy
        import pyarrow

        meta["scikit_learn"] = sklearn.__version__
        meta["pandas"] = pandas.__version__
        meta["numpy"] = numpy.__version__
        meta["pyarrow"] = pyarrow.__version__
    except Exception:
        pass
    return meta


@dataclass
class ModelArtifactMeta:
    model_id: str
    model_version: str
    algorithm: str
    dataset_version: str
    feature_version: str
    label_version: str
    schema_version: str
    training_start: str
    training_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    hyperparams: dict[str, Any]
    metrics: dict[str, Any]
    calibration_method: str
    calibration_version: str
    training_time_ms: float
    inference_time_ms: float
    artifact_size_bytes: int
    estimated_memory_bytes: int
    random_seed: int
    created_at: str
    artifact_sha256: str
    feature_schema: dict[str, Any]
    env: dict[str, str] = field(default_factory=_env_meta)
    resource_heavy: bool = False
    registry_schema_version: str = "6.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRegistryError(Exception):
    pass


class ModelRegistry:
    """
    models/
      algorithm=<name>/
        version=<ver>/
          model.joblib
          meta.json
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._champion_path = self.root / "champion.json"

    def _dir(self, algorithm: str, version: str) -> Path:
        return self.root / f"algorithm={algorithm}" / f"version={version}"

    def save(
        self,
        result: "TrainResult",
        *,
        version: str,
        metrics: dict[str, Any],
        dataset_version: str = "5.0.0",
        label_version: str = "1",
        training_start: str = "",
        training_end: str = "",
        validation_start: str = "",
        validation_end: str = "",
        test_start: str = "",
        test_end: str = "",
        inference_time_ms: float = 0.0,
        hyperparams: Optional[dict[str, Any]] = None,
    ) -> ModelArtifactMeta:
        d = self._dir(result.algorithm, version)
        if d.exists() and (d / "model.joblib").exists():
            raise ModelRegistryError(
                f"Immutable version exists: {result.algorithm}/{version} — do not overwrite"
            )
        d.mkdir(parents=True, exist_ok=True)
        artifact_path = d / "model.joblib"
        payload = {
            "model": result.model,
            "preprocessor": result.preprocessor,
            "algorithm": result.algorithm,
            "feature_schema": result.feature_schema,
            "class_order": result.class_order,
            "calibrated": result.calibrated,
            "random_seed": result.random_seed,
        }
        __import__('joblib').dump(payload, artifact_path)
        raw = artifact_path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        size = len(raw)
        meta = ModelArtifactMeta(
            model_id=f"{result.algorithm}:{version}",
            model_version=version,
            algorithm=result.algorithm,
            dataset_version=dataset_version,
            feature_version=result.feature_schema.get("feature_version", "1"),
            label_version=label_version,
            schema_version="6.0",
            training_start=training_start,
            training_end=training_end,
            validation_start=validation_start,
            validation_end=validation_end,
            test_start=test_start,
            test_end=test_end,
            hyperparams=hyperparams or {},
            metrics=metrics,
            calibration_method="sigmoid_prefit" if result.calibrated else "none",
            calibration_version="1" if result.calibrated else "0",
            training_time_ms=result.training_time_ms,
            inference_time_ms=inference_time_ms,
            artifact_size_bytes=size,
            estimated_memory_bytes=size * 3,  # rough
            random_seed=result.random_seed,
            created_at=datetime.now(timezone.utc).isoformat(),
            artifact_sha256=sha,
            feature_schema=result.feature_schema,
            resource_heavy=result.resource_heavy,
        )
        (d / "meta.json").write_text(json.dumps(meta.to_dict(), indent=2, default=str), encoding="utf-8")
        return meta

    def load(
        self,
        algorithm: str,
        version: str,
        *,
        expected_feature_version: Optional[str] = None,
    ) -> tuple[Any, ModelArtifactMeta]:
        d = self._dir(algorithm, version)
        artifact_path = d / "model.joblib"
        meta_path = d / "meta.json"
        if not artifact_path.exists() or not meta_path.exists():
            raise ModelRegistryError(f"Artifact not found: {algorithm}/{version}")
        meta = ModelArtifactMeta(**json.loads(meta_path.read_text(encoding="utf-8")))
        raw = artifact_path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != meta.artifact_sha256:
            raise ModelRegistryError(
                f"SHA256 mismatch for {algorithm}/{version}: expected {meta.artifact_sha256}, got {sha}"
            )
        # dependency handshake
        env = _env_meta()
        for key in ("python_version", "scikit_learn"):
            if key in meta.env and key in env and meta.env[key] != env[key]:
                # soft warn for patch versions — hard fail major mismatch
                if meta.env[key].split(".")[0] != env[key].split(".")[0]:
                    raise ModelRegistryError(
                        f"Dependency major mismatch {key}: artifact={meta.env[key]} runtime={env[key]}"
                    )
        if expected_feature_version and meta.feature_version != expected_feature_version:
            raise ModelRegistryError(
                f"Feature version mismatch: artifact={meta.feature_version} expected={expected_feature_version}"
            )
        # RISK: joblib/pickle can execute arbitrary code — only load registry-internal artifacts after hash check
        payload = __import__('joblib').load(artifact_path)
        return payload, meta

    def set_champion(self, algorithm: str, version: str, reason: str = "") -> None:
        data = {
            "algorithm": algorithm,
            "version": version,
            "model_id": f"{algorithm}:{version}",
            "reason": reason,
            "set_at": datetime.now(timezone.utc).isoformat(),
        }
        self._champion_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_champion(self) -> Optional[dict[str, Any]]:
        if not self._champion_path.exists():
            return None
        return json.loads(self._champion_path.read_text(encoding="utf-8"))

    def list_versions(self, algorithm: str) -> list[str]:
        base = self.root / f"algorithm={algorithm}"
        if not base.exists():
            return []
        return sorted(p.name.replace("version=", "") for p in base.glob("version=*"))
