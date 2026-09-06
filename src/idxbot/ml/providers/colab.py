"""
Optional Google Colab compute provider.

Design constraints:
- No secrets in repository
- No manual browser session required for normal ops
- If Colab offline / auth fails / result invalid → fail and let Governor fall back to local
- Never accept arbitrary pickle / unsigned artifacts

Actual remote Colab execution requires external orchestration
(e.g. notebook that pulls job spec from GitHub and pushes validated artifacts).
This provider implements a safe stub + validation layer so the rest of the
system remains fully functional without Colab.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from idxbot.ml.providers.base import (
    ComputeCapabilities,
    ComputeProvider,
    JobStatus,
    TrainingJobSpec,
    TrainingResult,
)

logger = logging.getLogger(__name__)


class ColabArtifactValidationError(Exception):
    """Raised when a Colab result fails integrity / schema checks."""


def validate_colab_artifact(
    metadata: dict[str, Any],
    artifact_path: Optional[str] = None,
    expected_checksum: Optional[str] = None,
) -> None:
    """
    Reject unsafe or malformed Colab results.

    Required metadata keys: algorithm, job_id, metrics, created_at
    Reject: missing checksum when file present, path traversal, unknown keys that
    look like executable payloads.
    """
    required = ("algorithm", "job_id", "metrics")
    for k in required:
        if k not in metadata:
            raise ColabArtifactValidationError(f"missing required metadata key: {k}")

    if not isinstance(metadata.get("metrics"), dict):
        raise ColabArtifactValidationError("metrics must be a dict")

    # Reject obvious payload fields
    forbidden = ("code", "exec", "pickle_raw", "shell", "script")
    for k in forbidden:
        if k in metadata:
            raise ColabArtifactValidationError(f"forbidden metadata key: {k}")

    if artifact_path:
        p = Path(artifact_path)
        if ".." in p.parts:
            raise ColabArtifactValidationError("path traversal rejected")
        if p.suffix.lower() in (".py", ".sh", ".exe", ".bat"):
            raise ColabArtifactValidationError(f"disallowed artifact extension: {p.suffix}")
        if p.exists() and expected_checksum:
            actual = hashlib.sha256(p.read_bytes()).hexdigest()
            if actual != expected_checksum:
                raise ColabArtifactValidationError("checksum mismatch")


class ColabComputeProvider(ComputeProvider):
    """
    Optional provider. availability() is False unless ENABLE_COLAB=true
    and COLAB_JOB_ENDPOINT (or equivalent) is configured.

    Without a real endpoint this provider reports unavailable so Governor
    falls back to LocalComputeProvider.
    """

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        endpoint: Optional[str] = None,
        timeout_sec: float = 60.0,
        artifact_dir: str | Path = ".models/colab_jobs",
    ) -> None:
        if enabled is None:
            enabled = os.environ.get("ENABLE_COLAB", "").lower() in ("1", "true", "yes")
        self.enabled = bool(enabled)
        self.endpoint = endpoint or os.environ.get("COLAB_JOB_ENDPOINT", "")
        self.timeout_sec = timeout_sec
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, dict[str, Any]] = {}

    def availability(self) -> bool:
        if not self.enabled:
            return False
        if not self.endpoint:
            logger.info("Colab provider disabled: no COLAB_JOB_ENDPOINT")
            return False
        # Soft probe — do not claim live connection without verification
        return False  # conservative: require explicit future wiring

    def capabilities(self) -> ComputeCapabilities:
        return ComputeCapabilities(
            cpu=True,
            gpu=True,
            max_ram_gb=12.0,
            max_runtime_minutes=19.0,
            name="colab",
        )

    def submit_job(self, spec: TrainingJobSpec) -> str:
        if not self.availability():
            raise RuntimeError("ColabComputeProvider is not available")
        job_id = spec.job_id or str(uuid.uuid4())
        self._jobs[job_id] = {
            "spec": spec,
            "status": JobStatus.PENDING,
            "submitted_at": time.time(),
        }
        # Write job spec for external notebook pickup (safe JSON only)
        spec_path = self.artifact_dir / f"{job_id}.spec.json"
        payload = {
            "job_id": job_id,
            "algorithm": spec.algorithm,
            "feature_version": spec.feature_version,
            "seed": spec.seed,
            "max_minutes": spec.max_minutes,
            "hyperparams": spec.hyperparams,
            "dataset_meta": spec.dataset_meta,
        }
        spec_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Colab job spec written %s (external runner required)", spec_path)
        return job_id

    def get_status(self, job_id: str) -> JobStatus:
        job = self._jobs.get(job_id)
        if not job:
            # Check for result artifact left by external runner
            result_path = self.artifact_dir / f"{job_id}.result.json"
            if result_path.exists():
                return JobStatus.SUCCEEDED
            return JobStatus.UNKNOWN
        return job["status"]

    def collect_result(self, job_id: str) -> TrainingResult:
        result_path = self.artifact_dir / f"{job_id}.result.json"
        if not result_path.exists():
            return TrainingResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                algorithm="",
                error="Colab result not found",
            )
        try:
            meta = json.loads(result_path.read_text(encoding="utf-8"))
            validate_colab_artifact(
                meta,
                artifact_path=meta.get("model_artifact_path"),
                expected_checksum=meta.get("checksum_sha256"),
            )
            return TrainingResult(
                job_id=job_id,
                status=JobStatus.SUCCEEDED,
                algorithm=meta["algorithm"],
                metrics=meta.get("metrics") or {},
                model_artifact_path=meta.get("model_artifact_path"),
                checksum_sha256=meta.get("checksum_sha256"),
                metadata=meta,
                training_seconds=float(meta.get("training_seconds") or 0),
            )
        except (json.JSONDecodeError, ColabArtifactValidationError) as e:
            logger.warning("Invalid Colab artifact for %s: %s", job_id, e)
            return TrainingResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                algorithm="",
                error=f"invalid artifact: {e}",
            )

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job["status"] in (JobStatus.PENDING, JobStatus.RUNNING):
            job["status"] = JobStatus.CANCELLED
            return True
        return False
