"""
Local compute provider — default, always preferred for signal-safe paths.
Respects hard training budget (default 19 minutes).
"""

from __future__ import annotations

import hashlib
import logging
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


class LocalComputeProvider(ComputeProvider):
    def __init__(
        self,
        *,
        artifact_dir: str | Path = ".models/local_jobs",
        max_minutes: float = 19.0,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.max_minutes = max_minutes
        self._jobs: dict[str, dict[str, Any]] = {}

    def availability(self) -> bool:
        return True

    def capabilities(self) -> ComputeCapabilities:
        return ComputeCapabilities(
            cpu=True,
            gpu=False,
            max_ram_gb=4.0,
            max_runtime_minutes=self.max_minutes,
            name="local",
        )

    def submit_job(self, spec: TrainingJobSpec) -> str:
        job_id = spec.job_id or str(uuid.uuid4())
        self._jobs[job_id] = {
            "spec": spec,
            "status": JobStatus.PENDING,
            "submitted_at": time.time(),
            "result": None,
        }
        logger.info("LOCAL job submitted id=%s algo=%s", job_id, spec.algorithm)
        return job_id

    def get_status(self, job_id: str) -> JobStatus:
        job = self._jobs.get(job_id)
        if not job:
            return JobStatus.UNKNOWN
        return job["status"]

    def run_inline(
        self,
        spec: TrainingJobSpec,
        train_fn,
    ) -> TrainingResult:
        """
        Execute train_fn(spec) under time budget.
        train_fn must return dict with metrics, optional model_bytes / path.
        """
        job_id = self.submit_job(spec)
        self._jobs[job_id]["status"] = JobStatus.RUNNING
        budget_sec = min(spec.max_minutes, self.max_minutes) * 60.0
        t0 = time.monotonic()
        try:
            out = train_fn(spec)
            elapsed = time.monotonic() - t0
            if elapsed > budget_sec:
                logger.warning(
                    "LOCAL job %s exceeded budget (%.1fs > %.1fs) — result kept if valid",
                    job_id,
                    elapsed,
                    budget_sec,
                )
            checksum = None
            path = out.get("model_artifact_path")
            if path and Path(path).exists():
                checksum = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            result = TrainingResult(
                job_id=job_id,
                status=JobStatus.SUCCEEDED,
                algorithm=spec.algorithm,
                metrics=out.get("metrics") or {},
                model_artifact_path=path,
                checksum_sha256=checksum,
                metadata=out.get("metadata") or {},
                training_seconds=elapsed,
            )
            self._jobs[job_id]["status"] = JobStatus.SUCCEEDED
            self._jobs[job_id]["result"] = result
            return result
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.exception("LOCAL job %s failed: %s", job_id, e)
            result = TrainingResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                algorithm=spec.algorithm,
                error=str(e),
                training_seconds=elapsed,
            )
            self._jobs[job_id]["status"] = JobStatus.FAILED
            self._jobs[job_id]["result"] = result
            return result

    def collect_result(self, job_id: str) -> TrainingResult:
        job = self._jobs.get(job_id)
        if not job:
            return TrainingResult(
                job_id=job_id,
                status=JobStatus.UNKNOWN,
                algorithm="",
                error="job not found",
            )
        if job["result"] is not None:
            return job["result"]
        return TrainingResult(
            job_id=job_id,
            status=job["status"],
            algorithm=job["spec"].algorithm,
            error="result not ready",
        )

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job["status"] in (JobStatus.PENDING, JobStatus.RUNNING):
            job["status"] = JobStatus.CANCELLED
            return True
        return False
