"""Colab provider: unavailable fallback + artifact rejection."""

import json
from pathlib import Path

import pytest

from idxbot.ml.providers.colab import (
    ColabComputeProvider,
    ColabArtifactValidationError,
    validate_colab_artifact,
)
from idxbot.ml.providers.local import LocalComputeProvider
from idxbot.ml.providers.base import JobStatus, TrainingJobSpec


def test_colab_unavailable_by_default():
    p = ColabComputeProvider(enabled=False)
    assert p.availability() is False


def test_colab_enabled_without_endpoint_unavailable():
    p = ColabComputeProvider(enabled=True, endpoint="")
    assert p.availability() is False


def test_local_always_available():
    p = LocalComputeProvider()
    assert p.availability() is True
    caps = p.capabilities()
    assert caps.name == "local"
    assert caps.max_runtime_minutes <= 19.0 or caps.max_runtime_minutes == 19.0


def test_reject_forbidden_metadata():
    with pytest.raises(ColabArtifactValidationError):
        validate_colab_artifact(
            {"algorithm": "x", "job_id": "1", "metrics": {}, "exec": "rm -rf /"}
        )


def test_reject_missing_keys():
    with pytest.raises(ColabArtifactValidationError):
        validate_colab_artifact({"algorithm": "x"})


def test_reject_path_traversal():
    with pytest.raises(ColabArtifactValidationError):
        validate_colab_artifact(
            {"algorithm": "x", "job_id": "1", "metrics": {}},
            artifact_path="../etc/passwd",
        )


def test_valid_metadata_passes():
    validate_colab_artifact(
        {"algorithm": "logistic_regression", "job_id": "j1", "metrics": {"auc": 0.5}}
    )


def test_collect_invalid_result(tmp_path):
    p = ColabComputeProvider(enabled=True, endpoint="http://example.invalid")
    p.artifact_dir = tmp_path
    bad = tmp_path / "job1.result.json"
    bad.write_text(json.dumps({"code": "evil"}), encoding="utf-8")
    result = p.collect_result("job1")
    assert result.status == JobStatus.FAILED
    assert "invalid" in (result.error or "").lower()
