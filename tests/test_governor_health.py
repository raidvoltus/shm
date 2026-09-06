"""Lazy health check — metadata only, no full model load."""

import json
import tempfile
from pathlib import Path

from idxbot.governor.health import health_from_metadata, scan_registry_metadata


def test_missing_sha_unhealthy():
    h = health_from_metadata("logistic_regression", {"feature_version": "1"})
    assert h.healthy is False
    assert "sha" in h.reason


def test_feature_mismatch():
    h = health_from_metadata(
        "rf",
        {
            "artifact_sha256": "d" * 32,
            "feature_version": "1",
            "estimated_memory_bytes": 10,
            "artifact_size_bytes": 10,
        },
        expected_feature_version="2",
    )
    assert h.healthy is False
    assert "feature_version_mismatch" in h.reason


def test_scan_registry_no_joblib_load():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "algorithm=logistic_regression" / "version=v1"
        d.mkdir(parents=True)
        meta = {
            "artifact_sha256": "e" * 40,
            "estimated_memory_bytes": 12345,
            "artifact_size_bytes": 100,
            "feature_version": "1",
            "dataset_version": "5.0.0",
            "model_version": "v1",
        }
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        # no model.joblib — health must still work
        results = scan_registry_metadata(tmp, ["logistic_regression", "missing_algo"])
        assert results[0].healthy is True
        assert results[0].estimated_memory_bytes == 12345
        assert results[1].healthy is False
