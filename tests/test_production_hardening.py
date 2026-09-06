from __future__ import annotations

import json

import pytest

from idxbot.runtime.idempotency import ExecutionIdentity, IdempotencyStore
from idxbot.storage.backend import LocalStorageBackend, StateCorruptionError, VersionConflictError


def test_local_storage_checksum_tamper_fails_closed(tmp_path):
    store = LocalStorageBackend(tmp_path)
    store.save_state("x", {"state_version": 1, "value": 42}, expected_version=0)
    path = tmp_path / "x.json"
    data = json.loads(path.read_text())
    data["value"] = 99
    path.write_text(json.dumps(data))
    with pytest.raises(StateCorruptionError):
        store.load_state("x")


def test_local_storage_missing_expected_version_is_conflict(tmp_path):
    store = LocalStorageBackend(tmp_path)
    with pytest.raises(VersionConflictError):
        store.save_state("x", {"state_version": 2}, expected_version=1)


def test_idempotency_uses_optimistic_versioning(tmp_path):
    store = LocalStorageBackend(tmp_path)
    idem = IdempotencyStore(store)
    identity = ExecutionIdentity("r1", __import__("datetime").datetime.now(__import__("datetime").timezone.utc), "w", 0)
    assert not idem.already_executed(identity)
    idem.mark_completed(identity)
    assert idem.already_executed(identity)
    raw = store.load_state("idempotency_log")
    assert raw["state_version"] == 1
