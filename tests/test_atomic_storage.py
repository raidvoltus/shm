"""Atomic save, crash simulation, version conflict, corruption fail-closed."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from idxbot.storage.backend import (
    LocalStorageBackend,
    StateCorruptionError,
    VersionConflictError,
)


def test_atomic_save_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        be = LocalStorageBackend(tmp)
        be.save_state("p", {"cash": 10_000_000, "state_version": 1})
        loaded = be.load_state("p")
        assert loaded["cash"] == 10_000_000
        assert "_checksum" in loaded


def test_previous_state_survives_failed_write():
    """Simulate crash after temp write but before replace — old state remains."""
    with tempfile.TemporaryDirectory() as tmp:
        be = LocalStorageBackend(tmp)
        be.save_state("p", {"v": 1, "state_version": 1})
        assert be.load_state("p")["v"] == 1

        # Force failure during os.replace
        real_replace = os.replace

        def boom(src, dst):
            raise OSError("simulated atomic replace failure")

        with patch("os.replace", side_effect=boom):
            with pytest.raises(OSError):
                be.save_state("p", {"v": 2, "state_version": 2})

        # Previous valid state still readable
        loaded = be.load_state("p")
        assert loaded["v"] == 1


def test_temp_cleaned_on_failure():
    with tempfile.TemporaryDirectory() as tmp:
        be = LocalStorageBackend(tmp)
        be.save_state("p", {"v": 1, "state_version": 1})

        def boom_write(*a, **k):
            raise OSError("disk full")

        with patch.object(Path, "open", side_effect=boom_write):
            with pytest.raises(OSError):
                be.save_state("p", {"v": 99})

        assert be.load_state("p")["v"] == 1
        # no leftover .tmp that would confuse next load of final path
        assert not list(Path(tmp).glob("*.tmp"))


def test_corrupt_state_fail_closed():
    with tempfile.TemporaryDirectory() as tmp:
        be = LocalStorageBackend(tmp)
        path = be._path("p")
        path.write_text("{not valid json!!!", encoding="utf-8")
        with pytest.raises(StateCorruptionError) as ei:
            be.load_state("p")
        assert "FAIL CLOSED" in str(ei.value)


def test_version_conflict():
    with tempfile.TemporaryDirectory() as tmp:
        be = LocalStorageBackend(tmp)
        be.save_state("p", {"state_version": 5, "x": 1})
        with pytest.raises(VersionConflictError):
            be.save_state("p", {"state_version": 6, "x": 2}, expected_version=3)


def test_interrupted_save_no_reset():
    """Corrupt file must not silently reset balance."""
    with tempfile.TemporaryDirectory() as tmp:
        be = LocalStorageBackend(tmp)
        path = be._path("portfolio")
        path.write_bytes(b"\x00\x01corrupt")
        with pytest.raises(StateCorruptionError):
            be.load_state("portfolio")
