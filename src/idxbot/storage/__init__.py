"""Storage backends for IDX Signal Bot (local atomic only — no S3)."""

from __future__ import annotations

import os

from idxbot.storage.backend import (
    LocalStorageBackend,
    StateCorruptionError,
    StorageBackend,
    StorageError,
    VersionConflictError,
)
from idxbot.storage.state import AccountState, PortfolioState, Position

__all__ = [
    "AccountState",
    "LocalStorageBackend",
    "PortfolioState",
    "Position",
    "StateCorruptionError",
    "StorageBackend",
    "StorageError",
    "VersionConflictError",
    "build_storage_backend",
]


def build_storage_backend() -> StorageBackend:
    """Always local atomic storage. S3 is not supported."""
    return LocalStorageBackend(
        os.environ.get("IDXBOT_STATE_DIR", os.environ.get("STATE_DIR", ".state"))
    )
