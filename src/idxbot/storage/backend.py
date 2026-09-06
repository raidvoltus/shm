"""
Storage abstraction for IDX Signal Bot.

Local atomic filesystem storage only. S3/object storage is not used.

Atomic write pattern is mandatory. Integrity/version failures fail-closed.

Atomic write pattern:
  state.json → state.tmp → write → flush → fsync → validate → os.replace → state.json

No cloud credentials are stored in source code.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class StorageError(Exception):
    """Base storage failure."""


class StateCorruptionError(StorageError):
    """State file is unreadable or fails validation — fail closed."""


class VersionConflictError(StorageError):
    """Optimistic concurrency conflict — do not overwrite."""


class StorageBackend(ABC):
    """Minimal persistence contract."""

    @abstractmethod
    def load_state(self, key: str) -> Optional[dict[str, Any]]:
        """Load state by key. Returns None if missing."""
        ...

    @abstractmethod
    def save_state(
        self, key: str, data: dict[str, Any], *, expected_version: Optional[int] = None
    ) -> None:
        """Persist state under key (atomic where possible)."""
        ...

    @abstractmethod
    def delete_state(self, key: str) -> None:
        """Remove state for key (idempotent)."""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check whether key exists."""
        ...


def _checksum(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class LocalStorageBackend(StorageBackend):
    """
    Filesystem backend with atomic replace and optional version checks.

    Ephemeral on GitHub Actions runners — treat as job-local state only, not durable production DB.
    """

    def __init__(self, root: str | Path = ".state") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = key.replace("..", "").replace("/", "_").replace("\\", "_")
        return self.root / f"{safe}.json"

    def _tmp_path(self, key: str) -> Path:
        return self._path(key).with_suffix(".tmp")

    def load_state(self, key: str) -> Optional[dict[str, Any]]:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            raw = path.read_bytes()
            data = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StateCorruptionError(
                f"Corrupt state for key={key!r}: {exc}. FAIL CLOSED — do not reset balance."
            ) from exc
        if not isinstance(data, dict):
            raise StateCorruptionError(
                f"State for key={key!r} is not a dict. FAIL CLOSED."
            )
        stored_checksum = data.get("_checksum")
        if stored_checksum:
            unsigned = {k: v for k, v in data.items() if k != "_checksum"}
            canonical = json.dumps(
                unsigned, indent=2, default=str, sort_keys=True
            ).encode("utf-8")
            if stored_checksum != _checksum(canonical):
                raise StateCorruptionError(
                    f"Checksum mismatch for key={key!r}. FAIL CLOSED."
                )
        return data

    def save_state(
        self,
        key: str,
        data: dict[str, Any],
        *,
        expected_version: Optional[int] = None,
    ) -> None:
        """
        Atomic save:
          1. serialize to temp
          2. flush + fsync
          3. validate by re-reading temp
          4. os.replace onto final path

        If expected_version is set, refuse to overwrite when stored version differs.
        """
        path = self._path(key)
        tmp = self._tmp_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        if expected_version is not None:
            current = self.load_state(key) if path.exists() else None
            if current is None:
                if expected_version != 0:
                    raise VersionConflictError(
                        f"Version conflict for {key!r}: expected {expected_version}, "
                        "but state is missing. DO NOT OVERWRITE."
                    )
            else:
                current_ver = int(current.get("state_version", 0))
                if current_ver != expected_version:
                    raise VersionConflictError(
                        f"Version conflict for {key!r}: expected {expected_version}, "
                        f"found {current_ver}. DO NOT OVERWRITE."
                    )

        if "state_version" not in data:
            data = {**data, "state_version": (expected_version or 0) + 1}

        payload = json.dumps(data, indent=2, default=str, sort_keys=True).encode("utf-8")
        meta = {
            **data,
            "_checksum": _checksum(payload),
        }
        final_payload = json.dumps(meta, indent=2, default=str, sort_keys=True).encode(
            "utf-8"
        )

        try:
            with tmp.open("wb") as f:
                f.write(final_payload)
                f.flush()
                os.fsync(f.fileno())

            check = json.loads(tmp.read_bytes().decode("utf-8"))
            if not isinstance(check, dict):
                raise StorageError("Temp validation failed: not a dict")

            os.replace(tmp, path)
        except Exception:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            raise

    def delete_state(self, key: str) -> None:
        path = self._path(key)
        tmp = self._tmp_path(key)
        if path.exists():
            path.unlink()
        if tmp.exists():
            tmp.unlink()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()
