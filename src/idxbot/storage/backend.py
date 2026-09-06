"""
Storage abstraction for IDX Signal Bot.

GitHub Actions runners are ephemeral. Production persistence must use
object storage (S3-compatible). LocalStorageBackend is for local tests only.

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

    Not suitable for production GitHub Actions runners (ephemeral disk).
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

        # Optimistic concurrency
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

        # Ensure version field present
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

            # Validate temp is readable JSON
            check = json.loads(tmp.read_bytes().decode("utf-8"))
            if not isinstance(check, dict):
                raise StorageError("Temp validation failed: not a dict")

            os.replace(tmp, path)  # atomic on POSIX
        except Exception:
            # Leave previous state.json intact
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


class ObjectStorageBackend(StorageBackend):
    """
    Contract for S3-compatible object storage.

    Implementation is deferred. Credentials via env / IAM — never hardcoded.
    """

    def __init__(self, bucket: str, prefix: str = "idxbot/") -> None:
        self.bucket = bucket
        self.prefix = prefix
        raise NotImplementedError(
            "ObjectStorageBackend is a contract only. "
            "Concrete S3-compatible implementation will be added later. "
            "Do not hardcode credentials."
        )

    def load_state(self, key: str) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    def save_state(
        self, key: str, data: dict[str, Any], *, expected_version: Optional[int] = None
    ) -> None:
        raise NotImplementedError

    def delete_state(self, key: str) -> None:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError
