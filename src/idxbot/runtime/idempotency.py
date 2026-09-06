"""
Execution identity and idempotency for duplicate-run protection.

Identical (run_id, scheduled_at) must not produce duplicate portfolio mutations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from idxbot.storage.backend import StorageBackend


@dataclass(frozen=True)
class ExecutionIdentity:
    run_id: str
    scheduled_at: datetime
    market_window: str
    state_version: int

    def key(self) -> str:
        return f"{self.run_id}|{self.scheduled_at.isoformat()}|{self.market_window}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scheduled_at": self.scheduled_at.isoformat(),
            "market_window": self.market_window,
            "state_version": self.state_version,
        }


class IdempotencyStore:
    """Persists completed execution identities to prevent re-processing."""

    KEY = "idempotency_log"

    def __init__(self, backend: StorageBackend) -> None:
        self.backend = backend

    def _load(self) -> tuple[dict[str, Any], int]:
        data = self.backend.load_state(self.KEY)
        if data is None:
            return {"completed": {}, "state_version": 0}, 0
        return data, int(data.get("state_version", 0))

    def already_executed(self, identity: ExecutionIdentity) -> bool:
        store, _ = self._load()
        return identity.key() in store.get("completed", {})

    def mark_completed(self, identity: ExecutionIdentity) -> None:
        store, version = self._load()
        completed = store.setdefault("completed", {})
        completed[identity.key()] = {
            **identity.to_dict(),
            "completed_at": datetime.now(identity.scheduled_at.tzinfo).isoformat(),
        }
        # Keep last 200 entries to bound size
        if len(completed) > 200:
            keys = sorted(completed.keys())
            for k in keys[: len(completed) - 200]:
                del completed[k]
        store["state_version"] = version + 1
        self.backend.save_state(self.KEY, store, expected_version=version)
