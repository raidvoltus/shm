"""
Bounded, TTL-aware signal delivery ledger.

Guarantee model: AT-MOST-ONCE notification (prefer miss over duplicate BUY/SELL).

Does NOT store secrets, tokens, chat IDs, or market payloads.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_TTL_HOURS = 24
DEFAULT_MAX_ENTRIES = 500
DEFAULT_PENDING_LEASE_MINUTES = 30


class DeliveryStatus(str, Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    TRANSIENT_FAILURE = "TRANSIENT_FAILURE"
    PERMANENT_FAILURE = "PERMANENT_FAILURE"
    UNKNOWN_DELIVERY_STATE = "UNKNOWN_DELIVERY_STATE"


TERMINAL_BLOCKING = frozenset(
    {
        DeliveryStatus.SUCCESS.value,
        DeliveryStatus.UNKNOWN_DELIVERY_STATE.value,
        DeliveryStatus.PENDING.value,
        DeliveryStatus.PERMANENT_FAILURE.value,
    }
)


class LedgerUnavailable(RuntimeError):
    """Persistent ledger cannot be read/written — fail-closed for notifications."""


@dataclass(frozen=True)
class LedgerEntry:
    signal_id: str
    created_at: str
    expires_at: str
    delivery_status: str
    symbol: str = ""
    intent: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "signal_id": self.signal_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "delivery_status": self.delivery_status,
        }
        if self.symbol:
            d["symbol"] = self.symbol
        if self.intent:
            d["intent"] = self.intent
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LedgerEntry":
        return cls(
            signal_id=str(d["signal_id"]),
            created_at=str(d["created_at"]),
            expires_at=str(d["expires_at"]),
            delivery_status=str(d["delivery_status"]),
            symbol=str(d.get("symbol") or ""),
            intent=str(d.get("intent") or ""),
        )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


class LedgerStore(Protocol):
    def load(self) -> dict[str, Any]:
        ...

    def save(self, payload: dict[str, Any]) -> None:
        ...


class MemoryLedgerStore:
    """In-process store for unit tests."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "entries": []}

    def load(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._data))

    def save(self, payload: dict[str, Any]) -> None:
        self._data = json.loads(json.dumps(payload))


class FileLedgerStore:
    """Atomic local JSON file store (tests / local dev)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": SCHEMA_VERSION, "entries": []}
        try:
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else {}
        except (OSError, json.JSONDecodeError) as e:
            raise LedgerUnavailable(f"file ledger read failed: {type(e).__name__}") from e
        if not isinstance(data, dict):
            raise LedgerUnavailable("file ledger corrupt: not an object")
        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("entries", [])
        return data

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        try:
            fd, tmp = tempfile.mkstemp(
                dir=str(self.path.parent), prefix=".ledger-", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(text)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError as e:
            raise LedgerUnavailable(f"file ledger write failed: {type(e).__name__}") from e


class IdempotencyLedger:
    """
    Cross-run delivery ledger with TTL and MAX_ENTRIES bound.

    Policy: FAIL_CLOSED when store raises LedgerUnavailable (caller decides).
    """

    def __init__(
        self,
        store: LedgerStore,
        *,
        ttl_hours: int = DEFAULT_TTL_HOURS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        pending_lease_minutes: int = DEFAULT_PENDING_LEASE_MINUTES,
    ) -> None:
        self.store = store
        self.ttl = timedelta(hours=ttl_hours)
        self.max_entries = max_entries
        self.pending_lease = timedelta(minutes=pending_lease_minutes)

    def _cleanup(self, entries: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for e in entries:
            try:
                exp = _parse_iso(str(e.get("expires_at", "")))
            except (TypeError, ValueError):
                continue
            if exp <= now:
                continue
            status = str(e.get("delivery_status", ""))
            if status == DeliveryStatus.PENDING.value:
                try:
                    created = _parse_iso(str(e.get("created_at", "")))
                    if created + self.pending_lease <= now:
                        continue
                except (TypeError, ValueError):
                    continue
            kept.append(e)
        kept.sort(key=lambda x: (str(x.get("expires_at", "")), str(x.get("signal_id", ""))))
        if len(kept) > self.max_entries:
            kept = kept[-self.max_entries :]
        return kept

    def _index(self, entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {str(e["signal_id"]): e for e in entries if e.get("signal_id")}

    def should_skip(self, signal_id: str) -> bool:
        data = self.store.load()
        now = _utcnow()
        entries = self._cleanup(list(data.get("entries") or []), now)
        idx = self._index(entries)
        e = idx.get(signal_id)
        if not e:
            return False
        return str(e.get("delivery_status")) in TERMINAL_BLOCKING

    def reserve(self, signal_id: str, *, symbol: str = "", intent: str = "") -> bool:
        data = self.store.load()
        now = _utcnow()
        entries = self._cleanup(list(data.get("entries") or []), now)
        idx = self._index(entries)
        existing = idx.get(signal_id)
        if existing and str(existing.get("delivery_status")) in TERMINAL_BLOCKING:
            return False
        entries = [e for e in entries if str(e.get("signal_id")) != signal_id]
        entry = LedgerEntry(
            signal_id=signal_id,
            created_at=_iso(now),
            expires_at=_iso(now + self.ttl),
            delivery_status=DeliveryStatus.PENDING.value,
            symbol=symbol,
            intent=intent,
        )
        entries.append(entry.to_dict())
        entries = self._cleanup(entries, now)
        payload = {"schema_version": SCHEMA_VERSION, "entries": entries}
        self.store.save(payload)
        return True

    def finalize(
        self,
        signal_id: str,
        status: DeliveryStatus,
        *,
        symbol: str = "",
        intent: str = "",
    ) -> None:
        if status == DeliveryStatus.PENDING:
            return
        data = self.store.load()
        now = _utcnow()
        entries = self._cleanup(list(data.get("entries") or []), now)
        entries = [e for e in entries if str(e.get("signal_id")) != signal_id]
        entry = LedgerEntry(
            signal_id=signal_id,
            created_at=_iso(now),
            expires_at=_iso(now + self.ttl),
            delivery_status=status.value,
            symbol=symbol,
            intent=intent,
        )
        entries.append(entry.to_dict())
        entries = self._cleanup(entries, now)
        payload = {"schema_version": SCHEMA_VERSION, "entries": entries}
        self.store.save(payload)

    def snapshot(self) -> dict[str, Any]:
        data = self.store.load()
        now = _utcnow()
        entries = self._cleanup(list(data.get("entries") or []), now)
        return {"schema_version": SCHEMA_VERSION, "entries": entries}
