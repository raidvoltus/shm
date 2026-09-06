"""Structured governor decision events — no secrets."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


def build_audit_event(
    snapshot: dict[str, Any],
    selection: dict[str, Any],
    ensemble: Optional[dict[str, Any]] = None,
    *,
    retraining_status: str = "n/a",
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "memory_limit_bytes": snapshot.get("memory_limit_bytes"),
        "memory_used_bytes": snapshot.get("memory_used_bytes"),
        "memory_available_bytes": snapshot.get("memory_available_bytes"),
        "cpu_count": snapshot.get("cpu_count"),
        "cpu_load": snapshot.get("cpu_load"),
        "disk_available_bytes": snapshot.get("disk_available_bytes"),
        "selected_models": selection.get("selected_models"),
        "disabled_models": selection.get("disabled_models"),
        "degradation_level": selection.get("degradation_level"),
        "ensemble_mode": (ensemble or {}).get("ensemble_mode"),
        "estimated_memory_bytes": selection.get("estimated_memory_bytes"),
        "reason": selection.get("reason"),
        "decision": selection.get("decision"),
        "retraining_status": retraining_status,
        "weights": (ensemble or {}).get("weights"),
    }
