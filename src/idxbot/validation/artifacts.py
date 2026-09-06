"""Immutable validation run artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ValidationArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_run(self, run_id: str, payload: dict[str, Any]) -> Path:
        d = self.root / "phase8" / run_id
        if d.exists() and (d / "manifest.json").exists():
            raise FileExistsError(f"Immutable run exists: {run_id}")
        d.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": "8.0",
        }
        (d / "statistics.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raw = (d / "statistics.json").read_bytes()
        checksum = {"sha256": hashlib.sha256(raw).hexdigest()}
        (d / "checksum.json").write_text(json.dumps(checksum, indent=2), encoding="utf-8")
        return d
