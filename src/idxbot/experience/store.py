"""
Partitioned permanent Experience Store.

experiences/
  year=YYYY/
    month=MM/
      experiences.parquet

Dedup key: symbol + timestamp + horizon + feature_version + label_version
Append is idempotent. Paper account reset must NOT touch this store.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).read().to_pylist()
    except ImportError:
        # JSONL fallback when pyarrow unavailable
        jpath = path.with_suffix(".jsonl")
        if not jpath.exists():
            return []
        import json
        rows = []
        for line in jpath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        df = pd.DataFrame(rows)
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path)
    except ImportError:
        import json
        jpath = path.with_suffix(".jsonl")
        existing = []
        if jpath.exists():
            for line in jpath.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    existing.append(json.loads(line))
        # dedup by experience_id if present
        seen = {r.get("experience_id") for r in existing if r.get("experience_id")}
        for r in rows:
            eid = r.get("experience_id")
            if eid and eid in seen:
                continue
            existing.append(r)
            if eid:
                seen.add(eid)
        with jpath.open("w", encoding="utf-8") as f:
            for r in existing:
                f.write(json.dumps(r, default=str) + "\n")


@dataclass
class ExperienceRecord:
    experience_id: str
    symbol: str
    timestamp: str
    horizon: int
    label: str
    realized_return: Optional[float] = None
    outcome: Optional[str] = None
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    regime: Optional[str] = None
    features_ref: Optional[str] = None
    dataset_version: str = "5.0.0"
    feature_version: str = "1"
    label_version: str = "1"
    model_version: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @staticmethod
    def make_id(
        symbol: str,
        timestamp: str,
        horizon: int,
        feature_version: str,
        label_version: str,
    ) -> str:
        raw = f"{symbol}|{timestamp}|{horizon}|{feature_version}|{label_version}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperienceStore:
    """Partitioned append-only experience store with deterministic dedup."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _partition_path(self, timestamp: str) -> Path:
        # timestamp ISO → year/month
        ts = timestamp[:10]  # YYYY-MM-DD
        year, month = ts[:4], ts[5:7]
        return self.root / f"year={year}" / f"month={month}" / "experiences.parquet"

    def append(self, records: Sequence[ExperienceRecord]) -> dict[str, Any]:
        """
        Idempotent append into active partitions only.
        Complexity: O(active partitions touched), not O(full history).
        """
        by_path: dict[Path, list[ExperienceRecord]] = {}
        for rec in records:
            path = self._partition_path(rec.timestamp)
            by_path.setdefault(path, []).append(rec)

        added = 0
        skipped = 0
        for path, group in by_path.items():
            existing = _read_parquet(path)
            ids = {str(r.get("experience_id")) for r in existing}
            new_rows = []
            for rec in group:
                if rec.experience_id in ids:
                    skipped += 1
                    continue
                ids.add(rec.experience_id)
                new_rows.append(rec.to_dict())
                added += 1
            if new_rows:
                merged = existing + new_rows
                # stable sort
                merged.sort(key=lambda r: (str(r.get("timestamp")), str(r.get("symbol")), int(r.get("horizon", 0))))
                _write_parquet(path, merged)
        return {"added": added, "skipped_duplicates": skipped, "partitions_touched": len(by_path)}

    def append_from_dataset_rows(
        self, rows: Sequence[dict[str, Any]], *, model_version: Optional[str] = None
    ) -> dict[str, Any]:
        records = []
        for r in rows:
            if r.get("label") == "PENDING":
                continue
            eid = ExperienceRecord.make_id(
                str(r["symbol"]),
                str(r["timestamp"]),
                int(r["horizon"]),
                str(r.get("feature_version", "1")),
                str(r.get("label_version", "1")),
            )
            records.append(
                ExperienceRecord(
                    experience_id=eid,
                    symbol=str(r["symbol"]),
                    timestamp=str(r["timestamp"]),
                    horizon=int(r["horizon"]),
                    label=str(r["label"]),
                    realized_return=r.get("future_return"),
                    outcome=str(r["label"]),
                    dataset_version=str(r.get("dataset_version", "5.0.0")),
                    feature_version=str(r.get("feature_version", "1")),
                    label_version=str(r.get("label_version", "1")),
                    model_version=model_version,
                )
            )
        return self.append(records)

    def count(self) -> int:
        total = 0
        for path in self.root.rglob("experiences.parquet"):
            total += len(_read_parquet(path))
        return total

    def load_partition(self, year: int, month: int) -> list[dict[str, Any]]:
        path = self.root / f"year={year:04d}" / f"month={month:02d}" / "experiences.parquet"
        return _read_parquet(path)

    def unique_ids(self) -> set[str]:
        ids: set[str] = set()
        for path in self.root.rglob("experiences.parquet"):
            for r in _read_parquet(path):
                ids.add(str(r.get("experience_id")))
        return ids
