"""
Partitioned Parquet dataset + delta architecture.

market_data/
  symbol=BBCA/
    year=2024/
      month=03/
        data.parquet

delta/
  date=2024-03-15/
    batch-001.parquet

Compaction merges deltas into partitions without rewriting entire history
on every incremental candle.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from idxbot.data.normalize import NormalizedBar


def _read_parquet_file(path: Path) -> list[dict[str, Any]]:
    """Read single parquet without hive-partition schema merge conflicts."""
    return pq.ParquetFile(path).read().to_pylist()


def _df_to_table(df: pd.DataFrame) -> pa.Table:
    if "symbol" in df.columns:
        df = df.copy()
        df["symbol"] = df["symbol"].astype(str)
    return pa.Table.from_pandas(df, preserve_index=False)


def _bars_to_table(bars: Sequence[NormalizedBar]) -> pa.Table:
    records = [b.to_record() for b in bars]
    if not records:
        return pa.table({})
    for r in records:
        if r.get("symbol") is not None:
            r["symbol"] = str(r["symbol"])
    return _df_to_table(pd.DataFrame(records))


@dataclass
class Provenance:
    provider: str
    retrieved_at: str
    source_timestamp: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: str = "1d"
    dataset_type: str = "ohlcv"
    adjustment_mode: str = "ADJUSTED"
    schema_version: str = "3.0"
    data_version: str = "1"
    request_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class PartitionedParquetStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.market_dir = self.root / "market_data"
        self.delta_dir = self.root / "delta"
        self.meta_dir = self.root / "meta"
        self.market_dir.mkdir(parents=True, exist_ok=True)
        self.delta_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def _partition_path(self, symbol: str, ts: datetime) -> Path:
        return (
            self.market_dir
            / f"symbol={symbol.upper()}"
            / f"year={ts.year:04d}"
            / f"month={ts.month:02d}"
            / "data.parquet"
        )

    def write_bars(
        self, bars: Sequence[NormalizedBar], *, overwrite_partition: bool = False
    ) -> list[Path]:
        by_part: dict[Path, list[NormalizedBar]] = {}
        for b in bars:
            p = self._partition_path(b.symbol, b.timestamp)
            by_part.setdefault(p, []).append(b)

        written: list[Path] = []
        for path, group in by_part.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            existing: list[dict[str, Any]] = []
            if path.exists() and not overwrite_partition:
                existing = _read_parquet_file(path)
            merged: dict[tuple[str, str], dict[str, Any]] = {}
            for rec in existing:
                key = (str(rec.get("symbol")), str(rec.get("timestamp")))
                merged[key] = rec
            for b in group:
                rec = b.to_record()
                key = (str(rec["symbol"]), str(rec["timestamp"]))
                merged[key] = rec
            rows = sorted(merged.values(), key=lambda r: str(r.get("timestamp")))
            table = _df_to_table(pd.DataFrame(rows))
            pq.write_table(table, path)
            written.append(path)
        return written

    def write_delta(
        self, bars: Sequence[NormalizedBar], *, batch_date: str, batch_id: int = 1
    ) -> Path:
        ddir = self.delta_dir / f"date={batch_date}"
        ddir.mkdir(parents=True, exist_ok=True)
        path = ddir / f"batch-{batch_id:03d}.parquet"
        pq.write_table(_bars_to_table(bars), path)
        return path

    def read_partition(self, symbol: str, year: int, month: int) -> list[dict[str, Any]]:
        path = (
            self.market_dir
            / f"symbol={symbol.upper()}"
            / f"year={year:04d}"
            / f"month={month:02d}"
            / "data.parquet"
        )
        if not path.exists():
            return []
        return _read_parquet_file(path)

    def list_deltas(self, batch_date: Optional[str] = None) -> list[Path]:
        if batch_date:
            d = self.delta_dir / f"date={batch_date}"
            if not d.exists():
                return []
            return sorted(d.glob("batch-*.parquet"))
        return sorted(self.delta_dir.glob("date=*/batch-*.parquet"))

    def save_provenance(self, key: str, prov: Provenance) -> Path:
        path = self.meta_dir / f"{key}.json"
        path.write_text(json.dumps(prov.to_dict(), indent=2), encoding="utf-8")
        return path

    def load_cursor(self, key: str = "last_successful") -> Optional[dict[str, Any]]:
        path = self.meta_dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_cursor(self, data: dict[str, Any], key: str = "last_successful") -> None:
        path = self.meta_dir / f"{key}.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


class DataCompactor:
    """delta files → dedupe → sort → validate → merge → partitioned historical."""

    def __init__(self, store: PartitionedParquetStore) -> None:
        self.store = store

    def compact(self, batch_date: str) -> dict[str, Any]:
        paths = self.store.list_deltas(batch_date)
        if not paths:
            return {"status": "noop", "rows": 0, "partitions": 0}

        rows: list[dict[str, Any]] = []
        for p in paths:
            rows.extend(_read_parquet_file(p))

        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for r in rows:
            key = (str(r.get("symbol")), str(r.get("timestamp")))
            merged[key] = r
        deduped = list(merged.values())

        bars: list[NormalizedBar] = []
        for r in deduped:
            try:
                bars.append(NormalizedBar.model_validate(r))
            except Exception:
                continue

        written = self.store.write_bars(bars)
        return {
            "status": "ok",
            "rows": len(bars),
            "partitions": len(written),
            "delta_files": len(paths),
        }
