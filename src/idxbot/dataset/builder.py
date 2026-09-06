"""
DatasetBuilder — joins features + labels into a deterministic ML dataset.

No random shuffle. Chronological order preserved.
Labels are separate from features (INVARIANT 1).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from idxbot.dataset.labels import LabelClass, LabelGenerator, LabelResult


@dataclass
class DatasetManifest:
    dataset_version: str
    created_at: str
    feature_version: str
    label_version: str
    source_version: str
    row_count: int
    symbols: list[str]
    timeframe: str
    label_horizons: list[int]
    embargo_period: int
    schema_version: str = "5.0"
    adjustment_mode: str = "ADJUSTED"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetBuilder:
    def __init__(
        self,
        label_generator: Optional[LabelGenerator] = None,
        *,
        feature_version: str = "1",
        data_version: str = "1",
        dataset_version: str = "5.0.0",
        source: str = "idxbot",
    ) -> None:
        self.label_generator = label_generator or LabelGenerator()
        self.feature_version = feature_version
        self.data_version = data_version
        self.dataset_version = dataset_version
        self.source = source

    def build(
        self,
        feature_rows: Sequence[dict[str, Any]],
        closes_by_symbol: dict[str, list[float]],
        timestamps_by_symbol: dict[str, list[str]],
    ) -> tuple[list[dict[str, Any]], DatasetManifest]:
        """
        Join features with multi-horizon labels.

        closes_by_symbol / timestamps_by_symbol must be aligned oldest→newest
        and used for label generation only.
        """
        # Index features by (symbol, timestamp)
        feat_index: dict[tuple[str, str], dict[str, Any]] = {}
        for r in feature_rows:
            key = (str(r["symbol"]), str(r["timestamp"]))
            feat_index[key] = r

        labels: list[LabelResult] = []
        for sym in sorted(closes_by_symbol.keys()):
            closes = closes_by_symbol[sym]
            ts = timestamps_by_symbol[sym]
            labels.extend(self.label_generator.label_symbol(sym, ts, closes))

        rows: list[dict[str, Any]] = []
        symbols: set[str] = set()
        for lab in labels:
            key = (lab.symbol, lab.timestamp)
            feat = feat_index.get(key)
            if feat is None:
                continue
            row = {
                **{k: v for k, v in feat.items()},
                "horizon": lab.horizon,
                "future_return": lab.future_return,
                "label_threshold": lab.threshold,
                "label": lab.label.value,
                "label_start_idx": lab.label_start_idx,
                "label_end_idx": lab.label_end_idx,
                "observation_idx": lab.observation_idx,
                "feature_version": self.feature_version,
                "data_version": self.data_version,
                "dataset_version": self.dataset_version,
                "label_version": self.label_generator.label_version,
                "adjustment_mode": feat.get("adjustment_mode", "ADJUSTED"),
                "source": self.source,
                "timeframe": feat.get("timeframe", "1d"),
            }
            rows.append(row)
            symbols.add(lab.symbol)

        # Deterministic order: timestamp, symbol, horizon
        rows.sort(key=lambda r: (str(r["timestamp"]), r["symbol"], r["horizon"]))

        manifest = DatasetManifest(
            dataset_version=self.dataset_version,
            created_at=datetime.now(timezone.utc).isoformat(),
            feature_version=self.feature_version,
            label_version=self.label_generator.label_version,
            source_version=self.data_version,
            row_count=len(rows),
            symbols=sorted(symbols),
            timeframe="1d",
            label_horizons=list(self.label_generator.horizons),
            embargo_period=self.label_generator.embargo_period(),
        )
        return rows, manifest

    @staticmethod
    def training_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Exclude PENDING labels from training."""
        return [r for r in rows if r.get("label") != LabelClass.PENDING.value]

    @staticmethod
    def deterministic_hash(rows: Sequence[dict[str, Any]]) -> str:
        payload = json.dumps(rows, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()
