"""
Purged walk-forward OOF generation — serial, disk-backed, float32.

Window i → model m → predict → flush parquet (no full matrix in RAM).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from idxbot.ml.candidates import CLASS_ORDER, list_candidates
from idxbot.ml.trainer import ModelTrainer, TrainResult


def _write_oof(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.oof.parquet")
    df = pd.DataFrame(rows)
    # float32 for prob columns
    for c in ("probability_down", "probability_flat", "probability_up"):
        if c in df.columns:
            df[c] = df[c].astype(np.float32)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp)
    tmp.replace(path)


class OOFGenerator:
    """
    Expanding walk-forward OOF for base models.

    For each fold: train on past, predict on validation fold, flush to disk.
    """

    def __init__(
        self,
        *,
        n_folds: int = 3,
        min_train: int = 30,
        val_size: int = 15,
        embargo: int = 6,
        algorithms: Optional[Sequence[str]] = None,
        random_seed: int = 42,
    ) -> None:
        self.n_folds = n_folds
        self.min_train = min_train
        self.val_size = val_size
        self.embargo = embargo
        self.algorithms = list(algorithms) if algorithms else [c.algorithm for c in list_candidates()]
        self.random_seed = random_seed

    def generate(
        self,
        rows: Sequence[dict[str, Any]],
        output_dir: str | Path,
    ) -> dict[str, Any]:
        """
        Serial OOF: for each fold × model → flush partition.
        Returns summary metadata (not full matrix).
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        n = len(rows)
        # build fold boundaries from the end
        folds = []
        end = n
        for fold_id in range(self.n_folds):
            val_end = end
            val_start = max(self.min_train + self.embargo, val_end - self.val_size)
            if val_start >= val_end:
                break
            train_end = max(0, val_start - self.embargo)
            if train_end < self.min_train:
                break
            folds.append((fold_id, 0, train_end, val_start, val_end))
            end = val_start
        folds = list(reversed(folds))

        total_rows = 0
        files = []
        trainer = ModelTrainer(self.random_seed)

        for fold_id, tr0, tr1, va0, va1 in folds:
            train_slice = list(rows[tr0:tr1])
            val_slice = list(rows[va0:va1])
            if len(train_slice) < 10 or len(val_slice) < 1:
                continue
            for algo in self.algorithms:
                try:
                    result = trainer.train(algo, train_slice, None, calibrate=False)
                except Exception:
                    continue  # model unavailable → skip, don't crash ensemble
                probs = result.predict_proba_dict(val_slice)
                batch = []
                for r, p in zip(val_slice, probs):
                    batch.append(
                        {
                            "timestamp": r["timestamp"],
                            "symbol": r["symbol"],
                            "horizon": r.get("horizon", 1),
                            "actual_label": r["label"],
                            "model_id": algo,
                            "model_version": "oof",
                            "probability_down": p["DOWN"],
                            "probability_flat": p["FLAT"],
                            "probability_up": p["UP"],
                            "fold_id": fold_id,
                            "train_end_idx": tr1,
                            "val_start_idx": va0,
                        }
                    )
                path = out / f"fold={fold_id}" / f"model={algo}.parquet"
                _write_oof(path, batch)
                files.append(str(path))
                total_rows += len(batch)

        return {
            "n_folds": len(folds),
            "n_rows": total_rows,
            "n_files": len(files),
            "files": files,
            "algorithms": self.algorithms,
            "embargo": self.embargo,
        }

    @staticmethod
    def load_oof(output_dir: str | Path) -> list[dict[str, Any]]:
        """Load all OOF partitions (use carefully on large data)."""
        root = Path(output_dir)
        rows: list[dict[str, Any]] = []
        for p in sorted(root.rglob("*.parquet")):
            if p.name.endswith(".tmp.oof.parquet"):
                continue
            rows.extend(pq.ParquetFile(p).read().to_pylist())
        return rows
