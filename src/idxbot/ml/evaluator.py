"""Multi-metric evaluation + regime breakdown. Test set = holdout only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from idxbot.ml.candidates import CLASS_ORDER, CLASS_TO_INT
from idxbot.ml.trainer import TrainResult


@dataclass
class EvalMetrics:
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    log_loss: float
    confusion: list[list[int]]
    n_samples: int
    class_hit_rate: dict[str, float] = field(default_factory=dict)
    avg_future_return_by_pred: dict[str, float] = field(default_factory=dict)
    regime_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    inference_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "log_loss": self.log_loss,
            "confusion": self.confusion,
            "n_samples": self.n_samples,
            "class_hit_rate": self.class_hit_rate,
            "avg_future_return_by_pred": self.avg_future_return_by_pred,
            "regime_metrics": self.regime_metrics,
            "inference_time_ms": self.inference_time_ms,
        }


class ModelEvaluator:
    def evaluate(
        self,
        result: TrainResult,
        rows: Sequence[dict[str, Any]],
        *,
        include_regime: bool = True,
    ) -> EvalMetrics:
        import time

        if not rows:
            raise ValueError("empty evaluation set")
        t0 = time.perf_counter()
        probs = result.predict_proba_dict(rows)
        preds = [max(p, key=p.get) for p in probs]  # type: ignore
        inference_ms = (time.perf_counter() - t0) * 1000

        y_true = [r["label"] for r in rows]
        y_true_int = [CLASS_TO_INT[y] for y in y_true]
        y_pred_int = [CLASS_TO_INT[p] for p in preds]
        P = np.array([[p[c] for c in CLASS_ORDER] for p in probs])

        # log_loss needs labels present in both
        try:
            ll = float(log_loss(y_true_int, P, labels=list(range(len(CLASS_ORDER)))))
        except Exception:
            ll = float("nan")

        cm = confusion_matrix(y_true_int, y_pred_int, labels=list(range(3))).tolist()

        # class hit rate
        hit: dict[str, float] = {}
        for c in CLASS_ORDER:
            idx = [i for i, yt in enumerate(y_true) if yt == c]
            if not idx:
                hit[c] = float("nan")
            else:
                hit[c] = sum(1 for i in idx if preds[i] == c) / len(idx)

        # avg future return by predicted class
        avg_ret: dict[str, float] = {}
        for c in CLASS_ORDER:
            rets = [
                float(rows[i].get("future_return") or 0.0)
                for i, p in enumerate(preds)
                if p == c and rows[i].get("future_return") is not None
            ]
            avg_ret[c] = sum(rets) / len(rets) if rets else float("nan")

        regime_metrics: dict[str, dict[str, float]] = {}
        if include_regime:
            for key in ("trend_regime", "volatility_regime"):
                buckets: dict[str, list[int]] = {}
                for i, r in enumerate(rows):
                    v = r.get(key)
                    if v is None:
                        continue
                    # coarse buckets
                    if key == "trend_regime":
                        b = "BULL" if float(v) > 0 else ("BEAR" if float(v) < 0 else "SIDEWAYS")
                    else:
                        b = "HIGH_VOL" if float(v) >= 0.66 else ("LOW_VOL" if float(v) <= 0.33 else "MID_VOL")
                    buckets.setdefault(b, []).append(i)
                for b, idxs in buckets.items():
                    if len(idxs) < 3:
                        continue
                    acc = sum(1 for i in idxs if preds[i] == y_true[i]) / len(idxs)
                    regime_metrics[f"{key}:{b}"] = {"accuracy": acc, "n": float(len(idxs))}

        return EvalMetrics(
            accuracy=float(accuracy_score(y_true_int, y_pred_int)),
            balanced_accuracy=float(balanced_accuracy_score(y_true_int, y_pred_int)),
            precision_macro=float(
                precision_score(y_true_int, y_pred_int, average="macro", zero_division=0)
            ),
            recall_macro=float(
                recall_score(y_true_int, y_pred_int, average="macro", zero_division=0)
            ),
            f1_macro=float(f1_score(y_true_int, y_pred_int, average="macro", zero_division=0)),
            log_loss=ll,
            confusion=cm,
            n_samples=len(rows),
            class_hit_rate=hit,
            avg_future_return_by_pred=avg_ret,
            regime_metrics=regime_metrics,
            inference_time_ms=inference_ms,
        )
