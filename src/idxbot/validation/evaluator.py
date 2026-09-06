"""
Walk-forward multi-model evaluator.

Final Test is NEVER used for training, selection, or calibration.
Champion decision: WIN / HOLD / FAIL (no live promotion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from idxbot.ml.candidates import CLASS_ORDER, CLASS_TO_INT, get_candidate, list_candidates
from idxbot.ml.trainer import ModelTrainer
from idxbot.ml.evaluator import ModelEvaluator
from idxbot.ensemble.engine import EnsembleEngine
from idxbot.ensemble.weighted import WeightedAverageEnsemble
from idxbot.ensemble.contracts import ModelPrediction
from idxbot.validation.final_test import FinalTestGuard, FinalTestViolation
from idxbot.validation.stats import (
    compute_stability,
    holm_correction,
    wilcoxon_signed_rank,
)
from idxbot.validation.walkforward import RollingWalkForward, WalkForwardFold, dynamic_embargo


@dataclass
class FoldMetrics:
    fold_id: int
    model_id: str
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    log_loss: float
    brier: float
    precision_macro: float
    recall_macro: float
    directional_accuracy: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ChampionDecision:
    status: str  # WIN | HOLD | FAIL
    champion: str
    challenger: str
    reason: str
    p_value: float
    adjusted_p_value: float
    effect_size: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class WalkForwardEvaluator:
    def __init__(
        self,
        *,
        guard: Optional[FinalTestGuard] = None,
        max_train_lookback: int = 200,
        val_size: int = 25,
        step: int = 15,
        embargo: int = 6,
        random_seed: int = 42,
        algorithms: Optional[Sequence[str]] = None,
    ) -> None:
        self.guard = guard or FinalTestGuard()
        self.wf = RollingWalkForward(
            max_train_lookback=max_train_lookback,
            val_size=val_size,
            step=step,
            embargo=embargo,
            min_train=30,
        )
        self.random_seed = random_seed
        self.algorithms = list(algorithms) if algorithms else [
            "logistic_regression",
            "ridge_classifier",
            "random_forest",
        ]

    def evaluate(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        final_test_rows: Optional[Sequence[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        if final_test_rows:
            self.guard.lock(final_test_rows)

        folds = list(self.wf.split(rows))
        fold_metrics: list[FoldMetrics] = []
        per_model_bal_acc: dict[str, list[float]] = {a: [] for a in self.algorithms}
        per_model_bal_acc["weighted_ensemble"] = []

        trainer = ModelTrainer(self.random_seed)

        for fold in folds:
            train_rows = [rows[i] for i in fold.train_indices]
            val_rows = [rows[i] for i in fold.validation_indices]
            # HARD: final test cannot enter training
            self.guard.assert_not_in_training(train_rows, context="walkforward_train")

            model_results = {}
            for algo in self.algorithms:
                try:
                    result = trainer.train(algo, train_rows, None, calibrate=False)
                    model_results[algo] = result
                    m = self._metrics(algo, result, val_rows, fold.fold_id)
                    fold_metrics.append(m)
                    per_model_bal_acc[algo].append(m.balanced_accuracy)
                except Exception:
                    continue

            # weighted ensemble on available models
            if len(model_results) >= 2:
                eng = EnsembleEngine()
                preds = eng.predict_from_train_results(model_results, val_rows)
                y_true = [r["label"] for r in val_rows]
                y_pred = [p.predicted_class for p in preds]
                bal = float(
                    balanced_accuracy_score(
                        [CLASS_TO_INT[y] for y in y_true],
                        [CLASS_TO_INT[p] for p in y_pred],
                    )
                )
                per_model_bal_acc["weighted_ensemble"].append(bal)
                fold_metrics.append(
                    FoldMetrics(
                        fold_id=fold.fold_id,
                        model_id="weighted_ensemble",
                        accuracy=float(accuracy_score(
                            [CLASS_TO_INT[y] for y in y_true],
                            [CLASS_TO_INT[p] for p in y_pred],
                        )),
                        balanced_accuracy=bal,
                        f1_macro=float(
                            f1_score(
                                [CLASS_TO_INT[y] for y in y_true],
                                [CLASS_TO_INT[p] for p in y_pred],
                                average="macro",
                                zero_division=0,
                            )
                        ),
                        log_loss=float("nan"),
                        brier=float("nan"),
                        precision_macro=float(
                            precision_score(
                                [CLASS_TO_INT[y] for y in y_true],
                                [CLASS_TO_INT[p] for p in y_pred],
                                average="macro",
                                zero_division=0,
                            )
                        ),
                        recall_macro=float(
                            recall_score(
                                [CLASS_TO_INT[y] for y in y_true],
                                [CLASS_TO_INT[p] for p in y_pred],
                                average="macro",
                                zero_division=0,
                            )
                        ),
                        directional_accuracy=bal,
                        n_samples=len(val_rows),
                    )
                )

        # stability + stats
        stability = {
            mid: compute_stability(scores).to_dict()
            for mid, scores in per_model_bal_acc.items()
            if scores
        }
        baseline = "logistic_regression"
        comparisons = []
        raw_ps = []
        for mid, scores in per_model_bal_acc.items():
            if mid == baseline or not scores or not per_model_bal_acc.get(baseline):
                continue
            # compare errors = 1 - bal_acc so lower is better for wilcoxon pairing on loss
            a = [1 - s for s in per_model_bal_acc[baseline][: len(scores)]]
            b = [1 - s for s in scores]
            w = wilcoxon_signed_rank(a, b)
            comparisons.append({"challenger": mid, **w})
            raw_ps.append(w["p_value"])

        adjusted = holm_correction(raw_ps) if raw_ps else []
        for i, adj in enumerate(adjusted):
            comparisons[i]["adjusted_p_value"] = adj["adjusted_p_value"]
            comparisons[i]["significant_adj"] = adj["significant"]

        decision = self._decide(baseline, comparisons, stability)

        # Final test one-way report only (after lock) — never feeds back
        final_report = None
        if final_test_rows and decision.status in ("WIN", "HOLD", "FAIL"):
            self.guard.mark_final_evaluated()
            final_report = {"n": len(final_test_rows), "note": "reporting_only_no_feedback"}

        return {
            "n_folds": len(folds),
            "fold_metrics": [m.to_dict() for m in fold_metrics],
            "stability": stability,
            "comparisons": comparisons,
            "champion_decision": decision.to_dict(),
            "final_test_report": final_report,
            "embargo": self.wf.embargo,
            "max_train_lookback": self.wf.max_train_lookback,
        }

    def _metrics(self, model_id: str, result: Any, val_rows: Sequence[dict], fold_id: int) -> FoldMetrics:
        probs = result.predict_proba_dict(val_rows)
        y_true = [r["label"] for r in val_rows]
        y_pred = [max(p, key=p.get) for p in probs]  # type: ignore
        yt = [CLASS_TO_INT[y] for y in y_true]
        yp = [CLASS_TO_INT[p] for p in y_pred]
        P = np.array([[p[c] for c in CLASS_ORDER] for p in probs])
        try:
            ll = float(log_loss(yt, P, labels=list(range(3))))
        except Exception:
            ll = float("nan")
        # multiclass brier: mean squared error of prob vectors
        Y = np.zeros_like(P)
        for i, t in enumerate(yt):
            Y[i, t] = 1.0
        brier = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
        return FoldMetrics(
            fold_id=fold_id,
            model_id=model_id,
            accuracy=float(accuracy_score(yt, yp)),
            balanced_accuracy=float(balanced_accuracy_score(yt, yp)),
            f1_macro=float(f1_score(yt, yp, average="macro", zero_division=0)),
            log_loss=ll,
            brier=brier,
            precision_macro=float(precision_score(yt, yp, average="macro", zero_division=0)),
            recall_macro=float(recall_score(yt, yp, average="macro", zero_division=0)),
            directional_accuracy=float(balanced_accuracy_score(yt, yp)),
            n_samples=len(val_rows),
        )

    def _decide(
        self,
        baseline: str,
        comparisons: list[dict[str, Any]],
        stability: dict[str, Any],
    ) -> ChampionDecision:
        if not comparisons:
            return ChampionDecision(
                status="HOLD",
                champion=baseline,
                challenger="",
                reason="no_comparisons",
                p_value=float("nan"),
                adjusted_p_value=float("nan"),
                effect_size=float("nan"),
            )
        # pick challenger with best mean stability if significant
        best = None
        for c in comparisons:
            if c.get("significant_adj") and c.get("p_value", 1) == c.get("p_value"):
                if best is None or c.get("effect_size", 0) > best.get("effect_size", 0):
                    best = c
        if best is None:
            return ChampionDecision(
                status="HOLD",
                champion=baseline,
                challenger=comparisons[0]["challenger"],
                reason="insufficient_evidence_p_ge_0.05",
                p_value=float(comparisons[0].get("p_value", float("nan"))),
                adjusted_p_value=float(comparisons[0].get("adjusted_p_value", float("nan"))),
                effect_size=float(comparisons[0].get("effect_size", float("nan"))),
            )
        return ChampionDecision(
            status="WIN",
            champion=best["challenger"],
            challenger=best["challenger"],
            reason="significant_after_holm",
            p_value=float(best["p_value"]),
            adjusted_p_value=float(best["adjusted_p_value"]),
            effect_size=float(best["effect_size"]),
        )


def evaluate_final_out_of_sample(
    guard: FinalTestGuard,
    rows: Sequence[dict[str, Any]],
    model_predict_fn,
) -> dict[str, Any]:
    """
    ONE-WAY final OOS evaluation after model lock.
    Must not feed back into model parameters.
    """
    if not guard.locked:
        raise FinalTestViolation("Final Test not locked")
    guard.mark_final_evaluated()
    # reporting only
    return {"n": len(rows), "status": "reporting_only"}
