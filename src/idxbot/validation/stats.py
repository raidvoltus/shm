"""Statistical comparison utilities — Wilcoxon, Holm correction, stability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from scipy.stats import wilcoxon


def wilcoxon_signed_rank(
    a: Sequence[float], b: Sequence[float], *, alpha: float = 0.05
) -> dict[str, Any]:
    """
    Paired Wilcoxon signed-rank test on fold metrics (e.g. log_loss or errors).
    H0: distribution of differences is symmetric about zero.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if len(x) != len(y) or len(x) < 3:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "sample_count": int(len(x)),
            "effect_size": float("nan"),
            "significant": False,
            "alpha": alpha,
        }
    diff = x - y
    # zero differences dropped by wilcoxon by default in recent scipy
    try:
        stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "sample_count": int(len(x)),
            "effect_size": float("nan"),
            "significant": False,
            "alpha": alpha,
        }
    # rank-biserial effect size approximation
    n = len(diff)
    effect = 1 - (2 * float(stat)) / (n * (n + 1) / 2) if n > 0 else float("nan")
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "sample_count": n,
        "effect_size": float(effect),
        "significant": bool(p < alpha),
        "alpha": alpha,
    }


def holm_correction(p_values: Sequence[float], alpha: float = 0.05) -> list[dict[str, Any]]:
    """Holm step-down adjusted p-values."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda t: t[1] if t[1] == t[1] else 1.0)
    adjusted = [1.0] * m
    for rank, (orig_i, p) in enumerate(indexed):
        if p != p:  # nan
            adjusted[orig_i] = float("nan")
            continue
        adj = min(1.0, (m - rank) * p)
        adjusted[orig_i] = adj
    # enforce monotonicity
    sorted_adj = [adjusted[i] for i, _ in indexed]
    for i in range(1, len(sorted_adj)):
        if sorted_adj[i] == sorted_adj[i]:
            sorted_adj[i] = max(sorted_adj[i], sorted_adj[i - 1])
    for rank, (orig_i, _) in enumerate(indexed):
        if adjusted[orig_i] == adjusted[orig_i]:
            adjusted[orig_i] = sorted_adj[rank]
    return [
        {
            "raw_p_value": float(p_values[i]) if p_values[i] == p_values[i] else float("nan"),
            "adjusted_p_value": adjusted[i],
            "significant": bool(adjusted[i] < alpha) if adjusted[i] == adjusted[i] else False,
        }
        for i in range(m)
    ]


@dataclass
class StabilityReport:
    mean: float
    median: float
    std: float
    min: float
    max: float
    worst_fold: int
    best_fold: int
    n_folds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "worst_fold": self.worst_fold,
            "best_fold": self.best_fold,
            "n_folds": self.n_folds,
        }


def compute_stability(fold_scores: Sequence[float]) -> StabilityReport:
    arr = np.asarray(fold_scores, dtype=np.float64)
    if len(arr) == 0:
        return StabilityReport(
            mean=float("nan"),
            median=float("nan"),
            std=float("nan"),
            min=float("nan"),
            max=float("nan"),
            worst_fold=-1,
            best_fold=-1,
            n_folds=0,
        )
    return StabilityReport(
        mean=float(np.nanmean(arr)),
        median=float(np.nanmedian(arr)),
        std=float(np.nanstd(arr)),
        min=float(np.nanmin(arr)),
        max=float(np.nanmax(arr)),
        worst_fold=int(np.nanargmin(arr)),
        best_fold=int(np.nanargmax(arr)),
        n_folds=len(arr),
    )
