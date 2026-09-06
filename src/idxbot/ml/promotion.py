"""
Champion / Challenger with Margin of Equivalency.

Default minimum improvement: 2% on primary metric (balanced_accuracy).
TEST set must NOT drive promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PromotionDecision:
    champion: str
    challenger: str
    promoted: bool
    reason: str
    champion_metric: float
    challenger_metric: float
    margin: float
    primary_metric: str = "balanced_accuracy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "champion": self.champion,
            "challenger": self.challenger,
            "promoted": self.promoted,
            "reason": self.reason,
            "champion_metric": self.champion_metric,
            "challenger_metric": self.challenger_metric,
            "margin": self.margin,
            "primary_metric": self.primary_metric,
        }


class ChampionChallenger:
    def __init__(self, margin: float = 0.02, primary_metric: str = "balanced_accuracy") -> None:
        self.margin = margin
        self.primary_metric = primary_metric

    def decide(
        self,
        champion_id: str,
        challenger_id: str,
        champion_metrics: dict[str, Any],
        challenger_metrics: dict[str, Any],
    ) -> PromotionDecision:
        c_val = float(champion_metrics.get(self.primary_metric, 0.0))
        h_val = float(challenger_metrics.get(self.primary_metric, 0.0))
        improvement = h_val - c_val
        if improvement >= self.margin:
            return PromotionDecision(
                champion=champion_id,
                challenger=challenger_id,
                promoted=True,
                reason="improvement_meets_margin",
                champion_metric=c_val,
                challenger_metric=h_val,
                margin=self.margin,
                primary_metric=self.primary_metric,
            )
        return PromotionDecision(
            champion=champion_id,
            challenger=challenger_id,
            promoted=False,
            reason="improvement_below_margin",
            champion_metric=c_val,
            challenger_metric=h_val,
            margin=self.margin,
            primary_metric=self.primary_metric,
        )
