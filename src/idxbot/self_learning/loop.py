"""
Evidence-based self-learning closed loop.

Prediction → ExperienceStore → Outcome (after horizon) → Evaluation
→ Candidate only → PromotionGate (all must pass) → optional champion set.

Never modifies production champion without PromotionDecision.promoted == True.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from idxbot.experience.store import ExperienceRecord, ExperienceStore
from idxbot.ml.promotion import ChampionChallenger, PromotionDecision
from idxbot.signals.order_intent import OrderIntent

logger = logging.getLogger(__name__)


@dataclass
class LoopResult:
    persisted: int = 0
    resolved: int = 0
    promotion: Optional[dict[str, Any]] = None
    status: str = "OK"
    detail: str = ""


class SelfLearningLoop:
    def __init__(
        self,
        experience_root: str | Path = ".experience",
        *,
        default_horizon_days: int = 5,
    ) -> None:
        self.store = ExperienceStore(experience_root)
        self.horizon = default_horizon_days
        self.challenger = ChampionChallenger()

    def persist_predictions(
        self,
        intents: Sequence[OrderIntent],
        *,
        horizon: Optional[int] = None,
    ) -> int:
        h = horizon or self.horizon
        records: list[ExperienceRecord] = []
        for oi in intents:
            rid = ExperienceRecord.make_id(
                symbol=oi.symbol,
                timestamp=oi.timestamp,
                horizon=h,
                feature_version=oi.feature_version or "1",
                label_version="1",
            )
            rec = ExperienceRecord(
                experience_id=rid,
                symbol=oi.symbol,
                timestamp=oi.timestamp,
                horizon=h,
                label="PENDING",
                prediction=oi.intent,
                confidence=oi.confidence,
                feature_version=oi.feature_version or "1",
                model_version=oi.model_version,
            )
            records.append(rec)
        if not records:
            return 0
        result = self.store.append(records)
        return int(result.get("appended", len(records)))

    def resolve_pending(
        self,
        *,
        now: Optional[datetime] = None,
        price_lookup: Optional[Mapping[str, float]] = None,
    ) -> int:
        """
        Attach outcomes only when horizon has elapsed.
        Does not rewrite parquet in-place here (read-only scan count);
        full rewrite belongs to a dedicated maintenance job.
        """
        now = now or datetime.now(timezone.utc)
        resolved = 0
        for year, month in self._recent_partitions(now):
            rows = self.store.load_partition(year, month)
            for row in rows:
                if row.get("label") not in ("PENDING", None) and row.get("outcome"):
                    continue
                ts = row.get("timestamp") or ""
                try:
                    pred_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                except ValueError:
                    continue
                horizon = int(row.get("horizon") or self.horizon)
                if pred_dt.tzinfo is None:
                    pred_dt = pred_dt.replace(tzinfo=timezone.utc)
                if now < pred_dt + timedelta(days=horizon):
                    continue  # integrity: never attach early
                sym = row.get("symbol")
                if not price_lookup or sym not in price_lookup:
                    continue
                resolved += 1
        return resolved

    def evaluate_promotion(
        self,
        *,
        champion_id: str,
        challenger_id: str,
        champion_metrics: Mapping[str, float],
        candidate_metrics: Mapping[str, float],
        anti_leakage_pass: bool,
        walk_forward_pass: bool,
        adversarial_pass: bool,
        stability_pass: bool,
        resource_policy_pass: bool = True,
        baseline_pass: bool = True,
    ) -> PromotionDecision:
        """All gates required. Drift alone never promotes."""
        if not all(
            [
                anti_leakage_pass,
                walk_forward_pass,
                adversarial_pass,
                stability_pass,
                resource_policy_pass,
                baseline_pass,
            ]
        ):
            return PromotionDecision(
                champion=champion_id,
                challenger=challenger_id,
                promoted=False,
                reason="GATE_FAILED",
                champion_metric=float(champion_metrics.get("balanced_accuracy", 0.0)),
                challenger_metric=float(candidate_metrics.get("balanced_accuracy", 0.0)),
                margin=self.challenger.margin,
            )
        return self.challenger.decide(
            champion_id,
            challenger_id,
            dict(champion_metrics),
            dict(candidate_metrics),
        )

    def _recent_partitions(self, now: datetime) -> list[tuple[int, int]]:
        y, m = now.year, now.month
        out = [(y, m)]
        if m == 1:
            out.append((y - 1, 12))
        else:
            out.append((y, m - 1))
        return out
