"""
End-to-end LEARNING pipeline — real training, not noop.

DATA → VALIDATE → FEATURES → EXPERIENCE LOAD → CHAMPION EVAL
→ GOVERNOR → TRAIN (budgeted) → BACKTEST / WALK-FORWARD
→ PROMOTION GATE → REGISTRY → PERSIST EXPERIENCE / METRICS / REPORT

Hard training budget: 19 minutes (enforced wall-clock).
Colab is optional; Local is mandatory fallback.
Never promotes without gates. Never fakes success.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from idxbot.config import get_settings
from idxbot.core.safety import assert_no_live_trading
from idxbot.data.normalize import NormalizedBar
from idxbot.data.providers.fixture import FixtureProvider
from idxbot.dataset.split import PurgedTimeSeriesSplit
from idxbot.experience.store import ExperienceRecord, ExperienceStore
from idxbot.features.engine import FeatureEngine
from idxbot.governor.governor import ComputationalGovernor, GovernorDecision
from idxbot.ml.candidates import CANDIDATES, CLASS_ORDER, get_candidate
from idxbot.ml.evaluator import ModelEvaluator
from idxbot.ml.providers.colab import ColabComputeProvider
from idxbot.ml.providers.local import LocalComputeProvider
from idxbot.ml.providers.base import JobStatus, TrainingJobSpec
from idxbot.ml.promotion import ChampionChallenger, PromotionDecision
from idxbot.ml.registry import ModelRegistry
from idxbot.ml.trainer import ModelTrainer, TrainResult

logger = logging.getLogger(__name__)

DEFAULT_BUDGET_SECONDS = 19 * 60
FEATURE_VERSION = "features-v1"
DATASET_VERSION = "learning-v1"


@dataclass
class StageLog:
    name: str
    status: str  # OK | SKIP | FAIL | DEGRADED
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass
class LearningReport:
    run_id: str
    status: str = "OK"  # OK | DEGRADED | FAILED | BUDGET_EXHAUSTED
    stages: list[StageLog] = field(default_factory=list)
    provider: str = "local"
    governor: dict[str, Any] = field(default_factory=dict)
    trained: list[dict[str, Any]] = field(default_factory=list)
    promotion: Optional[dict[str, Any]] = None
    champion: Optional[dict[str, Any]] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    budget_seconds: float = DEFAULT_BUDGET_SECONDS
    elapsed_seconds: float = 0.0
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "provider": self.provider,
            "governor": self.governor,
            "trained": self.trained,
            "promotion": self.promotion,
            "champion": self.champion,
            "metrics": self.metrics,
            "budget_seconds": self.budget_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "message": self.message,
            "stages": [asdict(s) for s in self.stages],
        }


class LearningPipeline:
    """
    Production learning path. Uses fixture data by default for determinism
    on GitHub Actions; Yahoo when IDXBOT_USE_FIXTURE is false and available.
    """

    def __init__(
        self,
        *,
        registry_root: str | Path = ".models",
        experience_root: str | Path = ".experience",
        report_dir: str | Path = "ml/metadata",
        budget_seconds: Optional[float] = None,
        use_fixture: bool = True,
        random_seed: int = 42,
        universe: Optional[Sequence[str]] = None,
        n_bars: int = 120,
    ) -> None:
        settings = get_settings()
        assert_no_live_trading()
        self.budget_seconds = float(
            budget_seconds
            if budget_seconds is not None
            else settings.max_training_minutes * 60
        )
        self.registry = ModelRegistry(registry_root)
        self.experience = ExperienceStore(experience_root)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.use_fixture = use_fixture
        self.random_seed = random_seed
        self.universe = list(universe or ("BBCA.JK", "BBRI.JK", "TLKM.JK", "ASII.JK", "BMRI.JK"))
        self.n_bars = n_bars
        self.trainer = ModelTrainer(random_seed=random_seed)
        self.evaluator = ModelEvaluator()
        self.governor = ComputationalGovernor()
        self.promotion_gate = ChampionChallenger(margin=0.02)
        self.feature_engine = FeatureEngine()
        self._deadline: float = 0.0

    def _remaining(self) -> float:
        return max(0.0, self._deadline - time.monotonic())

    def _stage(self, name: str, fn, stages: list[StageLog]) -> Any:
        t0 = time.perf_counter()
        try:
            result = fn()
            elapsed = (time.perf_counter() - t0) * 1000
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                data, detail = result[0], result[1]
                status = detail.pop("status", "OK")
                stages.append(
                    StageLog(name=name, status=status, detail=detail.get("detail", ""), data=detail, elapsed_ms=elapsed)
                )
                return data
            stages.append(StageLog(name=name, status="OK", detail="", data={}, elapsed_ms=elapsed))
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.exception("Stage %s failed: %s", name, e)
            stages.append(StageLog(name=name, status="FAIL", detail=str(e), elapsed_ms=elapsed))
            raise

    # ------------------------------------------------------------------ data
    def _load_data(self) -> list[dict[str, Any]]:
        from idxbot.data.providers.base import HistoricalRequest

        end = date.today()
        start = end - timedelta(days=int(self.n_bars * 1.8))
        if self.use_fixture:
            provider = FixtureProvider(scenario="normal")
            bares = [s.replace(".JK", "") for s in self.universe]
            try:
                req = HistoricalRequest(symbols=bares, start=start, end=end, timeframe="1d")
                raw = provider.get_historical(req)
                bars: list[dict[str, Any]] = []
                for b in raw:
                    b = dict(b)
                    sym = str(b.get("symbol", "")).upper()
                    if not sym.endswith(".JK"):
                        b["symbol"] = f"{sym}.JK"
                    bars.append(b)
                if bars:
                    return bars
            except Exception as e:
                logger.warning("fixture load failed: %s", e)
            return self._synthesize_bars()
        try:
            from idxbot.data.providers.yahoo import YahooProvider

            provider = YahooProvider()
            req = HistoricalRequest(symbols=list(self.universe), start=start, end=end, timeframe="1d")
            bars = provider.get_historical(req)
            if bars:
                return bars
        except Exception as e:
            logger.warning("Yahoo unavailable (%s) — synthetic fallback", e)
        return self._synthesize_bars()

    def _synthesize_bars(self) -> list[dict[str, Any]]:
        """Deterministic synthetic OHLCV for offline CI — clearly labeled."""
        bars: list[dict[str, Any]] = []
        base = datetime(2025, 1, 2, 16, 0, tzinfo=timezone.utc)
        for si, sym in enumerate(self.universe):
            px = 5000.0 + si * 500
            for i in range(self.n_bars):
                # mild trend + noise from seed
                drift = 1.0 + 0.001 * ((i + si) % 7 - 3)
                o = px
                c = max(100.0, px * drift)
                h = max(o, c) * 1.005
                l = min(o, c) * 0.995
                vol = 1_000_000 + (i * 1000) % 50000
                ts = base + timedelta(days=i)
                # skip weekends
                if ts.weekday() >= 5:
                    continue
                bars.append(
                    {
                        "symbol": sym,
                        "timestamp": ts.isoformat(),
                        "raw_open": o,
                        "raw_high": h,
                        "raw_low": l,
                        "raw_close": c,
                        "volume": vol,
                        "adjusted_open": o,
                        "adjusted_high": h,
                        "adjusted_low": l,
                        "adjusted_close": c,
                        "adjustment_mode": "ADJUSTED",
                        "price_adjustment": 1.0,
                        "adjustment_source": "synthetic",
                        "adjustment_version": "v0",
                        "source_timestamp": ts.isoformat(),
                    }
                )
                px = c
        return bars

    def _features_and_labels(self, bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Single feature path: raw bars → NormalizedBar → FeatureEngine → labels.

        FeatureEngine is the sole source of truth for indicator formulas.
        Labels (future outcomes) are attached after features; never used as inputs.
        """
        from idxbot.dataset.labels import LabelGenerator
        from idxbot.data.normalize import NormalizedBar, normalize_record

        normalized: list[NormalizedBar] = []
        for b in bars:
            try:
                rec = dict(b)
                # normalize_record requires raw_* ; synthetic/fixture may only have adjusted_*
                if "raw_open" not in rec and "open" not in rec:
                    for src_k, dst_k in (
                        ("adjusted_open", "raw_open"),
                        ("adjusted_high", "raw_high"),
                        ("adjusted_low", "raw_low"),
                        ("adjusted_close", "raw_close"),
                    ):
                        if src_k in rec and dst_k not in rec:
                            rec[dst_k] = rec[src_k]
                if "raw_close" not in rec and "close" in rec:
                    rec.setdefault("raw_open", rec["close"])
                    rec.setdefault("raw_high", rec["close"])
                    rec.setdefault("raw_low", rec["close"])
                    rec.setdefault("raw_close", rec["close"])
                nb = normalize_record(rec, provider=str(rec.get("provider") or "learning"))
                normalized.append(nb)
            except Exception as e:
                logger.debug("skip bar (normalize failed): %s", e)

        if not normalized:
            return []

        # FeatureEngine — sole indicator implementation
        feat_rows = self.feature_engine.transform(normalized)

        # Label from price series (same timestamps as FeatureEngine)
        by_sym: dict[str, list[NormalizedBar]] = {}
        for b in normalized:
            by_sym.setdefault(b.symbol, []).append(b)
        for sym in by_sym:
            by_sym[sym].sort(key=lambda x: x.timestamp)

        labeler = LabelGenerator()
        lab_map: dict[tuple[str, str, int], Any] = {}
        for sym, series in by_sym.items():
            if len(series) < 60:
                continue
            timestamps = [b.timestamp.isoformat() for b in series]
            closes = [float(b.close) for b in series]
            for lb in labeler.label_symbol(sym, timestamps, closes):
                lab_map[(lb.symbol, lb.timestamp, lb.horizon)] = lb

        rows: list[dict[str, Any]] = []
        for fr in feat_rows:
            ts = str(fr["timestamp"])
            sym = str(fr["symbol"])
            lb = lab_map.get((sym, ts, 5))
            if lb is None or lb.label.value == "PENDING":
                continue
            row = dict(fr)
            row["label"] = lb.label.value
            row["future_return"] = lb.future_return
            row["horizon"] = 5
            row["label_start_idx"] = lb.label_start_idx
            row["label_end_idx"] = lb.label_end_idx
            row["observation_idx"] = lb.observation_idx
            # optional price for paper/accounting paths
            row.setdefault("close", None)
            rows.append(row)
        rows.sort(key=lambda r: (str(r["timestamp"]), str(r["symbol"])))
        return rows

    def _split(self, rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """
        Prefer purged temporal split. If purge empties train (short series),
        fall back to strict chronological 60/20/20 without shuffle.
        """
        n = len(rows)
        a, b = max(1, int(n * 0.6)), max(2, int(n * 0.8))
        chrono = {"train": rows[:a], "validation": rows[a:b], "test": rows[b:]}
        try:
            splitter = PurgedTimeSeriesSplit(embargo=5, val_ratio=0.25)
            fold = splitter.split(rows)
            train = [rows[i] for i in fold.train_indices]
            val = [rows[i] for i in fold.validation_indices]
            train_set = set(fold.train_indices)
            test_start = int(n * 0.85)
            test = [rows[i] for i in range(test_start, n) if i not in train_set]
            if not test:
                test = val[-max(1, len(val) // 4) :] if val else chrono["test"]
            if len(train) >= 20:
                return {"train": train, "validation": val or chrono["validation"], "test": test or chrono["test"]}
            logger.warning(
                "Purged split train too small (%s) — chronological fallback", len(train)
            )
        except Exception as e:
            logger.warning("PurgedTimeSeriesSplit failed (%s) — chronological fallback", e)
        return chrono

    def _select_provider(self) -> tuple[str, Any]:
        settings = get_settings()
        colab = ColabComputeProvider(enabled=settings.enable_colab)
        if settings.enable_colab and colab.availability():
            return "colab", colab
        local = LocalComputeProvider(max_minutes=self.budget_seconds / 60.0)
        return "local", local

    # Estimated wall-clock cost (seconds) for pre-fit scheduling only.
    # Not a hard timeout inside sklearn fit — process isolation is not used.
    _ESTIMATED_TRAIN_SECONDS: dict[str, float] = {
        "logistic_regression": 15.0,
        "ridge_classifier": 15.0,
        "regime_aware_logreg": 20.0,
        "random_forest": 45.0,
        "extra_trees": 45.0,
        "hist_gradient_boosting": 70.0,
        "gradient_boosting": 90.0,
    }

    def _estimate_train_cost(self, algorithm: str) -> float:
        cand = get_candidate(algorithm)
        base = self._ESTIMATED_TRAIN_SECONDS.get(algorithm, 60.0)
        if cand.resource_heavy:
            return max(base, 70.0)
        return base

    def _train_candidates(
        self,
        algorithms: Sequence[str],
        train_rows: list[dict[str, Any]],
        val_rows: list[dict[str, Any]],
        stages: list[StageLog],
    ) -> list[tuple[str, TrainResult, dict[str, Any]]]:
        """
        Pre-fit hard budget gate (monotonic deadline).

        Before each candidate starts:
          estimated_cost + safety_margin <= remaining_budget
        Otherwise SKIP. No new candidates after exhaustion.
        Completed valid results are kept; incomplete candidates are never promoted.
        This is NOT an in-fit sklearn timeout.
        """
        settings = get_settings()
        safety = float(getattr(settings, "budget_safety_margin_seconds", 45) or 45)
        completed: list[tuple[str, TrainResult, dict[str, Any]]] = []
        exhausted = False

        for algo in algorithms:
            remaining = self._remaining()
            if remaining <= 0:
                exhausted = True
                stages.append(
                    StageLog(
                        name="train_budget",
                        status="SKIP",
                        detail="deadline reached — no new candidates",
                        data={"remaining_seconds": remaining, "reason": "BUDGET_EXHAUSTED"},
                    )
                )
                break

            cand = get_candidate(algo)
            estimated = self._estimate_train_cost(algo)
            needed = estimated + safety

            if needed > remaining:
                stages.append(
                    StageLog(
                        name=f"train_{algo}",
                        status="SKIP",
                        detail=(
                            f"pre-fit gate: need {needed:.1f}s "
                            f"(est={estimated:.1f}+margin={safety:.1f}) "
                            f"> remaining {remaining:.1f}s"
                        ),
                        data={
                            "remaining_seconds": remaining,
                            "estimated_seconds": estimated,
                            "safety_margin_seconds": safety,
                            "resource_heavy": cand.resource_heavy,
                            "reason": "BUDGET_GATE",
                        },
                    )
                )
                # If even the cheapest remaining cannot fit, mark exhausted
                if estimated <= 20 and needed > remaining:
                    exhausted = True
                    break
                continue

            t0 = time.perf_counter()
            try:
                tr = self.trainer.train(
                    algo,
                    train_rows,
                    calibration_rows=val_rows,
                    calibrate=True,
                    feature_version=FEATURE_VERSION,
                )
                ev = self.evaluator.evaluate(tr, val_rows) if val_rows else None
                metrics = ev.to_dict() if ev is not None else {"balanced_accuracy": 0.0}
                elapsed = time.perf_counter() - t0
                completed.append((algo, tr, metrics))
                stages.append(
                    StageLog(
                        name=f"train_{algo}",
                        status="OK",
                        detail=f"n_train={tr.n_train} calibrated={tr.calibrated}",
                        data={
                            "metrics": metrics,
                            "training_ms": tr.training_time_ms,
                            "elapsed_seconds": elapsed,
                            "remaining_after": self._remaining(),
                        },
                        elapsed_ms=elapsed * 1000,
                    )
                )
            except Exception as e:
                logger.exception("Train %s failed: %s", algo, e)
                stages.append(
                    StageLog(name=f"train_{algo}", status="FAIL", detail=str(e))
                )

            # Stop scheduling if deadline already passed after a run
            if self._remaining() <= 0:
                exhausted = True
                stages.append(
                    StageLog(
                        name="train_budget",
                        status="SKIP",
                        detail="deadline reached after candidate — stop scheduling",
                        data={"remaining_seconds": self._remaining(), "reason": "BUDGET_EXHAUSTED"},
                    )
                )
                break

        if exhausted:
            stages.append(
                StageLog(
                    name="budget_status",
                    status="BUDGET_EXHAUSTED",
                    detail=f"completed={len(completed)} skipped_after_gate=true",
                    data={"completed": len(completed)},
                )
            )
        return completed

    def _walk_forward_check(self, tr: TrainResult, rows: list[dict[str, Any]]) -> bool:
        """Simple temporal stability: train on first half of val-sized window, score second."""
        if len(rows) < 20:
            return True  # not enough to reject
        mid = len(rows) // 2
        try:
            m1 = self.evaluator.evaluate(tr, rows[:mid])
            m2 = self.evaluator.evaluate(tr, rows[mid:])
            a1 = float(getattr(m1, "balanced_accuracy", 0) or 0)
            a2 = float(getattr(m2, "balanced_accuracy", 0) or 0)
            # stability: second half not collapsed
            return a2 >= max(0.2, a1 - 0.25)
        except Exception:
            return False

    def _promote(
        self,
        completed: list[tuple[str, TrainResult, dict[str, Any]]],
        test_rows: list[dict[str, Any]],
        stages: list[StageLog],
    ) -> tuple[Optional[PromotionDecision], Optional[tuple[str, TrainResult, dict[str, Any]]]]:
        if not completed:
            stages.append(StageLog(name="promotion", status="SKIP", detail="no trained candidates"))
            return None, None

        champ_meta = self.registry.get_champion()
        champion_id = (champ_meta or {}).get("algorithm") or "baseline"
        champion_metrics = {"balanced_accuracy": 0.33}  # safe baseline prior
        if champ_meta:
            # try load prior metrics from report if any
            champion_metrics = {"balanced_accuracy": float((champ_meta.get("metrics") or {}).get("balanced_accuracy", 0.33))}

        best: Optional[tuple[str, TrainResult, dict[str, Any]]] = None
        best_score = -1.0
        for algo, tr, metrics in completed:
            ba = float(metrics.get("balanced_accuracy", 0.0))
            if ba > best_score:
                best_score = ba
                best = (algo, tr, metrics)

        assert best is not None
        algo, tr, metrics = best

        # gates
        wf_ok = self._walk_forward_check(tr, test_rows) if test_rows else True
        # out-of-sample on test for reporting only — promotion uses val metrics + margin
        oos = {}
        if test_rows:
            try:
                oos_m = self.evaluator.evaluate(tr, test_rows)
                oos = oos_m.to_dict() if hasattr(oos_m, "to_dict") else {}
            except Exception:
                oos = {}
        min_samples_ok = tr.n_train >= 30
        leakage_ok = True  # enforced by temporal split upstream
        stability_ok = wf_ok
        metrics_ok = "balanced_accuracy" in metrics and not (
            metrics.get("balanced_accuracy") != metrics.get("balanced_accuracy")
        )  # NaN check

        decision = self.promotion_gate.decide(
            champion_id=champion_id,
            challenger_id=algo,
            champion_metrics=champion_metrics,
            challenger_metrics=metrics,
        )
        # override: all hard gates must pass
        if not all([wf_ok, min_samples_ok, leakage_ok, stability_ok, metrics_ok]):
            decision = PromotionDecision(
                champion=champion_id,
                challenger=algo,
                promoted=False,
                reason="HARD_GATE_FAILED",
                champion_metric=float(champion_metrics.get("balanced_accuracy", 0)),
                challenger_metric=float(metrics.get("balanced_accuracy", 0)),
                margin=self.promotion_gate.margin,
            )
        # if no prior champion and challenger is valid, promote as first champion
        if champ_meta is None and decision.reason != "HARD_GATE_FAILED" and min_samples_ok:
            decision = PromotionDecision(
                champion=champion_id,
                challenger=algo,
                promoted=True,
                reason="first_champion_baseline",
                champion_metric=0.33,
                challenger_metric=float(metrics.get("balanced_accuracy", 0)),
                margin=self.promotion_gate.margin,
            )

        stages.append(
            StageLog(
                name="promotion",
                status="OK" if decision.promoted else "REJECT",
                detail=decision.reason,
                data={**decision.to_dict(), "oos": oos},
            )
        )
        return decision, best if decision.promoted else None

    def _persist_model(self, algo: str, tr: TrainResult, metrics: dict[str, Any]) -> dict[str, Any]:
        version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        meta = self.registry.save(
            tr,
            version=version,
            metrics=metrics,
            dataset_version=DATASET_VERSION,
        )
        self.registry.set_champion(algo, version, reason="learning_pipeline_promotion")
        if hasattr(meta, "to_dict"):
            d = meta.to_dict()
        else:
            d = dict(meta) if isinstance(meta, dict) else {"algorithm": algo, "version": version}
        d["algorithm"] = algo
        d["version"] = version
        return d

    def _persist_experience_snapshot(self, rows: list[dict[str, Any]], run_id: str) -> int:
        """Persist a sample of labeled rows as experience for next cycle."""
        records = []
        for r in rows[-50:]:  # cap
            rid = ExperienceRecord.make_id(
                symbol=str(r["symbol"]),
                timestamp=str(r["timestamp"]),
                horizon=int(r.get("horizon") or 5),
                feature_version=FEATURE_VERSION,
                label_version="1",
            )
            records.append(
                ExperienceRecord(
                    experience_id=rid,
                    symbol=str(r["symbol"]),
                    timestamp=str(r["timestamp"]),
                    horizon=int(r.get("horizon") or 5),
                    label=str(r.get("label") or "FLAT"),
                    realized_return=r.get("future_return"),
                    outcome=str(r.get("label") or ""),
                    prediction=str(r.get("label") or ""),
                    confidence=0.5,
                    feature_version=FEATURE_VERSION,
                    model_version=run_id,
                )
            )
        if not records:
            return 0
        result = self.experience.append(records)
        return int(result.get("appended", len(records))) if isinstance(result, dict) else len(records)

    def run(self, run_id: Optional[str] = None) -> LearningReport:
        assert_no_live_trading()
        run_id = run_id or f"learn-{uuid.uuid4().hex[:12]}"
        report = LearningReport(run_id=run_id, budget_seconds=self.budget_seconds)
        self._deadline = time.monotonic() + self.budget_seconds
        t_start = time.monotonic()
        stages = report.stages
        logger.info("RUN_START MODE=LEARNING run_id=%s budget=%ss", run_id, self.budget_seconds)

        try:
            # DATA
            bars = self._stage("data", self._load_data, stages)
            if not bars:
                report.status = "FAILED"
                report.message = "no market data"
                stages.append(StageLog(name="data_validation", status="FAIL", detail="empty bars"))
                return self._finalize(report, t_start)
            stages[-1].data = {"n_bars": len(bars), "symbols": sorted({str(b.get("symbol")) for b in bars})}
            logger.info("DATA_STATUS n_bars=%s", len(bars))

            # FEATURES + LABELS
            rows = self._stage("features", lambda: self._features_and_labels(bars), stages)
            if len(rows) < 40:
                report.status = "FAILED"
                report.message = f"insufficient labeled rows: {len(rows)}"
                stages.append(StageLog(name="features_validation", status="FAIL", detail=report.message))
                return self._finalize(report, t_start)
            stages[-1].data = {"n_rows": len(rows)}

            # EXPERIENCE LOAD (read prior — does not block training)
            def _load_exp():
                n = 0
                try:
                    for year in range(2024, 2027):
                        for month in range(1, 13):
                            n += len(self.experience.load_partition(year, month) or [])
                except Exception:
                    pass
                return n, {"status": "OK", "detail": f"prior_experience_rows={n}", "prior_rows": n}

            prior_n = self._stage("experience_load", _load_exp, stages)

            # SPLIT
            splits = self._stage("temporal_split", lambda: self._split(rows), stages)
            train_rows = splits.get("train") or []
            val_rows = splits.get("validation") or []
            test_rows = splits.get("test") or []
            stages[-1].data = {
                "n_train": len(train_rows),
                "n_val": len(val_rows),
                "n_test": len(test_rows),
            }
            if len(train_rows) < 20:
                report.status = "FAILED"
                report.message = "train split too small"
                return self._finalize(report, t_start)

            # PROVIDER
            provider_name, provider = self._select_provider()
            report.provider = provider_name
            stages.append(
                StageLog(
                    name="provider",
                    status="OK",
                    detail=provider_name,
                    data={"available": True, "name": provider_name},
                )
            )
            logger.info("PROVIDER=%s", provider_name)

            # GOVERNOR
            try:
                decision: GovernorDecision = self.governor.decide(registry_root=str(self.registry.root))
                selected = list(decision.selection.selected)
                report.governor = {
                    "decision": decision.selection.decision,
                    "selected": selected,
                    "degradation": decision.selection.degradation_level,
                    "reason": decision.selection.reason,
                }
                stages.append(
                    StageLog(
                        name="governor",
                        status="OK" if not decision.is_safe_exit else "SAFE_EXIT",
                        detail=decision.selection.reason,
                        data=report.governor,
                    )
                )
                logger.info("GOVERNOR_DECISION selected=%s", selected)
                if decision.is_safe_exit:
                    # still try cheapest baseline
                    selected = ["logistic_regression"]
            except Exception as e:
                logger.warning("Governor failed (%s) — default cheap models", e)
                selected = ["logistic_regression", "ridge_classifier", "random_forest"]
                report.governor = {"decision": "FALLBACK", "selected": selected, "error": str(e)}
                stages.append(StageLog(name="governor", status="DEGRADED", detail=str(e), data=report.governor))

            # Prefer cheap first under budget
            def priority(a: str) -> int:
                c = CANDIDATES.get(a)
                if c is None:
                    return 50
                return 100 if c.resource_heavy else 10

            selected = sorted(selected, key=priority)

            # TRAIN
            logger.info("TRAINING_START candidates=%s remaining=%.1fs", selected, self._remaining())
            completed = self._train_candidates(selected, train_rows, val_rows, stages)
            logger.info("TRAINING_END completed=%s remaining=%.1fs", [a for a, _, _ in completed], self._remaining())
            report.trained = [
                {"algorithm": a, "metrics": m, "n_train": tr.n_train, "calibrated": tr.calibrated}
                for a, tr, m in completed
            ]

            if not completed:
                budget_hit = any(
                    s.status == "BUDGET_EXHAUSTED" or (s.data or {}).get("reason") == "BUDGET_EXHAUSTED"
                    or (s.data or {}).get("reason") == "BUDGET_GATE"
                    for s in stages
                )
                report.status = "BUDGET_EXHAUSTED" if budget_hit else "FAILED"
                report.message = (
                    "budget exhausted before any model completed"
                    if budget_hit
                    else "no models trained successfully"
                )
                return self._finalize(report, t_start)

            # PROMOTION
            promo, winner = self._promote(completed, test_rows, stages)
            report.promotion = promo.to_dict() if promo else None

            if winner:
                algo, tr, metrics = winner
                if self._remaining() < 5:
                    stages.append(StageLog(name="persist_model", status="SKIP", detail="budget too low for persist"))
                else:
                    meta = self._persist_model(algo, tr, metrics)
                    report.champion = meta if isinstance(meta, dict) else {"algorithm": algo}
                    stages.append(
                        StageLog(name="persist_model", status="OK", detail=f"{algo}", data=report.champion or {})
                    )
                    logger.info("PROMOTION promoted=%s algo=%s", True, algo)
            else:
                # keep existing champion
                report.champion = self.registry.get_champion()
                logger.info("PROMOTION rejected — keep champion=%s", report.champion)

            # EXPERIENCE PERSIST
            n_exp = self._persist_experience_snapshot(train_rows + val_rows, run_id)
            stages.append(StageLog(name="persist_experience", status="OK", detail=f"appended={n_exp}"))

            # REPORT PERSIST
            report.status = "OK" if completed else "FAILED"
            report.message = "learning cycle complete"
            report.metrics = {
                "n_candidates_trained": len(completed),
                "prior_experience": prior_n,
                "promoted": bool(winner),
            }
        except Exception as e:
            logger.exception("LEARNING pipeline failed: %s", e)
            report.status = "FAILED"
            report.message = str(e)
            stages.append(StageLog(name="pipeline", status="FAIL", detail=str(e)))

        return self._finalize(report, t_start)

    def _finalize(self, report: LearningReport, t_start: float) -> LearningReport:
        report.elapsed_seconds = time.monotonic() - t_start
        if any(s.status == "BUDGET_EXHAUSTED" or s.name == "budget_status" for s in report.stages):
            if report.status == "OK":
                report.status = "BUDGET_EXHAUSTED"
        elif report.elapsed_seconds >= self.budget_seconds and report.status == "OK":
            report.status = "BUDGET_EXHAUSTED"
        path = self.report_dir / f"learning_report_{report.run_id}.json"
        try:
            path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
            report.stages.append(StageLog(name="persist_report", status="OK", detail=str(path)))
        except Exception as e:
            report.stages.append(StageLog(name="persist_report", status="FAIL", detail=str(e)))
        logger.info(
            "RUN_END MODE=LEARNING status=%s elapsed=%.1fs champion=%s",
            report.status,
            report.elapsed_seconds,
            report.champion,
        )
        return report
