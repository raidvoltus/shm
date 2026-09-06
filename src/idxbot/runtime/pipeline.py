"""
Explicit sequential autonomous pipeline.

Every stage returns a typed StageResult. No magic orchestration.
LIVE_TRADING remains false; output is OrderIntent only.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from idxbot.config import get_settings
from idxbot.core.safety import assert_no_live_trading
from idxbot.data.providers.registry import (
    ProviderRegistry,
    ProviderResult,
    ProviderStatus,
    build_default_registry,
)
from idxbot.data.quality.engine import QualityStatus
from idxbot.governor.governor import ComputationalGovernor, GovernorDecision
from idxbot.portfolio.governor import PortfolioGovernor, PortfolioDecision
from idxbot.signals.order_intent import OrderIntent, create_order_intent
from idxbot.signals.signal_engine import SignalEngine
from idxbot.telegram.notifier import TelegramNotifier
from idxbot.ml.inference import ChampionInferencer
from idxbot.self_learning.loop import SelfLearningLoop

logger = logging.getLogger(__name__)
JAKARTA = ZoneInfo("Asia/Jakarta")

# Default liquid universe (small for GH Actions Free).
# Fixture provider uses bare tickers (BBCA); production may use BBCA.JK.
DEFAULT_UNIVERSE = ("BBCA.JK", "BBRI.JK", "TLKM.JK", "ASII.JK", "BMRI.JK")


@dataclass
class StageResult:
    name: str
    status: str  # OK | SKIP | FAIL | DEGRADED
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    run_id: str
    status: str  # OK | DEGRADED | FAILED | SAFE_EXIT
    stages: list[StageResult] = field(default_factory=list)
    intents: list[OrderIntent] = field(default_factory=list)
    governor_state: str = "UNKNOWN"
    persistence_mode: str = "EPHEMERAL"
    health: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "governor_state": self.governor_state,
            "persistence_mode": self.persistence_mode,
            "message": self.message,
            "intents": [i.to_dict() for i in self.intents],
            "stages": [
                {"name": s.name, "status": s.status, "detail": s.detail, "data": s.data}
                for s in self.stages
            ],
            "health": self.health,
        }


class AutonomousPipeline:
    """
    Deterministic sequential stages for one autonomous run.
    """

    def __init__(
        self,
        *,
        provider_registry: Optional[ProviderRegistry] = None,
        governor: Optional[ComputationalGovernor] = None,
        portfolio_governor: Optional[PortfolioGovernor] = None,
        signal_engine: Optional[SignalEngine] = None,
        notifier: Optional[TelegramNotifier] = None,
        allow_fixture: bool = False,
        universe: Sequence[str] = DEFAULT_UNIVERSE,
        feature_version: str = "features-v1",
        model_version: str = "model-v1",
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.allow_fixture = allow_fixture or os.environ.get("IDXBOT_USE_FIXTURE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        self.registry = provider_registry or build_default_registry(allow_fixture=self.allow_fixture)
        self.governor = governor or ComputationalGovernor()
        self.portfolio_gov = portfolio_governor or PortfolioGovernor()
        self.signal_engine = signal_engine or SignalEngine(
            feature_version=feature_version,
            model_version=model_version,
            timezone=settings.timezone,
        )
        self.notifier = notifier or TelegramNotifier(
            enabled=settings.telegram_enabled,
            token=settings.telegram_bot_token or None,
            chat_id=settings.telegram_chat_id or None,
        )
        self.universe = list(universe)
        self.feature_version = feature_version
        self.model_version = model_version
        self._previous_intents: dict[str, str] = {}  # symbol -> intent (ephemeral unless loaded)
        self.registry_root = os.environ.get("IDXBOT_MODEL_REGISTRY", ".models")
        self.allow_momentum_fallback = os.environ.get(
            "IDXBOT_ALLOW_MOMENTUM_FALLBACK", ""
        ).lower() in ("1", "true", "yes")
        self.inferencer = ChampionInferencer(
            self.registry_root,
            expected_feature_version=feature_version,
        )
        self.learning = SelfLearningLoop(os.environ.get('IDXBOT_EXPERIENCE_ROOT', '.experience'))

    def load_previous_intents(self, state: Mapping[str, str]) -> None:
        self._previous_intents = {k.upper(): v for k, v in state.items()}

    def previous_intents_snapshot(self) -> dict[str, str]:
        return dict(self._previous_intents)

    def run(
        self,
        *,
        run_id: str,
        scheduled_at: datetime,
        account: Mapping[str, Any],
        dry_run: bool = False,
        resource_inject: Optional[dict] = None,
    ) -> PipelineResult:
        assert_no_live_trading()
        stages: list[StageResult] = []
        intents: list[OrderIntent] = []
        persistence_mode = "EPHEMERAL"  # honest default for GH Actions

        # --- Stage: Governor ---
        gdec = self.governor.decide(resource_inject=resource_inject)
        gov_state = self._governor_state_label(gdec)
        stages.append(
            StageResult(
                "governor",
                "OK",
                gov_state,
                {
                    "decision": gdec.selection.decision,
                    "target_count": gdec.selection.target_count,
                    "selected": list(gdec.selection.selected),
                    "audit": gdec.audit,
                },
            )
        )

        if gdec.is_safe_exit or gdec.selection.decision == "SAFE_EXIT":
            stages.append(StageResult("ml", "SKIP", "SAFE_EXIT — no ML"))
            # still emit HOLD intents for observability
            for sym in self.universe:
                oi = self.signal_engine.generate(
                    symbol=sym,
                    timestamp=scheduled_at,
                    probability=0.5,
                    governor_state="SAFE_EXIT",
                    active_models=0,
                    portfolio_allowed=True,
                    previous_intent=self._previous_intents.get(sym, "HOLD"),
                    run_id=run_id,
                )
                intents.append(oi)
                self._previous_intents[sym] = oi.intent
            result = PipelineResult(
                run_id=run_id,
                status="SAFE_EXIT",
                stages=stages,
                intents=intents,
                governor_state="SAFE_EXIT",
                persistence_mode=persistence_mode,
                message="Governor SAFE_EXIT",
            )
            if not dry_run:
                self._notify(intents, stages)
            return result

        # --- Stage: Fetch market data ---
        end = scheduled_at.date() if hasattr(scheduled_at, "date") else date.today()
        start = end - timedelta(days=120)
        symbol_data: dict[str, ProviderResult] = {}
        fetch_ok = 0
        universe = list(self.universe)
        # Fixture series uses bare tickers (BBCA); try both forms
        for sym in universe:
            pr = self.registry.fetch_historical(sym, start, end)
            if not pr.ok and sym.endswith(".JK"):
                bare = sym[:-3]
                pr2 = self.registry.fetch_historical(bare, start, end)
                if pr2.ok:
                    # re-canonical symbol in rows
                    for row in pr2.data:
                        row["symbol"] = sym
                    pr = pr2
            symbol_data[sym] = pr
            if pr.ok:
                fetch_ok += 1
            else:
                stages.append(
                    StageResult(
                        "data_fetch",
                        "FAIL",
                        f"{sym}: {pr.status.value} {pr.error}",
                        {"symbol": sym, "status": pr.status.value},
                    )
                )

        if fetch_ok == 0:
            stages.append(StageResult("data_fetch", "FAIL", "entire universe failed"))
            return PipelineResult(
                run_id=run_id,
                status="FAILED",
                stages=stages,
                intents=[],
                governor_state=gov_state,
                persistence_mode=persistence_mode,
                message="No market data",
                health={"data": "FAILED", "status": "FAILED"},
            )

        data_status = "DEGRADED" if any(p.status == ProviderStatus.FIXTURE for p in symbol_data.values()) else "OK"
        stages.append(
            StageResult(
                "data_fetch",
                data_status,
                f"{fetch_ok}/{len(self.universe)} symbols",
                {"providers": {s: p.provider_name for s, p in symbol_data.items() if p.ok}},
            )
        )

        # --- Stage: Quality + simple feature signal (momentum proxy when no trained model) ---
        # Full FeatureEngine + sklearn inference is available but heavy; we use deterministic
        # multi-horizon momentum probabilities from adjusted closes when models not loaded.
        # This keeps pipeline executable without registry artifacts while remaining causal.
        active_n = self._active_model_count(gdec)
        for sym, pr in symbol_data.items():
            if not pr.ok:
                continue
            q = self._quality_check(pr.data)
            if q != "VALID":
                stages.append(StageResult("quality", "FAIL", f"{sym} {q}", {"symbol": sym}))
                continue

            prev = self._previous_intents.get(sym.upper(), "HOLD")
            last_close = float(pr.data[-1].get("close") or pr.data[-1].get("adjusted_close") or 0)

            # --- ML path: champion only; no disguised momentum as production ML ---
            feature_row = self._simple_feature_row(pr.data)
            inf = self.inferencer.predict_row(feature_row)
            reasons: list[str] = []
            model_ver = self.model_version
            feat_ver = self.feature_version
            n_models = active_n

            if inf.status == "OK":
                primary_p = inf.probability_up
                probs = {"1D": primary_p, "5D": primary_p, "20D": primary_p}
                model_ver = inf.model_version or model_ver
                feat_ver = inf.feature_version or feat_ver
                n_models = max(1, inf.active_models)
                reasons.append("CHAMPION_INFERENCE")
            elif self.allow_momentum_fallback:
                probs = self._momentum_horizons(pr.data)
                primary_p = probs.get("5D", 0.5)
                reasons.append("MOMENTUM_FALLBACK_EXPLICIT")
            else:
                # Safe default: HOLD when no validated champion
                probs = {"1D": 0.5, "5D": 0.5, "20D": 0.5}
                primary_p = 0.5
                reasons.append(f"MODEL_UNAVAILABLE:{inf.status}")
                provisional = self.signal_engine.generate(
                    symbol=sym,
                    timestamp=scheduled_at,
                    probability=0.5,
                    horizons=probs,
                    previous_intent=prev,
                    governor_state=gov_state,
                    active_models=0,
                    portfolio_allowed=True,
                    run_id=run_id,
                    extra_reasons=reasons,
                )
                # force HOLD
                oi = create_order_intent(
                    symbol=sym,
                    timestamp=scheduled_at,
                    intent="HOLD",
                    confidence=0.5,
                    governor_state=gov_state,
                    feature_version=feat_ver,
                    model_version=model_ver,
                    active_models=0,
                    portfolio_allowed=True,
                    reason_codes=tuple(reasons) + ("HOLD_NO_CHAMPION",),
                    horizons=probs,
                    run_id=run_id,
                )
                intents.append(oi)
                self._previous_intents[sym.upper()] = "HOLD"
                stages.append(
                    StageResult(
                        "signal",
                        "DEGRADED",
                        f"{sym} HOLD (no champion)",
                        {"symbol": sym, "intent": "HOLD", "ml_status": inf.status},
                    )
                )
                continue

            provisional = self.signal_engine.generate(
                symbol=sym,
                timestamp=scheduled_at,
                probability=primary_p,
                horizons=probs,
                previous_intent=prev,
                governor_state=gov_state,
                active_models=n_models,
                portfolio_allowed=True,
                run_id=run_id,
                extra_reasons=reasons,
            )

            pdec: PortfolioDecision = self.portfolio_gov.evaluate(
                intent=provisional.intent,
                symbol=sym,
                confidence=provisional.confidence,
                price=last_close,
                account=account,
            )

            final_intent = pdec.final_intent
            reasons = list(provisional.reason_codes) + list(pdec.reason_codes)
            if not pdec.allowed and provisional.intent == "BUY":
                final_intent = "HOLD"

            oi = create_order_intent(
                symbol=sym,
                timestamp=scheduled_at,
                intent=final_intent,  # type: ignore[arg-type]
                confidence=provisional.confidence,
                governor_state=gov_state,
                feature_version=feat_ver,
                model_version=model_ver,
                active_models=n_models,
                portfolio_allowed=pdec.allowed,
                reason_codes=reasons,
                horizons=probs,
                run_id=run_id,
            )
            intents.append(oi)
            self._previous_intents[sym.upper()] = oi.intent
            stages.append(
                StageResult(
                    "signal",
                    "OK",
                    f"{sym} {oi.intent}",
                    {"symbol": sym, "intent": oi.intent, "confidence": oi.confidence, "ml": inf.status},
                )
            )

        stages.append(StageResult("ensemble", "OK" if active_n else "DEGRADED", f"active_models={active_n}"))
        stages.append(StageResult("portfolio_governor", "OK", f"intents={len(intents)}"))

        # Experience store (predictions only; outcomes resolved later)
        if not dry_run and intents:
            try:
                n_exp = self.learning.persist_predictions(intents)
                stages.append(StageResult("experience", "OK", f"persisted={n_exp}"))
            except Exception as e:
                stages.append(StageResult("experience", "FAIL", type(e).__name__))

        # Telegram (after intents + experience)
        if not dry_run:
            tg_ok = self._notify(intents, stages)
            stages.append(StageResult("telegram", "OK" if tg_ok else "DEGRADED", "sent" if tg_ok else "failed_or_skipped"))

        overall = "OK"
        if data_status == "DEGRADED":
            overall = "DEGRADED"
        if not intents:
            overall = "FAILED"

        return PipelineResult(
            run_id=run_id,
            status=overall,
            stages=stages,
            intents=intents,
            governor_state=gov_state,
            persistence_mode=persistence_mode,
            message=f"produced {len(intents)} intents",
            health={
                "status": overall,
                "data": data_status,
                "governor": gov_state,
                "telegram": "ok" if self.notifier.enabled else "disabled",
                "persistence": persistence_mode,
            },
        )

    def _governor_state_label(self, gdec: GovernorDecision) -> str:
        n = gdec.selection.target_count
        if gdec.selection.decision == "SAFE_EXIT" or n <= 0:
            return "SAFE_EXIT"
        return {7: "FULL_7", 5: "DEGRADED_5", 3: "DEGRADED_3", 1: "DEGRADED_1"}.get(n, f"N{n}")

    def _active_model_count(self, gdec: GovernorDecision) -> int:
        if gdec.selection.decision == "SAFE_EXIT":
            return 0
        return int(gdec.selection.target_count or len(gdec.selection.selected))

    def _quality_check(self, rows: list[dict[str, Any]]) -> str:
        if len(rows) < 30:
            return "INSUFFICIENT_DATA"
        seen = set()
        prev_ts = None
        for r in rows:
            ts = r.get("timestamp")
            if ts in seen:
                return "DUPLICATE_TIMESTAMP"
            seen.add(ts)
            o = float(r.get("open") or 0)
            h = float(r.get("high") or 0)
            l = float(r.get("low") or 0)
            c = float(r.get("close") or r.get("adjusted_close") or 0)
            v = float(r.get("volume") or 0)
            if min(o, h, l, c) <= 0:
                return "NEGATIVE_PRICE"
            if v < 0:
                return "NEGATIVE_VOLUME"
            if h < l or h < max(o, c) or l > min(o, c):
                return "INVALID_OHLC"
            if prev_ts is not None and str(ts) < str(prev_ts):
                return "NON_MONOTONIC"
            prev_ts = ts
        return "VALID"


    def _simple_feature_row(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Causal lightweight features for champion inference (no lookahead)."""
        closes = []
        volumes = []
        for r in rows:
            c = r.get("adjusted_close", r.get("close"))
            if c is not None:
                closes.append(float(c))
            volumes.append(float(r.get("volume") or 0))
        row: dict[str, Any] = {}
        if len(closes) < 25:
            return row
        def ret(n: int) -> float:
            if closes[-n-1] == 0:
                return 0.0
            return (closes[-1] / closes[-n-1]) - 1.0
        row["return_1d"] = ret(1)
        row["return_5d"] = ret(5)
        row["return_20d"] = ret(20)
        import statistics
        window = closes[-20:]
        mean = statistics.fmean(window)
        row["sma_20"] = mean
        row["price_vs_sma20"] = (closes[-1] / mean - 1.0) if mean else 0.0
        # rsi-ish
        gains = []
        losses = []
        for i in range(1, min(15, len(closes))):
            d = closes[-i] - closes[-i-1]
            gains.append(max(d, 0.0))
            losses.append(max(-d, 0.0))
        ag = statistics.fmean(gains) if gains else 0.0
        al = statistics.fmean(losses) if losses else 1e-9
        rs = ag / al if al else 0.0
        row["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
        row["rolling_volatility_20"] = statistics.pstdev([ret(i) for i in range(1, 21)]) if len(closes) > 21 else 0.0
        row["volume_ratio_20"] = (volumes[-1] / statistics.fmean(volumes[-20:])) if statistics.fmean(volumes[-20:]) else 1.0
        return row

    def _momentum_horizons(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        """
        Causal multi-horizon probability proxy from adjusted closes.
        P(up) ≈ sigmoid of forward-looking-free past return.
        Deterministic; no future bars.
        """
        closes = []
        for r in rows:
            c = r.get("adjusted_close", r.get("close"))
            if c is not None:
                closes.append(float(c))
        if len(closes) < 25:
            return {"1D": 0.5, "5D": 0.5, "20D": 0.5}

        def _p(ret: float) -> float:
            # squash to (0,1)
            x = max(-3.0, min(3.0, ret * 10))
            return 1.0 / (1.0 + pow(2.718281828, -x))

        def _ret(n: int) -> float:
            if len(closes) <= n or closes[-n - 1] == 0:
                return 0.0
            return (closes[-1] / closes[-n - 1]) - 1.0

        return {
            "1D": round(_p(_ret(1)), 6),
            "5D": round(_p(_ret(5)), 6),
            "20D": round(_p(_ret(20)), 6),
        }

    def _notify(self, intents: Sequence[OrderIntent], stages: list[StageResult]) -> bool:
        any_sent = False
        for oi in intents:
            if oi.intent == "HOLD":
                continue
            try:
                if self.notifier.notify_signal(oi):
                    any_sent = True
            except Exception as e:  # network boundary only
                logger.error("telegram_notify_error", extra={"error_type": type(e).__name__})
                stages.append(StageResult("telegram", "FAIL", type(e).__name__))
        return any_sent
