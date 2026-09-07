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
from idxbot.decision.engine import DecisionEngine
from idxbot.governor.governor import ComputationalGovernor, GovernorDecision
from idxbot.portfolio.governor import PortfolioGovernor, PortfolioDecision
from idxbot.runtime.paper_stage import run_paper_stage
from idxbot.signals.order_intent import OrderIntent, create_order_intent
from idxbot.signals.signal_engine import SignalEngine
from idxbot.telegram.message_composer import compose_with_fallback
from idxbot.telegram.notifier import TelegramNotifier
from idxbot.ml.inference import ChampionInferencer
from idxbot.self_learning.loop import SelfLearningLoop

logger = logging.getLogger(__name__)
JAKARTA = ZoneInfo("Asia/Jakarta")
DEFAULT_UNIVERSE = ("BBCA.JK", "BBRI.JK", "TLKM.JK", "ASII.JK", "BMRI.JK")


@dataclass
class StageResult:
    name: str
    status: str
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    run_id: str
    status: str
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
            "1", "true", "yes",
        )
        self.registry = provider_registry or build_default_registry(allow_fixture=self.allow_fixture)
        self.governor = governor or ComputationalGovernor()
        self.portfolio_gov = portfolio_governor or PortfolioGovernor()
        self.decision_engine = DecisionEngine()
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
        self._previous_intents: dict[str, str] = {}
        self._decision_by_sid: dict[str, Any] = {}
        self.registry_root = os.environ.get("IDXBOT_MODEL_REGISTRY", ".models")
        self.allow_momentum_fallback = os.environ.get(
            "IDXBOT_ALLOW_MOMENTUM_FALLBACK", ""
        ).lower() in ("1", "true", "yes")
        self.inferencer = ChampionInferencer(
            self.registry_root, expected_feature_version=feature_version,
        )
        self.learning = SelfLearningLoop(os.environ.get("IDXBOT_EXPERIENCE_ROOT", ".experience"))

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
        self._decision_by_sid = {}
        persistence_mode = "EPHEMERAL"

        gdec = self.governor.decide(resource_inject=resource_inject)
        gov_state = self._governor_state_label(gdec)
        stages.append(StageResult("governor", "OK", gov_state, {
            "decision": gdec.selection.decision,
            "target_count": gdec.selection.target_count,
            "selected": list(gdec.selection.selected),
            "audit": gdec.audit,
        }))

        if gdec.is_safe_exit or gdec.selection.decision == "SAFE_EXIT":
            stages.append(StageResult("ml", "SKIP", "SAFE_EXIT — no ML"))
            for sym in self.universe:
                oi = self.signal_engine.generate(
                    symbol=sym, timestamp=scheduled_at, probability=0.5,
                    governor_state="SAFE_EXIT", active_models=0, portfolio_allowed=True,
                    previous_intent=self._previous_intents.get(sym, "HOLD"), run_id=run_id,
                )
                intents.append(oi)
                self._previous_intents[sym] = oi.intent
            result = PipelineResult(
                run_id=run_id, status="SAFE_EXIT", stages=stages, intents=intents,
                governor_state="SAFE_EXIT", persistence_mode=persistence_mode,
                message="Governor SAFE_EXIT",
            )
            if not dry_run:
                self._notify(intents, stages)
            return result

        end = scheduled_at.date() if hasattr(scheduled_at, "date") else date.today()
        start = end - timedelta(days=120)
        symbol_data: dict[str, ProviderResult] = {}
        fetch_ok = 0
        for sym in list(self.universe):
            pr = self.registry.fetch_historical(sym, start, end)
            if not pr.ok and sym.endswith(".JK"):
                bare = sym[:-3]
                pr2 = self.registry.fetch_historical(bare, start, end)
                if pr2.ok:
                    for row in pr2.data:
                        row["symbol"] = sym
                    pr = pr2
            symbol_data[sym] = pr
            if pr.ok:
                fetch_ok += 1
            else:
                stages.append(StageResult("data_fetch", "FAIL", f"{sym}: {pr.status.value} {pr.error}", {"symbol": sym}))

        if fetch_ok == 0:
            stages.append(StageResult("data_fetch", "FAIL", "entire universe failed"))
            return PipelineResult(
                run_id=run_id, status="FAILED", stages=stages, intents=[],
                governor_state=gov_state, persistence_mode=persistence_mode,
                message="No market data", health={"data": "FAILED", "status": "FAILED"},
            )

        data_status = "DEGRADED" if any(p.status == ProviderStatus.FIXTURE for p in symbol_data.values()) else "OK"
        stages.append(StageResult("data_fetch", data_status, f"{fetch_ok}/{len(self.universe)} symbols",
            {"providers": {s: p.provider_name for s, p in symbol_data.items() if p.ok}}))

        active_n = self._active_model_count(gdec)
        for sym, pr in symbol_data.items():
            if not pr.ok:
                continue
            q = self._quality_check(pr.data)
            if q != "VALID":
                stages.append(StageResult("quality", "FAIL", f"{sym} {q}", {"symbol": sym}))
                sd = self.decision_engine.decide(
                    symbol=sym, timestamp=scheduled_at, raw_side="HOLD",
                    signal_probability=0.5, confidence=0.0, market_regime="UNKNOWN",
                    model_version=self.model_version, feature_version=self.feature_version,
                    governor_state=gov_state, price=None, missing_data=True,
                    data_stale=(q == "STALE"), extra_reasons=(f"QUALITY_{q}",),
                )
                oi = create_order_intent(
                    symbol=sym, timestamp=scheduled_at, intent="HOLD", confidence=0.0,
                    governor_state=gov_state, feature_version=self.feature_version,
                    model_version=self.model_version,
                    reason_codes=list(sd.risk_results) + list(sd.filter_results), run_id=run_id,
                )
                self._decision_by_sid[oi.signal_id] = sd
                intents.append(oi)
                continue

            prev = self._previous_intents.get(sym.upper(), "HOLD")
            last_close = float(pr.data[-1].get("close") or pr.data[-1].get("adjusted_close") or 0)
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
                probs = {"1D": 0.5, "5D": 0.5, "20D": 0.5}
                primary_p = 0.5
                reasons.append(f"MODEL_UNAVAILABLE:{inf.status}")
                sd = self.decision_engine.decide(
                    symbol=sym, timestamp=scheduled_at, raw_side="HOLD",
                    signal_probability=0.5, confidence=0.5, model_agreement=0.0,
                    model_version=model_ver, feature_version=feat_ver,
                    governor_state=gov_state, price=last_close if last_close > 0 else None,
                    extra_reasons=tuple(reasons) + ("HOLD_NO_CHAMPION",),
                )
                oi = create_order_intent(
                    symbol=sym, timestamp=scheduled_at, intent="HOLD", confidence=0.5,
                    governor_state=gov_state, feature_version=feat_ver, model_version=model_ver,
                    active_models=0, portfolio_allowed=True,
                    reason_codes=tuple(reasons) + ("HOLD_NO_CHAMPION",), horizons=probs, run_id=run_id,
                )
                self._decision_by_sid[oi.signal_id] = sd
                intents.append(oi)
                self._previous_intents[sym.upper()] = "HOLD"
                stages.append(StageResult("signal", "DEGRADED", f"{sym} NO_SIGNAL (no champion)",
                    {"symbol": sym, "intent": "HOLD", "ml_status": inf.status}))
                continue

            provisional = self.signal_engine.generate(
                symbol=sym, timestamp=scheduled_at, probability=primary_p, horizons=probs,
                previous_intent=prev, governor_state=gov_state, active_models=n_models,
                portfolio_allowed=True, run_id=run_id, extra_reasons=reasons,
            )
            pdec: PortfolioDecision = self.portfolio_gov.evaluate(
                intent=provisional.intent, symbol=sym, confidence=provisional.confidence,
                price=last_close, account=account,
            )
            final_intent = pdec.final_intent
            reasons = list(provisional.reason_codes) + list(pdec.reason_codes)
            if not pdec.allowed and provisional.intent == "BUY":
                final_intent = "HOLD"
                reasons.append("PORTFOLIO_BLOCK")

            closes = [float(r.get("close") or r.get("adjusted_close") or 0) for r in pr.data[-20:]]
            closes = [c for c in closes if c > 0]
            if len(closes) >= 5:
                rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
                import statistics
                vol = statistics.pstdev(rets) if len(rets) > 1 else 0.0
                volatility_score = min(1.0, vol * 25.0)
            else:
                volatility_score = 0.5
            liquidity_score = 0.6 if last_close > 0 else 0.0
            agreement = min(1.0, n_models / 7.0) if n_models else 0.0
            regime = getattr(provisional, "regime", "") or "UNKNOWN"

            sd = self.decision_engine.decide(
                symbol=sym, timestamp=scheduled_at, raw_side=final_intent,
                signal_probability=float(primary_p), confidence=float(provisional.confidence),
                model_agreement=agreement, volatility_score=volatility_score,
                liquidity_score=liquidity_score, expected_value=float(primary_p - 0.5),
                risk_reward=1.0, market_regime=regime, model_version=model_ver,
                feature_version=feat_ver, governor_state=gov_state, active_models=n_models,
                price=last_close if last_close > 0 else None, extra_reasons=tuple(reasons),
            )
            mapped = sd.to_order_intent_fields()
            oi = create_order_intent(
                symbol=sym, timestamp=scheduled_at, intent=mapped["intent"],  # type: ignore[arg-type]
                confidence=sd.confidence, governor_state=gov_state, feature_version=feat_ver,
                model_version=model_ver, active_models=n_models, portfolio_allowed=pdec.allowed,
                reason_codes=list(sd.filter_results) + list(sd.risk_results),
                horizons=probs, run_id=run_id,
            )
            self._decision_by_sid[oi.signal_id] = sd
            intents.append(oi)
            self._previous_intents[sym.upper()] = oi.intent
            stages.append(StageResult("signal", "OK", f"{sym} {sd.decision}", {
                "symbol": sym, "intent": oi.intent, "decision": sd.decision,
                "confidence": oi.confidence, "ml": inf.status, "risk_score": sd.risk_score,
            }))

        stages.append(StageResult("ensemble", "OK" if active_n else "DEGRADED", f"active_models={active_n}"))
        stages.append(StageResult("portfolio_governor", "OK", f"intents={len(intents)}"))

        if not dry_run and intents:
            try:
                n_exp = self.learning.persist_predictions(intents)
                stages.append(StageResult("experience", "OK", f"persisted={n_exp}"))
            except Exception as e:
                stages.append(StageResult("experience", "FAIL", type(e).__name__))

        portfolio_msg = None
        if not dry_run:
            try:
                portfolio_msg = run_paper_stage(
                    intents=intents, stages=stages, StageResult=StageResult,
                    symbol_data=symbol_data, run_id=run_id,
                    decision_by_sid=self._decision_by_sid,
                )
            except Exception as e:
                stages.append(StageResult("portfolio", "FAIL", type(e).__name__))
                if os.environ.get("IDXBOT_PORTFOLIO_REQUIRED", "").lower() in ("1", "true", "yes"):
                    return PipelineResult(
                        run_id=run_id, status="FAILED", stages=stages, intents=intents,
                        governor_state=gov_state, persistence_mode="PERSISTENT_LOCAL",
                        message="portfolio failure before telegram",
                    )

        if not dry_run:
            tg_ok = self._notify(intents, stages, portfolio_message=portfolio_msg)
            stages.append(StageResult("telegram", "OK" if tg_ok else "DEGRADED",
                "sent" if tg_ok else "failed_or_skipped"))

        overall = "OK"
        if data_status == "DEGRADED":
            overall = "DEGRADED"
        if not intents:
            overall = "FAILED"
        return PipelineResult(
            run_id=run_id, status=overall, stages=stages, intents=intents,
            governor_state=gov_state, persistence_mode=persistence_mode,
            message=f"produced {len(intents)} intents",
            health={"status": overall, "data": data_status, "governor": gov_state,
                    "telegram": "ok" if self.notifier.enabled else "disabled",
                    "persistence": persistence_mode},
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
        closes = []
        for r in rows:
            c = r.get("adjusted_close", r.get("close"))
            if c is not None:
                closes.append(float(c))
        row: dict[str, Any] = {}
        if len(closes) < 25:
            return row
        def ret(n: int) -> float:
            if closes[-n - 1] == 0:
                return 0.0
            return (closes[-1] / closes[-n - 1]) - 1.0
        row["return_1d"] = ret(1)
        row["return_5d"] = ret(5)
        row["return_20d"] = ret(20)
        import statistics
        window = closes[-20:]
        mean = statistics.fmean(window)
        row["sma_20"] = mean
        row["price_vs_sma20"] = (closes[-1] / mean - 1.0) if mean else 0.0
        return row

    def _momentum_horizons(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        closes = []
        for r in rows:
            c = r.get("adjusted_close", r.get("close"))
            if c is not None:
                closes.append(float(c))
        def _p(ret: float) -> float:
            x = max(-3.0, min(3.0, ret * 10))
            return 1.0 / (1.0 + pow(2.718281828, -x))
        def _ret(n: int) -> float:
            if len(closes) <= n or closes[-n - 1] == 0:
                return 0.0
            return (closes[-1] / closes[-n - 1]) - 1.0
        return {"1D": round(_p(_ret(1)), 6), "5D": round(_p(_ret(5)), 6), "20D": round(_p(_ret(20)), 6)}

    def _notify(self, intents: Sequence[OrderIntent], stages: list[StageResult], portfolio_message: str | None = None) -> bool:
        any_sent = False
        for oi in intents:
            try:
                sd = self._decision_by_sid.get(oi.signal_id)
                msg = compose_with_fallback(sd) if sd is not None else None
                if portfolio_message:
                    msg = (msg + "\n\n" + portfolio_message) if msg else portfolio_message
                if self.notifier.notify_signal(oi, decision=sd, message_text=msg):
                    any_sent = True
            except Exception as e:
                logger.error("telegram_notify_error", extra={"error_type": type(e).__name__})
                stages.append(StageResult("telegram", "FAIL", type(e).__name__))
        return any_sent
