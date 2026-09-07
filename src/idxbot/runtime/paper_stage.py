"""Paper portfolio stage applied after signal generation."""
from __future__ import annotations

import os
from typing import Any, Sequence

from idxbot.portfolio.store import (
    LocalPortfolioStore,
    PaperPortfolioEngine,
    PortfolioStoreError,
    format_portfolio_telegram,
)
from idxbot.signals.order_intent import OrderIntent


def run_paper_stage(
    *,
    intents: Sequence[OrderIntent],
    stages: list,
    StageResult: type,
    symbol_data: dict,
    run_id: str,
    decision_by_sid: dict,
) -> str | None:
    """TOP-1 BUY → persist → return portfolio telegram block."""
    store = LocalPortfolioStore(os.environ.get("IDXBOT_PORTFOLIO_DIR", ".state"))
    engine = PaperPortfolioEngine(store)
    try:
        bundle = engine.load()
    except Exception as e:
        stages.append(StageResult("portfolio_load", "FAIL", type(e).__name__))
        if os.environ.get("IDXBOT_PORTFOLIO_REQUIRED", "").lower() in ("1", "true", "yes"):
            raise
        return None
    stages.append(
        StageResult(
            "portfolio_load",
            "OK",
            f"cash={bundle.account.cash_available:.0f} npos={len(bundle.positions)}",
        )
    )
    buys = sorted(
        [i for i in intents if i.intent == "BUY"],
        key=lambda x: x.confidence,
        reverse=True,
    )
    if not buys:
        stages.append(StageResult("paper_buy", "SKIP", "no BUY"))
        return format_portfolio_telegram(bundle, top={"action": "NO_SIGNAL"})
    top = buys[0]
    top_price = 0.0
    pr = symbol_data.get(top.symbol)
    if pr and getattr(pr, "ok", False) and pr.data:
        top_price = float(pr.data[-1].get("close") or pr.data[-1].get("adjusted_close") or 0)
    if top_price <= 0:
        stages.append(StageResult("paper_buy", "SKIP", "no price"))
        return format_portfolio_telegram(bundle, top={"action": "NO_SIGNAL"})
    try:
        bundle, tx = engine.apply_buy(
            bundle,
            symbol=top.symbol.replace(".JK", ""),
            price=top_price,
            signal_id=top.signal_id,
            cycle_id=run_id,
            confidence=top.confidence,
            score=top.confidence * 100,
            reasons=tuple(top.reason_codes[:8]),
        )
        store.save(bundle)
        stages.append(
            StageResult(
                "paper_buy",
                "OK",
                f"{top.symbol} qty={tx['quantity']} cash={bundle.account.cash_available:.0f}",
                {"transaction_id": tx["transaction_id"]},
            )
        )
        return format_portfolio_telegram(
            bundle,
            top={
                "action": "BUY",
                "symbol": top.symbol.replace(".JK", ""),
                "price": top_price,
                "quantity": tx["quantity"],
                "score": top.confidence * 100,
                "confidence": top.confidence,
                "why": "TOP 1 setelah decision gate + risk filter.",
            },
        )
    except PortfolioStoreError as e:
        stages.append(StageResult("paper_buy", "FAIL", str(e)))
        return format_portfolio_telegram(
            bundle, top={"action": "NO_SIGNAL", "symbol": top.symbol}
        )
