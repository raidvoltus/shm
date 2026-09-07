"""
Deterministic message composer (LLM-role without external API).

Production default: template-based, CPU-friendly, no API key.
LLM/composer MUST NOT change decision, numbers, or SignalID.
"""

from __future__ import annotations

import re
from typing import Optional

from idxbot.decision.signal_decision import SignalDecision


def compose_report(decision: SignalDecision) -> str:
    d = decision
    label = d.decision if d.decision != "NO_SIGNAL" else "NO SIGNAL"
    conf_pct = f"{d.confidence * 100:.1f}%"
    risk = f"{d.risk_score:.0f}/100"
    rr = f"{d.risk_reward:.2f}"
    ev = f"{d.expected_value * 100:.2f}%" if abs(d.expected_value) <= 1 else f"{d.expected_value:.2f}"

    filters = d.filter_results or ()
    risks = d.risk_results or ()

    if d.decision in ("BUY", "SELL"):
        why = "; ".join(d.explanation_context[:4]) if d.explanation_context else "Setup passed gates"
        risk_lines = "; ".join(risks) if risks else "No hard-gate failures"
        filter_lines = "\n".join(f"• {f}" for f in filters[:8]) or "• (none)"
        return "\n".join(
            [
                "IDX SIGNAL REPORT",
                "",
                f"Decision: {label}",
                f"Symbol: {d.symbol}",
                "",
                f"Confidence: {conf_pct}",
                f"Risk Score: {risk}",
                f"R/R: {rr}",
                f"Expected Value: {ev}",
                f"Market Regime: {d.market_regime}",
                f"Model agreement: {d.model_agreement:.2f}",
                "",
                "Why:",
                why,
                "",
                "Risk:",
                risk_lines,
                "",
                "Filters:",
                filter_lines,
                "",
                f"Final: {label} SIGNAL",
                f"Signal ID: {d.signal_id}",
                f"Models: {d.model_version} / features {d.feature_version}",
            ]
        )

    blocked = list(risks) + [
        f
        for f in filters
        if "BLOCK" in f or "BELOW" in f or "ABOVE" in f or "NO_SETUP" in f or "HARD" in f
    ]
    if not blocked:
        blocked = list(filters[-5:]) or ["No qualifying setup"]
    why_no = (
        "; ".join(d.explanation_context[:5])
        if d.explanation_context
        else "Filters/risk did not clear"
    )
    return "\n".join(
        [
            "IDX SIGNAL REPORT",
            "",
            "Decision: NO SIGNAL",
            f"Symbol: {d.symbol}",
            f"Market Regime: {d.market_regime}",
            "",
            "Why no trade:",
            why_no,
            "",
            "ML assessment:",
            f"Confidence: {conf_pct}",
            f"Risk Score: {risk}",
            f"R/R: {rr}",
            f"Probability: {d.signal_probability:.2f}",
            "",
            "Blocked by:",
            *[f"• {b}" for b in blocked[:10]],
            "",
            "Final: NO TRADE",
            f"Signal ID: {d.signal_id}",
            f"Models: {d.model_version} / features {d.feature_version}",
        ]
    )


def validate_composed_message(decision: SignalDecision, text: str) -> bool:
    if not text or not text.strip():
        return False
    upper = text.upper()
    if decision.signal_id not in text:
        return False
    if decision.decision == "BUY":
        if "DECISION: SELL" in upper or "FINAL: SELL" in upper:
            return False
        if "DECISION: BUY" not in upper and "FINAL: BUY" not in upper:
            return False
    elif decision.decision == "SELL":
        if "DECISION: BUY" in upper or "FINAL: BUY SIGNAL" in upper:
            return False
        if "DECISION: SELL" not in upper and "FINAL: SELL" not in upper:
            return False
    else:
        if re.search(r"FINAL:\s*BUY", upper) or re.search(r"FINAL:\s*SELL", upper):
            return False
        if "DECISION: BUY" in upper or "DECISION: SELL" in upper:
            return False
        if "NO SIGNAL" not in upper and "NO TRADE" not in upper:
            return False
    ids = re.findall(r"\b[a-f0-9]{32,64}\b", text.lower())
    for found in ids:
        if found != decision.signal_id and len(found) == len(decision.signal_id):
            return False
    return True


def compose_with_fallback(decision: SignalDecision, llm_text: Optional[str] = None) -> str:
    if llm_text and validate_composed_message(decision, llm_text):
        return llm_text
    return compose_report(decision)
