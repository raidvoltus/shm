"""
Gemini Flash-Lite narrator only.

Does NOT select symbols, change scores, or mutate portfolio.
On failure: returns deterministic fallback narration.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are only a financial-signal narrator. "
    "Do not make or modify trading decisions. "
    "Do not select a stock. "
    "Do not change symbol, action, price, quantity, score, confidence, cash, equity, positions, or P/L. "
    "Only explain the authoritative structured result provided by the trading engine. "
    "If information is missing, do not invent it. "
    "Respond in concise Indonesian for Telegram (2-4 short sentences)."
)

DEFAULT_MODEL = "gemini-2.0-flash-lite"


def _deterministic_fallback(payload: dict[str, Any]) -> str:
    sig = payload.get("signal") or {}
    action = sig.get("action", "NO_SIGNAL")
    sym = sig.get("symbol", "")
    if action == "BUY":
        reasons = sig.get("reasons") or []
        why = ", ".join(str(r) for r in reasons[:3]) if reasons else "skor tertinggi setelah filter risiko"
        return (
            f"Mesin trading memilih {sym} sebagai TOP 1 dengan aksi {action}. "
            f"Alasan utama: {why}. "
            f"Angka harga, kuantitas, skor, dan saldo berasal dari engine — bukan dari narator."
        )
    return (
        "Tidak ada setup yang lolos filter pada siklus ini (NO SIGNAL). "
        "Portfolio tidak diubah. Angka di pesan ini berasal dari engine trading."
    )


def _validate_narration(payload: dict[str, Any], text: str) -> bool:
    if not text or not text.strip():
        return False
    sig = payload.get("signal") or {}
    action = str(sig.get("action", "")).upper()
    upper = text.upper()
    if action == "BUY" and "FINAL: SELL" in upper:
        return False
    if action == "NO_SIGNAL" and ("FINAL: BUY" in upper or "DECISION: BUY" in upper):
        return False
    return True


class GeminiNarrator:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        enabled: Optional[bool] = None,
        timeout: float = 20.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        if enabled is None:
            enabled = bool(self.api_key) and os.environ.get("GEMINI_ENABLED", "true").lower() not in (
                "0", "false", "no", "off"
            )
        self.enabled = enabled
        self.timeout = timeout

    def narrate(self, payload: dict[str, Any]) -> str:
        if not self.enabled or not self.api_key:
            return _deterministic_fallback(payload)
        try:
            text = self._call_api(payload)
            if _validate_narration(payload, text):
                return text.strip()
            logger.warning("gemini_narration_rejected")
            return _deterministic_fallback(payload)
        except Exception as e:
            logger.warning("gemini_failed", extra={"error_type": type(e).__name__})
            return _deterministic_fallback(payload)

    def _call_api(self, payload: dict[str, Any]) -> str:
        model = self.model
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self.api_key}"
        )
        body = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "Authoritative trading-engine result (JSON). "
                                "Explain only; do not change any numbers or decisions:\n"
                                + json.dumps(payload, ensure_ascii=False, sort_keys=True)
                            )
                        }
                    ],
                }
            ],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 256},
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        candidates = result.get("candidates") or []
        if not candidates:
            raise RuntimeError("empty candidates")
        parts = candidates[0].get("content", {}).get("parts") or []
        texts = [p.get("text", "") for p in parts if p.get("text")]
        if not texts:
            raise RuntimeError("empty text")
        return "\n".join(texts)


def narrate_signal(payload: dict[str, Any]) -> str:
    return GeminiNarrator().narrate(payload)
