"""
Telegram notifier.

Secrets from environment only:
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID

Never log secrets. Limited retries. Timeout required.
HOLD messages are optional (default: do not spam).
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from idxbot.signals.order_intent import OrderIntent

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
TIMEOUT_SEC = 10.0
RETRY_BACKOFF = 1.5


class TelegramNotifier:
    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: Optional[bool] = None,
        send_hold: bool = False,
    ) -> None:
        self.token = token if token is not None else os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID", "")
        if enabled is None:
            enabled = bool(self.token and self.chat_id)
        self.enabled = enabled
        self.send_hold = send_hold

    def _mask(self, s: str) -> str:
        if not s:
            return ""
        if len(s) <= 8:
            return "***"
        return s[:4] + "***" + s[-2:]

    def format_signal(self, intent: OrderIntent) -> str:
        conf_pct = f"{intent.confidence * 100:.2f}%"
        lines = [
            "IDX SIGNAL BOT",
            "",
            f"Symbol: {intent.symbol}",
            f"Intent: {intent.intent}",
            f"Confidence: {conf_pct}",
            f"Models: {intent.active_models}/7",
            f"Governor: {intent.governor_state}",
            f"Portfolio: {'ALLOWED' if intent.portfolio_allowed else 'BLOCKED'}",
            f"Timestamp: {intent.timestamp}",
        ]
        if intent.reason_codes:
            lines.append(f"Reasons: {', '.join(intent.reason_codes)}")
        return "\n".join(lines)

    def format_daily_summary(self, summary: dict[str, Any]) -> str:
        lines = [
            "IDX DAILY SUMMARY",
            "",
            f"Signals: {summary.get('signals_generated', 0)}",
            f"BUY: {summary.get('buy', 0)}  SELL: {summary.get('sell', 0)}  HOLD: {summary.get('hold', 0)}",
            f"Active models: {summary.get('active_models', 0)}",
            f"Governor: {summary.get('governor_state', 'N/A')}",
            f"Portfolio balance: {summary.get('portfolio_balance', 'N/A')}",
            f"Health: {summary.get('health_status', 'N/A')}",
            f"Timestamp: {summary.get('timestamp', '')}",
        ]
        return "\n".join(lines)

    def send_message(self, text: str) -> bool:
        if not self.enabled:
            logger.info("telegram_disabled")
            return False
        if not self.token or not self.chat_id:
            logger.warning("telegram_missing_credentials")
            return False

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        last_err: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
                    if 200 <= resp.status < 300:
                        logger.info(
                            "telegram_sent",
                            extra={"attempt": attempt, "chat_id_masked": self._mask(self.chat_id)},
                        )
                        return True
                    last_err = RuntimeError(f"HTTP {resp.status}")
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = e
                logger.warning(
                    "telegram_attempt_failed",
                    extra={"attempt": attempt, "error_type": type(e).__name__},
                )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)

        logger.error(
            "telegram_failed",
            extra={"error_type": type(last_err).__name__ if last_err else "unknown"},
        )
        return False

    def notify_signal(self, intent: OrderIntent) -> bool:
        if intent.intent == "HOLD" and not self.send_hold:
            return False
        text = self.format_signal(intent)
        return self.send_message(text)

    def notify_daily(self, summary: dict[str, Any]) -> bool:
        text = self.format_daily_summary(summary)
        return self.send_message(text)
