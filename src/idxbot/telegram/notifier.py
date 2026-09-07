"""
Telegram notifier.

Secrets from environment only:
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID

Never log secrets. Limited retries with delivery classification.
HOLD messages are optional (default: do not spam).

Cross-run: rely on deterministic OrderIntent.signal_id in message body.
Process-level: skip re-send of the same signal_id within this process.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from enum import Enum
from typing import Any, Optional, Set

from idxbot.signals.order_intent import OrderIntent

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
TIMEOUT_SEC = 10.0
RETRY_BACKOFF = 1.5


class DeliveryClass(str, Enum):
    SUCCESS = "SUCCESS"
    TRANSIENT_FAILURE = "TRANSIENT_FAILURE"
    PERMANENT_FAILURE = "PERMANENT_FAILURE"
    UNKNOWN_DELIVERY_STATE = "UNKNOWN_DELIVERY_STATE"
    SKIPPED = "SKIPPED"


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
        self._sent_signal_ids: Set[str] = set()

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
            f"SignalID: {intent.signal_id}",
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

    def send_message(self, text: str, *, idempotency_key: str = "") -> DeliveryClass:
        if not self.enabled:
            logger.info("telegram_disabled")
            return DeliveryClass.SKIPPED
        if not self.token or not self.chat_id:
            logger.warning("telegram_missing_credentials")
            return DeliveryClass.PERMANENT_FAILURE

        if idempotency_key and idempotency_key in self._sent_signal_ids:
            logger.info(
                "telegram_dedupe_skip",
                extra={"signal_id_prefix": idempotency_key[:12]},
            )
            return DeliveryClass.SKIPPED

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
                    if 200 <= resp.status < 300:
                        if idempotency_key:
                            self._sent_signal_ids.add(idempotency_key)
                        logger.info(
                            "telegram_sent",
                            extra={
                                "attempt": attempt,
                                "chat_id_masked": self._mask(self.chat_id),
                                "signal_id_prefix": (idempotency_key[:12] if idempotency_key else ""),
                            },
                        )
                        return DeliveryClass.SUCCESS
                    if 500 <= resp.status < 600:
                        cls = DeliveryClass.TRANSIENT_FAILURE
                    elif resp.status == 429:
                        cls = DeliveryClass.TRANSIENT_FAILURE
                    else:
                        logger.error(
                            "telegram_permanent_http",
                            extra={"status": resp.status, "attempt": attempt},
                        )
                        return DeliveryClass.PERMANENT_FAILURE
            except TimeoutError:
                logger.warning(
                    "telegram_unknown_delivery",
                    extra={"attempt": attempt, "error_type": "TimeoutError"},
                )
                return DeliveryClass.UNKNOWN_DELIVERY_STATE
            except urllib.error.HTTPError as e:
                if e.code == 429 or (500 <= e.code < 600):
                    cls = DeliveryClass.TRANSIENT_FAILURE
                    logger.warning(
                        "telegram_attempt_failed",
                        extra={"attempt": attempt, "error_type": "HTTPError", "code": e.code},
                    )
                else:
                    logger.error(
                        "telegram_permanent_http",
                        extra={"attempt": attempt, "code": getattr(e, "code", None)},
                    )
                    return DeliveryClass.PERMANENT_FAILURE
            except (urllib.error.URLError, OSError) as e:
                cls = DeliveryClass.TRANSIENT_FAILURE
                logger.warning(
                    "telegram_attempt_failed",
                    extra={"attempt": attempt, "error_type": type(e).__name__},
                )
            else:
                cls = DeliveryClass.TRANSIENT_FAILURE

            if cls == DeliveryClass.TRANSIENT_FAILURE and attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
                continue
            break

        logger.error("telegram_failed", extra={"class": "TRANSIENT_FAILURE"})
        return DeliveryClass.TRANSIENT_FAILURE

    def notify_signal(self, intent: OrderIntent) -> bool:
        if intent.intent == "HOLD" and not self.send_hold:
            return False
        text = self.format_signal(intent)
        result = self.send_message(text, idempotency_key=intent.signal_id)
        return result == DeliveryClass.SUCCESS

    def notify_daily(self, summary: dict[str, Any]) -> bool:
        text = self.format_daily_summary(summary)
        result = self.send_message(text)
        return result == DeliveryClass.SUCCESS
