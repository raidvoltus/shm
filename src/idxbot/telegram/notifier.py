"""
Telegram notifier with cross-run idempotency ledger.

Secrets from environment only:
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID

Never log secrets. Limited retries with delivery classification.
HOLD messages are optional (default: do not spam).

Cross-run: IdempotencyLedger (GitHub branch / file / memory).
Process-level: skip re-send of the same signal_id within this process.

Guarantee: AT-MOST-ONCE for BUY/SELL when ledger required (prefer miss over duplicate).
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


def _ledger_required() -> bool:
    raw = os.environ.get("IDXBOT_LEDGER_REQUIRED", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
        return True
    if os.environ.get("IDXBOT_LEDGER_BACKEND", "").strip().lower() == "github":
        return True
    return False


def _build_ledger():
    from idxbot.idempotency.github_store import build_ledger_store
    from idxbot.idempotency.ledger import IdempotencyLedger

    return IdempotencyLedger(build_ledger_store())


class TelegramNotifier:
    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: Optional[bool] = None,
        send_hold: bool = False,
        ledger: Any = None,
    ) -> None:
        self.token = token if token is not None else os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID", "")
        if enabled is None:
            enabled = bool(self.token and self.chat_id)
        self.enabled = enabled
        self.send_hold = send_hold
        self._sent_signal_ids: Set[str] = set()
        self._ledger = ledger

    def _get_ledger(self):
        if self._ledger is False:
            return None
        if self._ledger is not None:
            return self._ledger
        try:
            self._ledger = _build_ledger()
            return self._ledger
        except Exception as e:  # noqa: BLE001
            logger.error("ledger_init_failed", extra={"error_type": type(e).__name__})
            if _ledger_required():
                raise
            self._ledger = False
            return None

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
        from idxbot.idempotency.ledger import DeliveryStatus, LedgerUnavailable

        if intent.intent == "HOLD" and not self.send_hold:
            return False

        signal_id = intent.signal_id
        text = self.format_signal(intent)

        ledger = None
        try:
            ledger = self._get_ledger()
        except LedgerUnavailable:
            if _ledger_required():
                logger.error("ledger_unavailable_fail_closed", extra={"signal_id_prefix": signal_id[:12]})
                return False
            ledger = None
        except Exception as e:  # noqa: BLE001
            if _ledger_required():
                logger.error(
                    "ledger_unavailable_fail_closed",
                    extra={"error_type": type(e).__name__, "signal_id_prefix": signal_id[:12]},
                )
                return False
            ledger = None

        if ledger is not None:
            try:
                if ledger.should_skip(signal_id):
                    logger.info("telegram_ledger_skip", extra={"signal_id_prefix": signal_id[:12]})
                    return False
                reserved = ledger.reserve(signal_id, symbol=intent.symbol, intent=intent.intent)
                if not reserved:
                    logger.info("telegram_ledger_skip_reserve", extra={"signal_id_prefix": signal_id[:12]})
                    return False
            except LedgerUnavailable:
                if _ledger_required():
                    logger.error("ledger_unavailable_fail_closed")
                    return False
                ledger = None

        result = self.send_message(text, idempotency_key=signal_id)

        if ledger is not None:
            status_map = {
                DeliveryClass.SUCCESS: DeliveryStatus.SUCCESS,
                DeliveryClass.UNKNOWN_DELIVERY_STATE: DeliveryStatus.UNKNOWN_DELIVERY_STATE,
                DeliveryClass.PERMANENT_FAILURE: DeliveryStatus.PERMANENT_FAILURE,
                DeliveryClass.TRANSIENT_FAILURE: DeliveryStatus.TRANSIENT_FAILURE,
                DeliveryClass.SKIPPED: None,
            }
            st = status_map.get(result)
            if st is not None:
                try:
                    ledger.finalize(signal_id, st, symbol=intent.symbol, intent=intent.intent)
                except LedgerUnavailable:
                    logger.error(
                        "ledger_finalize_failed",
                        extra={"signal_id_prefix": signal_id[:12], "status": result.value},
                    )

        return result == DeliveryClass.SUCCESS

    def notify_daily(self, summary: dict[str, Any]) -> bool:
        text = self.format_daily_summary(summary)
        result = self.send_message(text)
        return result == DeliveryClass.SUCCESS
