"""
Structured JSON-ish logging.

Required fields when available:
  timestamp, level, event, run_id, market_date, market_session,
  state_version, duration_ms, status

NEVER log: telegram token, api key/secret, password, private key, credential.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any, Optional

SECRET_KEYS = frozenset(
    {
        "token",
        "api_key",
        "apikey",
        "api_secret",
        "password",
        "private_key",
        "secret",
        "credential",
        "telegram_bot_token",
        "authorization",
        "bearer",
    }
)

_SECRET_PATTERN = re.compile(
    r"(?i)(token|api[_-]?key|api[_-]?secret|password|private[_-]?key|credential|bearer)\s*[:=]\s*\S+"
)


def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k).lower() in SECRET_KEYS or any(
                s in str(k).lower() for s in ("token", "secret", "password", "key")
            ):
                out[k] = "***REDACTED***"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    if isinstance(obj, str):
        return _SECRET_PATTERN.sub(r"\1=***REDACTED***", obj)
    return obj


class StructuredLogger:
    def __init__(self, name: str = "idxbot") -> None:
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def log(
        self,
        level: str,
        event: str,
        *,
        run_id: Optional[str] = None,
        market_date: Optional[str] = None,
        market_session: Optional[str] = None,
        state_version: Optional[int] = None,
        duration_ms: Optional[float] = None,
        status: Optional[str] = None,
        **extra: Any,
    ) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.upper(),
            "event": event,
        }
        if run_id is not None:
            payload["run_id"] = run_id
        if market_date is not None:
            payload["market_date"] = market_date
        if market_session is not None:
            payload["market_session"] = market_session
        if state_version is not None:
            payload["state_version"] = state_version
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        if status is not None:
            payload["status"] = status
        payload.update(extra)
        safe = _redact(payload)
        line = json.dumps(safe, default=str)
        lvl = getattr(logging, level.upper(), logging.INFO)
        self._logger.log(lvl, line)

    def info(self, event: str, **kw: Any) -> None:
        self.log("INFO", event, **kw)

    def warning(self, event: str, **kw: Any) -> None:
        self.log("WARNING", event, **kw)

    def error(self, event: str, **kw: Any) -> None:
        self.log("ERROR", event, **kw)


_default: Optional[StructuredLogger] = None


def get_logger(name: str = "idxbot") -> StructuredLogger:
    global _default
    if _default is None:
        _default = StructuredLogger(name)
    return _default
