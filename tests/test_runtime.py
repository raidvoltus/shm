"""Runtime entry point, dry-run, health, balance persistence, logging secrets."""

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from idxbot.logging.structured import _redact, SECRET_KEYS
from idxbot.runtime.clock import MarketClock
from idxbot.runtime.runner import MarketRuntime
from idxbot.storage.backend import LocalStorageBackend

JAKARTA = ZoneInfo("Asia/Jakarta")


class FixedClock(MarketClock):
    """Clock that freezes `now` at a given aware datetime."""

    def __init__(self, fixed: datetime) -> None:
        super().__init__()
        self._fixed = fixed

    def now(self) -> datetime:
        return self._fixed


def test_health_healthy():
    with tempfile.TemporaryDirectory() as tmp:
        rt = MarketRuntime(storage=LocalStorageBackend(tmp))
        h = rt.health()
        assert h["status"] == "Healthy"
        assert h["checks"]["live_trading"] is False
        assert h["checks"]["paper_trading"] is True
        assert h["checks"]["initial_balance"] == 10_000_000


def test_dry_run_no_persist():
    with tempfile.TemporaryDirectory() as tmp:
        be = LocalStorageBackend(tmp)
        scheduled = datetime(2024, 6, 17, 10, 0, tzinfo=JAKARTA)
        rt = MarketRuntime(storage=be, clock=FixedClock(scheduled))
        result = rt.run(run_id="dry1", scheduled_at=scheduled, dry_run=True)
        assert result.status in ("OK", "SKIP")
        assert be.load_state("portfolio") is None


def test_balance_persists_across_runs():
    with tempfile.TemporaryDirectory() as tmp:
        be = LocalStorageBackend(tmp)
        scheduled = datetime(2024, 6, 17, 10, 0, tzinfo=JAKARTA)
        rt = MarketRuntime(storage=be, clock=FixedClock(scheduled))
        r1 = rt.run(run_id="p1", scheduled_at=scheduled, dry_run=False)
        assert r1.status == "OK", r1.message
        raw = be.load_state("portfolio")
        assert raw is not None
        assert raw["account"]["cash"] == 10_000_000

        scheduled2 = datetime(2024, 6, 17, 14, 0, tzinfo=JAKARTA)
        rt.clock = FixedClock(scheduled2)
        r2 = rt.run(run_id="p2", scheduled_at=scheduled2, dry_run=False)
        assert r2.status == "OK", r2.message
        raw2 = be.load_state("portfolio")
        assert raw2["account"]["cash"] == 10_000_000
        assert raw2["account"]["initial_balance"] == 10_000_000


def test_duplicate_run_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        be = LocalStorageBackend(tmp)
        scheduled = datetime(2024, 6, 17, 10, 0, tzinfo=JAKARTA)
        rt = MarketRuntime(storage=be, clock=FixedClock(scheduled))
        r1 = rt.run(run_id="same", scheduled_at=scheduled, dry_run=False)
        assert r1.status == "OK", r1.message
        r2 = rt.run(run_id="same", scheduled_at=scheduled, dry_run=False)
        assert r2.status == "SKIP"
        assert "DUPLICATE" in r2.message


def test_late_run_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        be = LocalStorageBackend(tmp)
        scheduled = datetime(2024, 6, 17, 9, 0, tzinfo=JAKARTA)
        late = scheduled.replace(hour=9, minute=40)
        rt = MarketRuntime(storage=be, clock=FixedClock(late))
        result = rt.run(run_id="late", scheduled_at=scheduled, dry_run=True)
        assert result.status == "SKIP"
        assert "latency" in result.message.lower() or "SKIP" in result.message


def _cli_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return env


def test_cli_help():
    r = subprocess.run(
        [sys.executable, "-m", "idxbot", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=_cli_env(),
        timeout=30,
    )
    assert r.returncode == 0
    out = r.stdout + r.stderr
    assert "dry-run" in out or "health" in out


def test_cli_health():
    r = subprocess.run(
        [sys.executable, "-m", "idxbot", "--health", "--json"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=_cli_env(),
        timeout=30,
    )
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["status"] == "Healthy"


def test_cli_dry_run_fast():
    t0 = time.perf_counter()
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "idxbot",
            "--dry-run",
            "--json",
            "--scheduled-at",
            "2024-06-17T10:00:00+07:00",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=_cli_env(),
        timeout=120,
    )
    elapsed = time.perf_counter() - t0
    assert r.returncode == 0, r.stderr
    assert elapsed < 120
    # structured logs may interleave; take last JSON object
    objects = []
    for ln in r.stdout.splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                objects.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    assert objects
    # final result has status
    statuses = [o.get("status") for o in objects if "status" in o]
    assert any(s in ("OK", "SKIP", "FATAL", "CONFLICT") for s in statuses)


def test_logging_redacts_secrets():
    payload = {
        "token": "123:ABC",
        "api_key": "sk-secret",
        "password": "hunter2",
        "private_key": "-----BEGIN",
        "normal": "ok",
        "msg": "token=should-hide api_key=xyz",
    }
    red = _redact(payload)
    assert red["token"] == "***REDACTED***"
    assert red["api_key"] == "***REDACTED***"
    assert red["password"] == "***REDACTED***"
    assert red["private_key"] == "***REDACTED***"
    assert red["normal"] == "ok"
    assert "***REDACTED***" in red["msg"]
    assert "token" in SECRET_KEYS or "api_key" in SECRET_KEYS
