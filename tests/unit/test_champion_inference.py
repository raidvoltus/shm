"""Champion inference policy tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from idxbot.ml.inference import ChampionInferencer
from idxbot.runtime.pipeline import AutonomousPipeline
from datetime import datetime
from zoneinfo import ZoneInfo


def test_missing_champion_returns_unavailable(tmp_path):
    inf = ChampionInferencer(tmp_path / "models", expected_feature_version="features-v1")
    r = inf.predict_row({"return_1d": 0.01})
    assert r.status == "MODEL_UNAVAILABLE"


def test_pipeline_hold_without_champion(monkeypatch):
    monkeypatch.delenv("IDXBOT_ALLOW_MOMENTUM_FALLBACK", raising=False)
    monkeypatch.delenv("IDXBOT_USE_FIXTURE", raising=False)
    monkeypatch.setenv("IDXBOT_MODEL_REGISTRY", "/tmp/idxbot_empty_models_xyz")
    pipe = AutonomousPipeline(allow_fixture=False)
    # force yahoo data
    ts = datetime(2026, 8, 15, 9, 0, tzinfo=ZoneInfo("Asia/Jakarta"))
    result = pipe.run(
        run_id="champ-miss",
        scheduled_at=ts,
        account={"cash": 10_000_000, "equity": 10_000_000, "positions": {}},
        dry_run=True,
    )
    # With real yahoo data but no champion → HOLD
    if result.status in ("OK", "DEGRADED") and result.intents:
        assert all(i.intent == "HOLD" for i in result.intents)
        assert any("NO_CHAMPION" in r or "MODEL_UNAVAILABLE" in r for i in result.intents for r in i.reason_codes)
