"""P0: dynamic universe, production fixture=false, Gemini narrator-only."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from idxbot.data.providers.idx_universe import discover_idx_symbols, _normalize_many
from idxbot.narrator.gemini import GeminiNarrator, _deterministic_fallback, SYSTEM_PROMPT
from idxbot.runtime.pipeline import DEFAULT_UNIVERSE, AutonomousPipeline


def test_normalize_dedupes_and_canonicalizes():
    got = _normalize_many(["bbca", "BBCA.JK", "BBRI", "bbca.jk", "BAD!!!"])
    assert "BBCA.JK" in got
    assert "BBRI.JK" in got
    assert got.count("BBCA.JK") == 1


def test_discover_seed_fallback_not_five_only(monkeypatch):
    monkeypatch.delenv("IDXBOT_UNIVERSE_URL", raising=False)

    def boom(*a, **k):
        raise RuntimeError("no net")

    with patch("idxbot.data.providers.idx_universe.requests.Session.post", side_effect=boom):
        with patch("idxbot.data.providers.idx_universe.requests.Session.get", side_effect=boom):
            r = discover_idx_symbols(timeout=1.0)
    assert r["valid"] > 5


def test_pipeline_default_universe_constant_is_legacy_only():
    assert len(DEFAULT_UNIVERSE) == 5


def test_pipeline_expands_beyond_five(monkeypatch):
    """Explicit large universe is accepted; discovery seed >5."""
    monkeypatch.setenv("IDXBOT_USE_FIXTURE", "true")
    large = [f"S{i:03d}.JK" for i in range(100)]
    pipe = AutonomousPipeline(allow_fixture=True, universe=large)
    assert len(pipe.universe) == 100
    with patch("idxbot.data.providers.idx_universe.requests.Session.get", side_effect=RuntimeError("no")):
        with patch("idxbot.data.providers.idx_universe.requests.Session.post", side_effect=RuntimeError("no")):
            r = discover_idx_symbols(timeout=1.0)
    assert r["valid"] > 5


def test_production_signal_workflow_fixture_false():
    root = Path(__file__).resolve().parents[2]
    yml = (root / ".github/workflows/signal.yml").read_text()
    assert 'IDXBOT_USE_FIXTURE: "false"' in yml
    assert 'IDXBOT_USE_FIXTURE: "true"' not in yml


def test_gemini_system_prompt_is_narrator_only():
    assert "narrator" in SYSTEM_PROMPT.lower()
    assert "Do not make or modify trading decisions" in SYSTEM_PROMPT


def test_gemini_fallback_does_not_change_numbers():
    payload = {
        "signal": {
            "symbol": "BBCA",
            "action": "BUY",
            "price": 8750,
            "quantity": 100,
            "score": 87.4,
            "confidence": 0.78,
            "reasons": ["momentum"],
        },
        "portfolio": {"cash_available": 1236875, "equity": 10000000, "positions": []},
    }
    text = _deterministic_fallback(payload)
    assert "BBCA" in text
    assert "FINAL: SELL" not in text.upper()


def test_gemini_disabled_without_key_uses_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    n = GeminiNarrator(api_key="", enabled=True)
    n.enabled = True
    n.api_key = ""
    out = n.narrate({"signal": {"action": "NO_SIGNAL", "symbol": ""}, "portfolio": {}})
    assert isinstance(out, str) and len(out) > 0


def test_gemini_failure_does_not_raise(monkeypatch):
    n = GeminiNarrator(api_key="fake", enabled=True)

    def boom(*a, **k):
        raise TimeoutError("api down")

    with patch.object(n, "_call_api", side_effect=boom):
        text = n.narrate({"signal": {"action": "BUY", "symbol": "BBCA", "reasons": []}, "portfolio": {}})
    assert isinstance(text, str) and len(text) > 0
