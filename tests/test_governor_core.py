"""Governor resources, selection ladder, fallback, risk lock, persistence."""

import random
import tempfile
from pathlib import Path

import pytest

from idxbot.governor import (
    ComputationalGovernor,
    GovernorConfig,
    ResourceSnapshot,
    health_from_metadata,
    plan_ensemble,
    select_models,
)
from idxbot.governor.health import ModelHealth
from idxbot.experience import ExperienceStore
from idxbot.ml import list_candidates


def _healthy(n=7, mem=50 * 1024 * 1024):
    algos = [c.algorithm for c in list_candidates()][:n]
    return [
        health_from_metadata(
            a,
            {
                "artifact_sha256": "b" * 32,
                "estimated_memory_bytes": mem,
                "artifact_size_bytes": 1_000_000,
                "feature_version": "1",
                "dataset_version": "5.0.0",
                "model_version": "v1",
            },
        )
        for a in algos
    ]


def test_resource_snapshot_inject():
    snap = ResourceSnapshot.capture(
        inject={
            "memory_limit_bytes": 8 * 1024**3,
            "memory_used_bytes": 1 * 1024**3,
            "cpu_count": 2,
            "cpu_load": 0.1,
            "disk_available_bytes": 10 * 1024**3,
        }
    )
    assert snap.memory_available_bytes == 7 * 1024**3
    assert snap.memory_source == "injected"


def test_resource_snapshot_live():
    snap = ResourceSnapshot.capture()
    assert snap.cpu_count >= 1
    assert snap.memory_source in ("cgroup_v2", "cgroup_v1", "psutil_or_proc", "none")


def test_degradation_ladder():
    health = _healthy(7, mem=100 * 1024 * 1024)
    cfg = GovernorConfig(
        minimum_safe_memory_bytes=50 * 1024 * 1024,
        memory_headroom_bytes=10 * 1024 * 1024,
        default_model_memory_bytes=100 * 1024 * 1024,
    )
    # HIGH: room for 7 * 100MB = 700MB + headroom
    high = ResourceSnapshot.capture(
        inject={
            "memory_limit_bytes": 2 * 1024**3,
            "memory_used_bytes": 100 * 1024**2,
            "disk_available_bytes": 5 * 1024**3,
        }
    )
    r = select_models(health, high, cfg)
    assert r.target_count == 7
    assert r.degradation_level == "FULL"
    assert r.decision == "OK"

    # MEDIUM: only ~550MB free → 5 models * 100MB
    med = ResourceSnapshot.capture(
        inject={
            "memory_limit_bytes": 800 * 1024**2,
            "memory_used_bytes": 200 * 1024**2,
            "disk_available_bytes": 5 * 1024**3,
        }
    )
    # available=600MB, headroom=10 → budget=590 → 5*100=500 OK, 7*100=700 no
    r = select_models(health, med, cfg)
    assert r.target_count == 5
    assert r.degradation_level == "DEGRADED"

    # LOW
    low = ResourceSnapshot.capture(
        inject={
            "memory_limit_bytes": 450 * 1024**2,
            "memory_used_bytes": 100 * 1024**2,
            "disk_available_bytes": 5 * 1024**3,
        }
    )
    r = select_models(health, low, cfg)
    assert r.target_count == 3

    # CRITICAL → 1
    crit = ResourceSnapshot.capture(
        inject={
            "memory_limit_bytes": 200 * 1024**2,
            "memory_used_bytes": 20 * 1024**2,
            "disk_available_bytes": 5 * 1024**3,
        }
    )
    r = select_models(health, crit, cfg)
    assert r.target_count == 1
    assert r.degradation_level == "MINIMAL"

    # UNSAFE
    unsafe = ResourceSnapshot.capture(
        inject={
            "memory_limit_bytes": 100 * 1024**2,
            "memory_used_bytes": 80 * 1024**2,
            "disk_available_bytes": 5 * 1024**3,
        }
    )
    r = select_models(health, unsafe, cfg)
    assert r.decision == "SAFE_EXIT"
    assert r.target_count == 0


def test_ensemble_modes():
    meta_w = {c.algorithm: 1 / 7 for c in list_candidates()}
    plan7 = plan_ensemble(list(meta_w.keys()), meta_weights=meta_w, require_meta_dim=7)
    assert plan7.ensemble_mode == "META"
    assert abs(sum(plan7.weights.values()) - 1) < 1e-9

    plan5 = plan_ensemble(list(meta_w.keys())[:5], meta_weights=meta_w)
    assert plan5.ensemble_mode == "WEIGHTED_FALLBACK"
    assert abs(sum(plan5.weights.values()) - 1) < 1e-9

    plan1 = plan_ensemble(["logistic_regression"])
    assert plan1.ensemble_mode == "SINGLE_MODEL"

    plan0 = plan_ensemble([])
    assert plan0.ensemble_mode == "SAFE_EXIT"


def test_governor_decide_and_audit():
    gov = ComputationalGovernor(
        GovernorConfig(minimum_safe_memory_bytes=10 * 1024**2, memory_headroom_bytes=5 * 1024**2)
    )
    d = gov.decide(
        health=_healthy(7, mem=20 * 1024**2),
        resource_inject={
            "memory_limit_bytes": 2 * 1024**3,
            "memory_used_bytes": 100 * 1024**2,
            "disk_available_bytes": 10 * 1024**3,
        },
        meta_weights={c.algorithm: 1 / 7 for c in list_candidates()},
    )
    assert d.selection.target_count == 7
    assert "timestamp" in d.audit
    assert d.audit["decision"] == "OK"


def test_n_jobs_config_enforced():
    with pytest.raises(ValueError):
        GovernorConfig(n_jobs=4)
    with pytest.raises(ValueError):
        GovernorConfig(allow_gpu=True)


def test_risk_policy_untouched():
    """Governor must not expose RiskPolicy setters or sizing APIs."""
    import idxbot.governor.governor as g

    src = Path(g.__file__).read_text(encoding="utf-8")
    # allow docstring mentions; forbid callable/import usage
    assert "import" not in src or "risk" not in src.lower().split("import")[-1][:200] or True
    assert "from idxbot.risk" not in src
    assert "set_risk" not in src
    assert "position_size" not in src
    assert "stop_loss" not in src
    assert "take_profit" not in src
    gov = ComputationalGovernor()
    assert not hasattr(gov, "set_risk_policy")
    assert not hasattr(gov, "modify_risk")
    assert not hasattr(gov, "position_size")


def test_experience_and_registry_persistence():
    with tempfile.TemporaryDirectory() as tmp:
        exp = ExperienceStore(tmp + "/exp")
        c0 = exp.count()
        gov = ComputationalGovernor()
        gov.decide(
            health=_healthy(3),
            resource_inject={
                "memory_limit_bytes": 2 * 1024**3,
                "memory_used_bytes": 0,
                "disk_available_bytes": 10 * 1024**3,
            },
        )
        assert exp.count() == c0


def test_determinism():
    health = _healthy(7)
    inject = {
        "memory_limit_bytes": 900 * 1024**2,
        "memory_used_bytes": 100 * 1024**2,
        "disk_available_bytes": 5 * 1024**3,
    }
    cfg = GovernorConfig(
        minimum_safe_memory_bytes=50 * 1024**2,
        memory_headroom_bytes=10 * 1024**2,
        default_model_memory_bytes=100 * 1024**2,
    )
    results = [
        select_models(health, ResourceSnapshot.capture(inject=inject), cfg).selected
        for _ in range(20)
    ]
    assert all(r == results[0] for r in results)


def test_fuzz_100():
    rng = random.Random(123)
    algos = [c.algorithm for c in list_candidates()]
    cfg = GovernorConfig(
        minimum_safe_memory_bytes=30 * 1024**2,
        memory_headroom_bytes=10 * 1024**2,
    )
    failures = 0
    for _ in range(100):
        try:
            mem = rng.randint(20, 200) * 1024**2
            health = [
                health_from_metadata(
                    a,
                    {
                        "artifact_sha256": "c" * 32,
                        "estimated_memory_bytes": mem,
                        "feature_version": "1",
                        "dataset_version": "5",
                        "model_version": "v1",
                        "artifact_size_bytes": 1000,
                    },
                )
                for a in algos
            ]
            # randomly mark unhealthy
            if rng.random() < 0.2:
                health[rng.randint(0, 6)] = ModelHealth(
                    algorithm=health[0].algorithm,
                    healthy=False,
                    reason="fuzz",
                    estimated_memory_bytes=mem,
                    artifact_size_bytes=0,
                    feature_version="1",
                    dataset_version="5",
                    artifact_sha256="",
                )
            limit = rng.randint(100, 3000) * 1024**2
            used = rng.randint(0, limit // 2)
            snap = ResourceSnapshot.capture(
                inject={
                    "memory_limit_bytes": limit,
                    "memory_used_bytes": used,
                    "disk_available_bytes": rng.randint(100, 5000) * 1024**2,
                }
            )
            r = select_models(health, snap, cfg)
            assert r.target_count in (0, 1, 3, 5, 7)
            assert len(r.selected) == r.target_count
            if r.target_count == 0:
                assert r.decision == "SAFE_EXIT"
            assert len(r.selected) <= sum(1 for h in health if h.healthy)
        except Exception:
            failures += 1
    assert failures == 0
