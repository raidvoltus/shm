"""Experience store: partition, dedup, incremental, paper isolation."""

import tempfile
import time
from datetime import date, timedelta

from idxbot.data.normalize import normalize_record
from idxbot.data.providers.fixture import _bar
from idxbot.dataset.builder import DatasetBuilder
from idxbot.dataset.labels import LabelGenerator
from idxbot.experience import ExperienceStore, ExperienceRecord
from idxbot.features import FeatureEngine
from idxbot.storage.state import AccountState, PortfolioState
from idxbot.storage.backend import LocalStorageBackend


def _rows(n=40):
    bars = []
    d = date(2024, 6, 3)
    price = 3000.0
    while len(bars) < n:
        if d.weekday() < 5:
            c = price * 1.001
            bars.append(normalize_record(_bar("EX", d, price, c * 1.01, c * 0.99, c, 1_000_000)))
            price = c
        d += timedelta(days=1)
    feats = FeatureEngine().transform(bars)
    closes = [b.close for b in bars]
    ts = [b.timestamp.isoformat() for b in bars]
    rows, _ = DatasetBuilder(LabelGenerator(horizons=(1, 5), threshold_mode="fixed")).build(
        feats, {"EX": closes}, {"EX": ts}
    )
    return DatasetBuilder.training_rows(rows)


def test_dedup_idempotent_append():
    with tempfile.TemporaryDirectory() as tmp:
        store = ExperienceStore(tmp)
        rows = _rows(30)
        r1 = store.append_from_dataset_rows(rows)
        r2 = store.append_from_dataset_rows(rows)
        assert r1["added"] > 0
        assert r2["added"] == 0
        assert r2["skipped_duplicates"] == r1["added"]
        assert store.count() == r1["added"]
        assert len(store.unique_ids()) == store.count()


def test_partition_structure():
    with tempfile.TemporaryDirectory() as tmp:
        store = ExperienceStore(tmp)
        store.append_from_dataset_rows(_rows(25))
        parts = list(store.root.rglob("experiences.parquet"))
        assert parts
        assert any("year=" in str(p) and "month=" in str(p) for p in parts)


def test_incremental_benchmark():
    with tempfile.TemporaryDirectory() as tmp:
        store = ExperienceStore(tmp)
        # seed history across months by crafting records
        recs = []
        for month in (1, 2, 3, 4, 5):
            for day in range(1, 6):
                ts = f"2024-{month:02d}-{day:02d}T16:00:00+07:00"
                eid = ExperienceRecord.make_id("BM", ts, 1, "1", "1")
                recs.append(
                    ExperienceRecord(
                        experience_id=eid,
                        symbol="BM",
                        timestamp=ts,
                        horizon=1,
                        label="FLAT",
                    )
                )
        store.append(recs)
        # append only June — should touch 1 partition
        june = [
            ExperienceRecord(
                experience_id=ExperienceRecord.make_id("BM", "2024-06-10T16:00:00+07:00", 1, "1", "1"),
                symbol="BM",
                timestamp="2024-06-10T16:00:00+07:00",
                horizon=1,
                label="UP",
            )
        ]
        t0 = time.perf_counter()
        result = store.append(june)
        elapsed = time.perf_counter() - t0
        assert result["partitions_touched"] == 1
        assert elapsed < 5.0  # far under heavy full-history rewrite


def test_paper_reset_does_not_touch_experience():
    with tempfile.TemporaryDirectory() as tmp:
        exp_dir = tmp + "/exp"
        state_dir = tmp + "/state"
        store = ExperienceStore(exp_dir)
        rows = _rows(20)
        store.append_from_dataset_rows(rows)
        before = store.count()
        ids_before = store.unique_ids()

        # paper account cycle
        backend = LocalStorageBackend(state_dir)
        ps = PortfolioState.create_initial(10_000_000)
        backend.save_state("portfolio", ps.to_dict())
        # reset balance
        ps2 = PortfolioState.create_initial(10_000_000)
        backend.save_state("portfolio", ps2.to_dict())

        assert store.count() == before
        assert store.unique_ids() == ids_before
