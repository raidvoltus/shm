"""Partitioned Parquet, delta, compaction, provenance."""

import tempfile
from datetime import date, datetime
from zoneinfo import ZoneInfo

from idxbot.data.normalize import normalize_batch
from idxbot.data.providers.fixture import FixtureProvider
from idxbot.data.providers.base import HistoricalRequest
from idxbot.data.store import DataCompactor, PartitionedParquetStore, Provenance

JAKARTA = ZoneInfo("Asia/Jakarta")


def _sample_bars():
    p = FixtureProvider()
    rows = p.get_historical(
        HistoricalRequest(
            symbols=["BBCA", "BBRI"],
            start=date(2024, 3, 1),
            end=date(2024, 3, 31),
        )
    )
    return normalize_batch(rows, provider=p.name())


def test_partition_write_read():
    with tempfile.TemporaryDirectory() as tmp:
        store = PartitionedParquetStore(tmp)
        bars = _sample_bars()
        paths = store.write_bars(bars)
        assert paths
        # read one partition
        data = store.read_partition("BBCA", 2024, 3)
        assert len(data) > 0
        assert all(r["symbol"] == "BBCA" for r in data)


def test_delta_write_and_compact():
    with tempfile.TemporaryDirectory() as tmp:
        store = PartitionedParquetStore(tmp)
        bars = _sample_bars()
        # initial historical
        store.write_bars(bars[:5])
        # incremental delta
        delta_path = store.write_delta(bars[5:8], batch_date="2024-03-20", batch_id=1)
        assert delta_path.exists()
        # another delta same day
        store.write_delta(bars[8:10], batch_date="2024-03-20", batch_id=2)

        compactor = DataCompactor(store)
        result = compactor.compact("2024-03-20")
        assert result["status"] == "ok"
        assert result["rows"] >= 1
        assert result["delta_files"] == 2


def test_delta_does_not_rewrite_all_history():
    """Adding one delta must not require rewriting unrelated partitions."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PartitionedParquetStore(tmp)
        bars = _sample_bars()
        store.write_bars(bars)
        before = {p: p.stat().st_mtime for p in store.market_dir.rglob("data.parquet")}
        # delta for a single new-ish bar
        from idxbot.data.providers.fixture import _bar
        from idxbot.data.normalize import normalize_record

        new = normalize_record(
            _bar("BBCA", date(2024, 4, 1), 2100, 2200, 2050, 2150, 1_000_000)
        )
        store.write_delta([new], batch_date="2024-04-01", batch_id=1)
        DataCompactor(store).compact("2024-04-01")
        after = {p: p.stat().st_mtime for p in store.market_dir.rglob("data.parquet")}
        # March partitions should be unchanged
        for p, m in before.items():
            if "month=03" in str(p):
                assert after.get(p) == m


def test_dedupe_on_rewrite():
    with tempfile.TemporaryDirectory() as tmp:
        store = PartitionedParquetStore(tmp)
        bars = _sample_bars()[:3]
        store.write_bars(bars)
        store.write_bars(bars)  # same again
        data = store.read_partition(bars[0].symbol, bars[0].timestamp.year, bars[0].timestamp.month)
        keys = [(r["symbol"], r["timestamp"]) for r in data]
        assert len(keys) == len(set(keys))


def test_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        store = PartitionedParquetStore(tmp)
        prov = Provenance(
            provider="fixture:normal",
            retrieved_at=datetime.now(JAKARTA).isoformat(),
            symbol="BBCA",
            adjustment_mode="ADJUSTED",
            request_id="req-1",
        )
        path = store.save_provenance("bbca_hist", prov)
        assert path.exists()
        store.save_cursor({"last_successful_timestamp": "2024-03-18T16:00:00+07:00"})
        cur = store.load_cursor()
        assert cur is not None
        assert "last_successful_timestamp" in cur
