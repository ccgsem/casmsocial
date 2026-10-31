from pathlib import Path

import duckdb
import pyarrow as pa

from casmsocial.casmpop import _ObserverLogWriter


def _tick_table(tick: int, rank: int = 0) -> pa.Table:
    return pa.table({
        "run_id": pa.array(["r1", "r1"]),
        "tick": pa.array([tick, tick], pa.int32()),
        "rank": pa.array([rank, rank], pa.int32()),
        "agent_id": pa.array([1, 2], pa.int64()),
    })


def _files(base: Path) -> list[str]:
    return sorted(str(p.relative_to(base)) for p in base.rglob("*.parquet"))


def _read(base: Path) -> list[tuple]:
    return duckdb.sql(
        f"SELECT run_id, tick, rank, agent_id FROM read_parquet('{base}/**/*.parquet', hive_partitioning=true) "
        "ORDER BY tick, rank, agent_id"
    ).fetchall()


def test_default_writes_one_file_per_tick_in_tick_partitions(tmp_path):
    base = tmp_path / "agent_log.parquet"
    writer = _ObserverLogWriter(str(base), {})
    for tick in (60, 120, 180):
        writer.write(_tick_table(tick))
    writer.flush()

    assert _files(base) == [
        "run_id=r1/tick=120/rank=0/part-0.parquet",
        "run_id=r1/tick=180/rank=0/part-0.parquet",
        "run_id=r1/tick=60/rank=0/part-0.parquet",
    ]


def test_no_tick_partition_buffers_flush_ticks_per_file(tmp_path):
    base = tmp_path / "agent_log.parquet"
    params = {"observers.partition_by_tick": False, "observers.flush_ticks": 2}
    writer = _ObserverLogWriter(str(base), params)
    for tick in (60, 120, 180):
        writer.write(_tick_table(tick))
    # Two ticks flushed on reaching flush_ticks; the third waits for on_end.
    assert _files(base) == ["run_id=r1/rank=0/part-t60-0.parquet"]
    writer.flush()

    assert _files(base) == [
        "run_id=r1/rank=0/part-t180-0.parquet",
        "run_id=r1/rank=0/part-t60-0.parquet",
    ]
    rows = _read(base)
    assert [row[1] for row in rows] == [60, 60, 120, 120, 180, 180]
    assert {row[0] for row in rows} == {"r1"}


def test_layouts_read_back_identically(tmp_path):
    default_base = tmp_path / "default"
    compact_base = tmp_path / "compact"
    default = _ObserverLogWriter(str(default_base), {})
    compact = _ObserverLogWriter(
        str(compact_base), {"observers.partition_by_tick": "false", "observers.flush_ticks": 10}
    )
    for tick in (60, 120):
        for rank in (0, 1):
            default.write(_tick_table(tick, rank))
            compact.write(_tick_table(tick, rank))
    default.flush()
    compact.flush()

    assert _read(default_base) == _read(compact_base)
    assert len(_files(default_base)) == 4
    assert len(_files(compact_base)) == 2


def test_flush_without_pending_rows_writes_nothing(tmp_path):
    base = tmp_path / "agent_log.parquet"
    writer = _ObserverLogWriter(str(base), {"observers.partition_by_tick": False, "observers.flush_ticks": 5})
    writer.flush()
    assert not base.exists()
