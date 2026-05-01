"""Unit tests for the pre-aggregate downsample cache.

Phase 22 T-04. The aggregate cache exists so the chart endpoints can
answer "give me the last 24 h at ~1-minute resolution" without pulling
every sample through lazy decode. The builder scans the index once,
decodes each sample via the lazy path, and accumulates bucketed values
on a 1-minute / 5-minute / 1-hour grid.

The chart metrics we care about (and that the plan calls out):

* ``cpu``      — system-wide CPU busy (utime + stime + ntime + ...) in ticks,
                 normalized by hertz + nrcpu to give a 0-1 busy fraction.
* ``mem_used`` — physmem - freemem - cachemem - buffermem - slabmem, in pages.
* ``mem_available`` — availablemem when present, else None.
* ``swap_used`` — totswap - freeswap, in pages.
* ``disk_read_sectors`` / ``disk_write_sectors`` — summed across all
  physical disks.
* ``net_rx_bytes`` / ``net_tx_bytes`` — summed across all interfaces.

Tests exercise:

* 1-minute grid exactly averages samples inside each minute bucket.
* Builder footprint stays small (hundreds of KB for 17k samples).
* ``lookup`` returns (hit=True, series) when the window is wholly
  covered by a grid, and (hit=False, None) otherwise so the chart
  route can fall back to lazy decode.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from atop_web.parser.aggregate import Aggregate, build_aggregate
from atop_web.parser.lazy import LazyRawLog


def _open_lazy(path: Path) -> LazyRawLog:
    return LazyRawLog.open(path)


def test_aggregate_builds_one_minute_grid(rawlog_path: Path):
    with _open_lazy(rawlog_path) as lazy:
        agg = build_aggregate(lazy)
        assert isinstance(agg, Aggregate)
        # The 1-minute grid must always be populated — the plan promises
        # it, so chart code can rely on it.
        assert "1m" in agg.grids
        grid_1m = agg.grids["1m"]
        # At least one bucket — the fixture covers multiple minutes.
        assert len(grid_1m.bucket_starts) >= 1


def test_aggregate_bucket_timestamps_are_minute_aligned(rawlog_path: Path):
    with _open_lazy(rawlog_path) as lazy:
        agg = build_aggregate(lazy)
    grid = agg.grids["1m"]
    for t in grid.bucket_starts:
        assert t % 60 == 0, f"1m bucket start {t} not minute-aligned"


def test_aggregate_mem_used_matches_direct_compute(rawlog_path: Path):
    # Cross-check one bucket against the value we compute directly from
    # the lazy decode. No tolerance — integer arithmetic throughout.
    with _open_lazy(rawlog_path) as lazy:
        agg = build_aggregate(lazy)
        grid = agg.grids["1m"]
        # Pick the middle bucket to stay away from partial-window edges.
        if len(grid.bucket_starts) < 3:
            pytest.skip("need at least 3 buckets for this check")
        bi = len(grid.bucket_starts) // 2
        t0 = grid.bucket_starts[bi]
        t1 = t0 + 60

        sum_used = 0
        count = 0
        for view in lazy:
            if t0 <= view.curtime < t1:
                mem = view.system_memory
                if mem is None:
                    continue
                sum_used += (
                    mem.physmem - mem.freemem - mem.cachemem - mem.buffermem - mem.slabmem
                )
                count += 1
        if count == 0:
            pytest.skip("middle bucket had no decoded memory samples")
        expected = sum_used // count  # integer mean, matches builder's policy
    got = grid.series["mem_used"][bi]
    assert got == expected, (
        f"mem_used bucket {bi} at t={t0}: expected {expected}, got {got}"
    )


def test_aggregate_cpu_busy_bounded(rawlog_path: Path):
    # CPU busy is a fraction in 0..1. Per bucket averaging must keep it
    # within that range on a real fixture.
    with _open_lazy(rawlog_path) as lazy:
        agg = build_aggregate(lazy)
    cpu = agg.grids["1m"].series["cpu_busy"]
    assert cpu, "expected some cpu_busy samples"
    for v in cpu:
        if v is None:
            continue
        assert 0.0 <= v <= 1.0 + 1e-6, f"cpu_busy out of range: {v}"


def test_aggregate_footprint_small(rawlog_path: Path):
    # Plan budget: under 500 KB total. The parallel-array backing is the
    # whole point; a plain list-of-dicts would blow this.
    with _open_lazy(rawlog_path) as lazy:
        agg = build_aggregate(lazy)
    assert agg.bytes_footprint() < 500 * 1024, agg.bytes_footprint()


def test_aggregate_metrics_surfaced(rawlog_path: Path):
    with _open_lazy(rawlog_path) as lazy:
        agg = build_aggregate(lazy)
    expected = {
        "cpu_busy",
        "mem_used",
        "mem_available",
        "swap_used",
        "disk_read_sectors",
        "disk_write_sectors",
        "net_rx_bytes",
        "net_tx_bytes",
    }
    assert expected.issubset(set(agg.grids["1m"].series.keys()))


def test_aggregate_lookup_hit_on_whole_range(rawlog_path: Path):
    with _open_lazy(rawlog_path) as lazy:
        agg = build_aggregate(lazy)
        start = lazy.index.timestamps[0]
        end = lazy.index.timestamps[-1]
    result = agg.lookup("cpu_busy", "1m", start, end)
    assert result.hit is True
    assert result.series is not None
    assert len(result.series) == len(result.bucket_starts)


def test_aggregate_lookup_miss_on_sub_bucket_range(rawlog_path: Path):
    with _open_lazy(rawlog_path) as lazy:
        agg = build_aggregate(lazy)
        # Pick a window that is narrower than the grid (e.g., 10 seconds)
        # so the aggregate cannot answer it — the route must fall back to
        # lazy decode.
        start = lazy.index.timestamps[0]
        end = start + 10
    result = agg.lookup("cpu_busy", "1m", start, end)
    # A 10-second window inside a 60-second bucket is a miss — the
    # aggregate cannot answer a sub-bucket query faithfully.
    assert result.hit is False


def test_aggregate_lookup_unknown_metric(rawlog_path: Path):
    with _open_lazy(rawlog_path) as lazy:
        agg = build_aggregate(lazy)
        start = lazy.index.timestamps[0]
        end = lazy.index.timestamps[-1]
    result = agg.lookup("does_not_exist", "1m", start, end)
    assert result.hit is False


def test_aggregate_respects_grid_argument(rawlog_path: Path):
    with _open_lazy(rawlog_path) as lazy:
        agg = build_aggregate(lazy)
    # 5m and 1h grids are populated only when enough samples; 1m is
    # always populated. Just assert presence — the fixture spans enough
    # time for the bigger grids when atop interval is typical (10 s).
    assert "1m" in agg.grids
    for name in ("5m", "1h"):
        if name in agg.grids:
            grid = agg.grids[name]
            step = {"5m": 300, "1h": 3600}[name]
            for t in grid.bucket_starts:
                assert t % step == 0, f"{name} bucket {t} misaligned"


def test_aggregate_preserves_index_len_ordering(rawlog_path: Path):
    with _open_lazy(rawlog_path) as lazy:
        agg = build_aggregate(lazy)
    for grid_name, grid in agg.grids.items():
        # Bucket starts are strictly increasing.
        starts = list(grid.bucket_starts)
        assert starts == sorted(starts)
        assert len(set(starts)) == len(starts), f"{grid_name} has duplicate starts"
