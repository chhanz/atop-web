"""T-09 Tool handlers lazy parity.

Each of the seven ``build_tool_specs`` handlers must return the same
result when given an eager ``RawLog`` vs a lazy ``LazyRawLog`` bound
to the same rawlog. Float values are compared with a tight tolerance
(1e-9 rel/abs) since both paths run the exact same arithmetic — any
drift means a divergence we want to catch.

Strings (ISO8601 timestamps, metric names) must match exactly.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from atop_web.llm import tools
from atop_web.parser import parse_file
from atop_web.parser.lazy import LazyRawLog


def _assert_equal(a, b, path="root"):
    """Deep equality with a 1e-9 tolerance on floats."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (
            f"{path}: {a!r} != {b!r}"
        )
        return
    if isinstance(a, dict):
        assert isinstance(b, dict), f"{path}: not dict: {type(b)}"
        assert set(a.keys()) == set(b.keys()), (
            f"{path}: keys differ: {set(a.keys())} vs {set(b.keys())}"
        )
        for k in a:
            _assert_equal(a[k], b[k], f"{path}.{k}")
        return
    if isinstance(a, list):
        assert isinstance(b, list), f"{path}: not list: {type(b)}"
        assert len(a) == len(b), f"{path}: len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_equal(x, y, f"{path}[{i}]")
        return
    assert a == b, f"{path}: {a!r} != {b!r}"


@pytest.fixture(scope="module")
def eager_rawlog(rawlog_path: Path):
    return parse_file(rawlog_path)


@pytest.fixture(scope="module")
def lazy_rawlog(rawlog_path: Path):
    lz = LazyRawLog.open(rawlog_path)
    yield lz
    lz.close()


@pytest.fixture(scope="module")
def specs(eager_rawlog, lazy_rawlog):
    eager_specs = {s.name: s for s in tools.build_tool_specs(eager_rawlog)}
    lazy_specs = {s.name: s for s in tools.build_tool_specs(lazy_rawlog)}
    return eager_specs, lazy_specs


def _iso_bounds(eager_rawlog):
    s = eager_rawlog.samples
    if len(s) < 2:
        return None, None
    from datetime import datetime, timezone

    fmt = lambda t: datetime.fromtimestamp(t, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return fmt(s[1].curtime), fmt(s[-2].curtime)


def test_get_capture_info_parity(specs):
    e, l = specs
    _assert_equal(e["get_capture_info"].call({}), l["get_capture_info"].call({}))


@pytest.mark.parametrize("metric", ["cpu", "mem", "disk", "net", "load1", "load5", "load15"])
def test_get_metric_stats_parity(specs, metric):
    e, l = specs
    args = {"metric": metric}
    _assert_equal(e["get_metric_stats"].call(args), l["get_metric_stats"].call(args))


def test_get_metric_stats_with_range(specs, eager_rawlog):
    e, l = specs
    lo, hi = _iso_bounds(eager_rawlog)
    if lo is None:
        pytest.skip("need 2+ samples")
    args = {"metric": "cpu", "start": lo, "end": hi}
    _assert_equal(e["get_metric_stats"].call(args), l["get_metric_stats"].call(args))


@pytest.mark.parametrize("metric", ["cpu", "rss", "mem", "disk", "net"])
def test_get_top_processes_parity(specs, metric):
    e, l = specs
    args = {"metric": metric, "limit": 10}
    _assert_equal(e["get_top_processes"].call(args), l["get_top_processes"].call(args))


def test_get_top_processes_with_range(specs, eager_rawlog):
    e, l = specs
    lo, hi = _iso_bounds(eager_rawlog)
    if lo is None:
        pytest.skip("need 2+ samples")
    args = {"metric": "cpu", "limit": 5, "start": lo, "end": hi}
    _assert_equal(e["get_top_processes"].call(args), l["get_top_processes"].call(args))


@pytest.mark.parametrize("metric", ["cpu", "mem", "disk", "net"])
def test_find_spikes_parity(specs, metric):
    e, l = specs
    args = {"metric": metric, "window_seconds": 60}
    _assert_equal(e["find_spikes"].call(args), l["find_spikes"].call(args))


def test_find_spikes_with_threshold(specs):
    e, l = specs
    args = {"metric": "cpu", "threshold_pct": 10.0, "window_seconds": 120}
    _assert_equal(e["find_spikes"].call(args), l["find_spikes"].call(args))


def test_get_process_count_no_pattern(specs):
    e, l = specs
    _assert_equal(
        e["get_process_count"].call({}), l["get_process_count"].call({})
    )


def test_get_process_count_with_pattern(specs):
    e, l = specs
    args = {"pattern": "*"}
    _assert_equal(
        e["get_process_count"].call(args), l["get_process_count"].call(args)
    )


def test_get_samples_in_range_parity(specs, eager_rawlog):
    e, l = specs
    lo, hi = _iso_bounds(eager_rawlog)
    if lo is None:
        pytest.skip("need 2+ samples")
    args = {"start": lo, "end": hi, "metrics": ["cpu", "mem", "disk", "net"]}
    _assert_equal(
        e["get_samples_in_range"].call(args),
        l["get_samples_in_range"].call(args),
    )


def test_compare_ranges_parity(specs, eager_rawlog):
    e, l = specs
    if len(eager_rawlog.samples) < 6:
        pytest.skip("need 6+ samples")
    from datetime import datetime, timezone

    fmt = lambda t: datetime.fromtimestamp(t, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    s = eager_rawlog.samples
    mid = len(s) // 2
    args = {
        "range_a": {"start": fmt(s[0].curtime), "end": fmt(s[mid - 1].curtime)},
        "range_b": {"start": fmt(s[mid].curtime), "end": fmt(s[-1].curtime)},
        "metric": "cpu",
    }
    _assert_equal(
        e["compare_ranges"].call(args), l["compare_ranges"].call(args)
    )
