"""Unit tests for the Phase 20 tool handlers.

We exercise each tool against the small fixture rawlog so we can verify
shape and units without needing a live Bedrock endpoint. The tests
deliberately do not rely on specific metric values (those depend on the
capture) but do assert the contract the chat router relies on:

* every tool returns JSON serializable dicts,
* timestamps are ISO8601 UTC strings,
* raw counters like ``cpu_ticks`` never leak out.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atop_web.llm import tools
from atop_web.parser.reader import parse_file


@pytest.fixture(scope="module")
def rawlog(rawlog_path: Path):
    return parse_file(rawlog_path, max_samples=30)


@pytest.fixture(scope="module")
def specs(rawlog):
    return tools.build_tool_specs(rawlog)


@pytest.fixture(scope="module")
def by_name(specs):
    return {s.name: s for s in specs}


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_build_tool_specs_has_seven_tools(specs):
    names = {s.name for s in specs}
    assert names == {
        "get_metric_stats",
        "get_top_processes",
        "find_spikes",
        "get_process_count",
        "get_samples_in_range",
        "get_capture_info",
        "compare_ranges",
    }
    for s in specs:
        assert s.description
        assert isinstance(s.input_schema, dict)
        assert s.input_schema.get("type") == "object"


def test_tool_results_are_json_serializable(by_name, rawlog):
    cap = by_name["get_capture_info"].call({})
    assert json.dumps(cap)
    assert "." not in str(cap["sample_count"])  # integer


# ---------------------------------------------------------------------------
# 1. get_metric_stats
# ---------------------------------------------------------------------------


def test_get_metric_stats_cpu(by_name):
    out = by_name["get_metric_stats"].call({"metric": "cpu"})
    assert out["unit"] == "percent"
    assert out["count"] > 0
    assert out["max"]["at"].endswith("Z")
    assert out["min"]["at"].endswith("Z")
    assert out["avg"] >= 0
    # Never leak raw counter names.
    assert "cpu_ticks" not in json.dumps(out)
    assert "utime" not in json.dumps(out)


def test_get_metric_stats_rejects_unknown_metric(by_name):
    out = by_name["get_metric_stats"].call({"metric": "pikachu"})
    assert "error" in out


def test_get_metric_stats_out_of_range_returns_note(by_name, rawlog):
    # Pick a range that cannot possibly overlap the capture.
    out = by_name["get_metric_stats"].call(
        {
            "metric": "cpu",
            "start": "1990-01-01T00:00:00Z",
            "end": "1990-01-01T00:10:00Z",
        }
    )
    assert out["count"] == 0


# ---------------------------------------------------------------------------
# 2. get_top_processes
# ---------------------------------------------------------------------------


def test_get_top_processes_cpu_returns_percent(by_name):
    out = by_name["get_top_processes"].call({"metric": "cpu", "limit": 3})
    assert out["metric"] == "cpu"
    assert len(out["processes"]) <= 3
    for row in out["processes"]:
        assert "cpu_pct" in row
        assert "cpu_ticks" not in row  # raw counter must not leak
        assert isinstance(row["cpu_pct"], (int, float))


def test_get_top_processes_rss_returns_mib(by_name):
    out = by_name["get_top_processes"].call({"metric": "rss", "limit": 3})
    for row in out["processes"]:
        assert "rss_mib" in row
        assert "rmem_kb" not in row


def test_get_top_processes_unknown_metric(by_name):
    out = by_name["get_top_processes"].call({"metric": "bogus"})
    assert "error" in out


# ---------------------------------------------------------------------------
# 3. find_spikes
# ---------------------------------------------------------------------------


def test_find_spikes_returns_iso_windows(by_name):
    out = by_name["find_spikes"].call({"metric": "cpu", "window_seconds": 120})
    assert out["metric"] == "cpu"
    for spike in out["spikes"]:
        assert spike["start"].endswith("Z")
        assert spike["end"].endswith("Z")
        assert spike["center"].endswith("Z")


def test_find_spikes_respects_threshold(by_name):
    # A very high threshold should yield zero spikes without erroring.
    out = by_name["find_spikes"].call(
        {"metric": "cpu", "threshold_pct": 10_000}
    )
    assert out["spike_count"] == 0
    assert out["spikes"] == []


# ---------------------------------------------------------------------------
# 4. get_process_count
# ---------------------------------------------------------------------------


def test_get_process_count_total(by_name):
    out = by_name["get_process_count"].call({})
    assert out["count_max"] >= out["count_min"] >= 0
    assert out["samples"] > 0


def test_get_process_count_glob(by_name):
    out = by_name["get_process_count"].call({"pattern": "*"})
    # Glob ``*`` matches every process, so count should equal total.
    assert out["count_max"] == out["total_process_max"]


# ---------------------------------------------------------------------------
# 5. get_samples_in_range
# ---------------------------------------------------------------------------


def test_get_samples_in_range_caps_rows(by_name, rawlog):
    s, e = rawlog.samples[0].curtime, rawlog.samples[-1].curtime
    out = by_name["get_samples_in_range"].call(
        {
            "start": s,
            "end": e,
            "metrics": ["cpu", "mem"],
        }
    )
    assert out["sampled_rows"] <= 60
    for row in out["rows"]:
        assert row["at"].endswith("Z")


def test_get_samples_in_range_requires_bounds(by_name):
    out = by_name["get_samples_in_range"].call({})
    assert "error" in out


# ---------------------------------------------------------------------------
# 6. get_capture_info
# ---------------------------------------------------------------------------


def test_get_capture_info_shape(by_name, rawlog):
    info = by_name["get_capture_info"].call({})
    assert info["start"].endswith("Z")
    assert info["end"].endswith("Z")
    assert info["sample_count"] == len(rawlog.samples)
    assert info["ncpu"] >= 1


# ---------------------------------------------------------------------------
# 7. compare_ranges
# ---------------------------------------------------------------------------


def test_compare_ranges_computes_delta(by_name, rawlog):
    mid = rawlog.samples[len(rawlog.samples) // 2].curtime
    start = rawlog.samples[0].curtime
    end = rawlog.samples[-1].curtime
    out = by_name["compare_ranges"].call(
        {
            "metric": "cpu",
            "range_a": {"start": start, "end": mid},
            "range_b": {"start": mid, "end": end},
        }
    )
    assert out["metric"] == "cpu"
    assert "avg" in out["range_a"]
    assert "avg" in out["range_b"]
    assert "avg_delta" in out["delta"]
