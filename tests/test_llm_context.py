"""Tests for the Phase 18.5 interval aware context fields.

The chat endpoint's ``<range/>`` tag feature broke silently when a
capture's sample interval was wider than the suggested range (e.g. 559s
intervals vs a 10 minute tag), so these tests pin down the guarantees
the context builder now makes:

* ``capture.interval_seconds`` is the median timestamp delta.
* ``capture.recommended_min_range_seconds`` is ``max(interval*20, 600)``.
* ``spike_candidates`` windows are widened to at least the recommended
  minimum so the model can echo them back without landing between
  samples.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atop_web.llm import context
from atop_web.parser.reader import parse_file


@pytest.fixture(scope="module")
def rawlog(rawlog_path: Path):
    return parse_file(rawlog_path, max_samples=30)


def test_build_all_context_includes_interval_seconds(rawlog):
    ctx = context.build_all_context(rawlog)
    cap = ctx["capture"]
    assert "interval_seconds" in cap
    assert "recommended_min_range_seconds" in cap
    # A real capture has positive spacing; if this is None the median
    # helper broke and every spike window would collapse.
    assert cap["interval_seconds"] is not None
    assert cap["interval_seconds"] > 0
    assert cap["recommended_min_range_seconds"] >= context.MIN_RANGE_FLOOR_SECONDS


def test_build_range_context_includes_interval_seconds(rawlog):
    samples = rawlog.samples
    ctx = context.build_range_context(
        rawlog, samples[0].curtime, samples[-1].curtime
    )
    cap = ctx["capture"]
    assert cap["interval_seconds"] is not None
    assert cap["recommended_min_range_seconds"] >= context.MIN_RANGE_FLOOR_SECONDS
    # Range mode also reports the overall capture span so the model knows
    # how much room it has to widen a tag before hitting the edges. The
    # visible fields are ISO8601 strings (to keep the model from doing
    # epoch arithmetic); the epoch aliases keep the integer bounds.
    assert cap["start_epoch"] == samples[0].curtime
    assert cap["end_epoch"] == samples[-1].curtime
    assert cap["start"].endswith("Z")
    assert cap["end"].endswith("Z")


def test_recommended_min_range_scales_with_interval():
    # Dense capture: 5s interval -> floor wins (600s).
    assert context._recommended_min_range(5) == context.MIN_RANGE_FLOOR_SECONDS
    # Sparse capture (typical of our failing log): 559s interval -> 20x
    # samples dominates the floor.
    assert context._recommended_min_range(559) == 559 * 20
    # Missing interval -> falls back to the floor; never returns 0.
    assert context._recommended_min_range(None) == context.MIN_RANGE_FLOOR_SECONDS
    assert context._recommended_min_range(0) == context.MIN_RANGE_FLOOR_SECONDS


def test_median_interval_uses_timestamp_deltas():
    # Two fake samples spaced 60 seconds apart: median is 60. We only
    # need the Sample dataclass so we synthesize minimal instances.
    from atop_web.parser.reader import Sample

    def mk(ts: int) -> Sample:
        return Sample(
            curtime=ts,
            interval=0,  # deliberately zero so only deltas are used
            ndeviat=0,
            nactproc=0,
            ntask=0,
            totproc=0,
            totrun=0,
            totslpi=0,
            totslpu=0,
            totzomb=0,
        )

    samples = [mk(100), mk(160), mk(220), mk(280)]
    assert context._median_interval_seconds(samples) == 60
    # A single sample has no delta to measure.
    assert context._median_interval_seconds([mk(100)]) is None


def test_spike_windows_are_widened_to_recommended_minimum(rawlog):
    from datetime import datetime

    ctx = context.build_all_context(rawlog)
    interval = ctx["capture"]["interval_seconds"]
    min_range = ctx["capture"]["recommended_min_range_seconds"]
    spikes = ctx.get("spike_candidates", [])
    if not spikes:
        pytest.skip("no spike candidates detected for this fixture")

    def _parse(iso: str) -> int:
        # Timestamps are pre formatted ISO8601 UTC so the model does not
        # have to do integer arithmetic; the test has to parse back to
        # compare widths.
        return int(
            datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
        )

    for spike in spikes:
        width = _parse(spike["end"]) - _parse(spike["start"])
        # Allow a tiny slack because _expand_spike_window uses integer
        # halves; the widened window must be close to min_range and at
        # least interval wide.
        assert width >= min_range - 1, spike
        assert width >= (interval or 0)
        # Center is captured explicitly so UI can show the exact moment.
        assert spike.get("center") is not None
        # Both endpoints must be ISO8601 UTC ``Z`` strings, not epochs.
        assert isinstance(spike["start"], str) and spike["start"].endswith("Z")
        assert isinstance(spike["end"], str) and spike["end"].endswith("Z")


def test_expand_spike_window_floors_on_interval():
    # With a small floor and a large interval, the helper still keeps the
    # half width at least as wide as one sample so the range cannot be
    # thinner than a single capture tick.
    s, e = context._expand_spike_window(1000, interval_seconds=600, min_range_seconds=300)
    assert e - s >= 600
