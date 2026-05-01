"""Unit tests for ``SampleIndex`` and ``parse_stream(..., lazy=True)``.

Phase 22 T-02. The index is the on-disk map that lets later stages decode
any single sample on demand without re-walking the file. These tests lock
down the invariants downstream code depends on:

* length equals the eager sample count
* per-sample ``curtime`` / ``ndeviat`` match eager
* ``array.array`` backing storage gives a tight memory footprint
  (target: under 40 bytes/sample including typed-array overhead)
* ``slice_by_time`` is bisect-based and correct at the range edges
* ``parse_stream(stream, lazy=True)`` returns a ``RawLog`` whose
  ``samples`` list is empty but whose ``index`` is populated — this is
  the knob Stage B wires into the session store.
"""

from __future__ import annotations

import array
import io
from pathlib import Path

import pytest

from atop_web.parser import parse_file
from atop_web.parser.index import SampleIndex
from atop_web.parser.reader import parse_stream


def _lazy_from_path(path: Path):
    with path.open("rb") as fh:
        return parse_stream(fh, lazy=True)


def test_lazy_parse_returns_index_and_empty_samples(rawlog_path: Path):
    # Lazy mode: samples list is empty, index is populated. This is the
    # contract the session store (T-05) relies on to decide which code
    # path to follow.
    lazy = _lazy_from_path(rawlog_path)
    assert lazy.samples == [], "lazy mode must not materialize Sample objects"
    assert lazy.index is not None
    assert isinstance(lazy.index, SampleIndex)
    assert len(lazy.index) > 0


def test_index_timestamps_match_eager(rawlog_path: Path):
    eager = parse_file(rawlog_path)
    lazy = _lazy_from_path(rawlog_path)
    assert len(lazy.index) == len(eager.samples)
    for i, sam in enumerate(eager.samples):
        assert lazy.index.timestamps[i] == sam.curtime, f"sample {i}"
        assert lazy.index.ndeviats[i] == sam.ndeviat, f"sample {i}"


def test_index_offsets_are_ordered_and_within_file(rawlog_path: Path):
    lazy = _lazy_from_path(rawlog_path)
    size = rawlog_path.stat().st_size
    offsets = list(lazy.index.offsets)
    assert offsets == sorted(offsets), "offsets must be monotonic"
    assert offsets[0] > 0, "first offset is past the rawheader"
    assert offsets[-1] < size


def test_index_uses_array_array_backing(rawlog_path: Path):
    lazy = _lazy_from_path(rawlog_path)
    idx = lazy.index
    # The index is hot — it is walked on every range query. array.array
    # gives us C-packed storage; a plain Python list here would blow the
    # memory budget on large files.
    assert isinstance(idx.offsets, array.array)
    assert isinstance(idx.timestamps, array.array)
    assert isinstance(idx.scomplens, array.array)
    assert isinstance(idx.pcomplens, array.array)
    assert isinstance(idx.ndeviats, array.array)


def test_index_memory_footprint_budget(rawlog_path: Path):
    # Budget: under 40 bytes per sample for the typed-array storage.
    # At n=17k this is 680 KB which is the ceiling we promised in the
    # Phase 22 plan.
    lazy = _lazy_from_path(rawlog_path)
    n = len(lazy.index)
    assert n > 0
    footprint = lazy.index.mem_bytes()
    per_sample = footprint / n
    assert per_sample < 40.0, (
        f"per-sample index cost too high: {per_sample:.1f} B "
        f"(total {footprint} B for {n} samples)"
    )


def test_slice_by_time_returns_closed_interval(rawlog_path: Path):
    eager = parse_file(rawlog_path)
    lazy = _lazy_from_path(rawlog_path)
    assert len(eager.samples) >= 3

    # Pick a closed interval covering the middle third of the samples —
    # both endpoints should round to existing samples.
    lo = eager.samples[1].curtime
    hi = eager.samples[-2].curtime
    start, end = lazy.index.slice_by_time(lo, hi)

    assert start == 1
    assert end == len(eager.samples) - 1  # exclusive upper
    expected_times = [s.curtime for s in eager.samples[1 : len(eager.samples) - 1]]
    got_times = list(lazy.index.timestamps[start:end])
    assert got_times == expected_times


def test_slice_by_time_empty_window():
    # Range with no matching samples: both indices collapse to the same
    # value (empty slice).
    idx = SampleIndex(
        offsets=array.array("q", [100, 200, 300]),
        timestamps=array.array("q", [10, 20, 30]),
        scomplens=array.array("I", [1, 1, 1]),
        pcomplens=array.array("I", [2, 2, 2]),
        ndeviats=array.array("I", [5, 5, 5]),
        spec=None,
    )
    start, end = idx.slice_by_time(100, 200)
    assert start == end
    # Boundary: single point query grabs exactly that sample when present.
    start, end = idx.slice_by_time(20, 20)
    assert start == 1
    assert end == 2


def test_index_iteration_yields_offsets_and_times(rawlog_path: Path):
    lazy = _lazy_from_path(rawlog_path)
    # Iterating the index returns (offset, curtime, scomplen, pcomplen,
    # ndeviat) tuples — enough for a decoder to reconstruct a SampleView
    # without needing to index back into the arrays.
    rows = list(lazy.index)
    assert len(rows) == len(lazy.index)
    first = rows[0]
    assert first[0] == lazy.index.offsets[0]
    assert first[1] == lazy.index.timestamps[0]
    assert first[2] == lazy.index.scomplens[0]
    assert first[3] == lazy.index.pcomplens[0]
    assert first[4] == lazy.index.ndeviats[0]


def test_eager_mode_still_works_unchanged(rawlog_path: Path):
    # The lazy switch must not have regressed eager parsing — the eager
    # RawLog still has samples and still has no index.
    eager = parse_file(rawlog_path, max_samples=2)
    assert len(eager.samples) == 2
    assert eager.samples[0].curtime > 0


def test_index_holds_version_spec(rawlog_path: Path):
    # Consumers (T-03 SampleView) reach back through the index to find
    # the right spec for decoding payloads — version info must travel
    # together with the offsets.
    lazy = _lazy_from_path(rawlog_path)
    assert lazy.index.spec is not None
    assert lazy.index.spec.name in ("atop_2_12", "atop_2_7")
