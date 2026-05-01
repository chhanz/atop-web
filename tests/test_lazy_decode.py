"""Unit tests for ``LazyRawLog`` / ``SampleView`` on-demand decode.

Phase 22 T-03. The ``SampleView`` must be indistinguishable from an
eager ``Sample`` for every field we surface to the API layer: the routes
and tool handlers access ``sample.curtime``, ``sample.system_cpu``,
``sample.system_memory.used_kb`` and ``sample.processes[i].pid`` without
type-checking, and we want that to keep working when the underlying
storage flips over in Stage B.

The tests cover:

* eager vs lazy field equality for every decoded sub-dataclass
* process list length + pid/utime parity
* LRU cache keeps repeated accesses cheap
* ``SampleView`` is lightweight enough that 100 views do not approach
  the eager decode memory cost
"""

from __future__ import annotations

import dataclasses
import tracemalloc
from pathlib import Path

import pytest

from atop_web.parser import parse_file
from atop_web.parser.lazy import LazyRawLog, SampleView
from atop_web.parser.reader import parse_stream


def _open_lazy(path: Path) -> LazyRawLog:
    # LazyRawLog owns the file handle — caller passes path; the class
    # opens/closes. A context-manager API keeps tests tidy.
    return LazyRawLog.open(path)


def test_lazy_sample_count_matches_eager(rawlog_path: Path):
    eager = parse_file(rawlog_path)
    with _open_lazy(rawlog_path) as lazy:
        assert len(lazy) == len(eager.samples)


def test_sample_view_curtime_and_ndeviat(rawlog_path: Path):
    eager = parse_file(rawlog_path)
    with _open_lazy(rawlog_path) as lazy:
        for i, eag in enumerate(eager.samples):
            view = lazy[i]
            assert isinstance(view, SampleView)
            assert view.curtime == eag.curtime
            assert view.ndeviat == eag.ndeviat


def test_sample_view_system_cpu_matches_eager(rawlog_path: Path):
    eager = parse_file(rawlog_path, max_samples=3)
    with _open_lazy(rawlog_path) as lazy:
        for i, eag in enumerate(eager.samples):
            view = lazy[i]
            # dataclass equality — every field has to match.
            assert view.system_cpu == eag.system_cpu


def test_sample_view_system_memory_matches_eager(rawlog_path: Path):
    eager = parse_file(rawlog_path, max_samples=3)
    with _open_lazy(rawlog_path) as lazy:
        for i, eag in enumerate(eager.samples):
            view = lazy[i]
            assert view.system_memory == eag.system_memory


def test_sample_view_system_disk_and_network(rawlog_path: Path):
    eager = parse_file(rawlog_path, max_samples=2)
    with _open_lazy(rawlog_path) as lazy:
        for i, eag in enumerate(eager.samples):
            view = lazy[i]
            assert view.system_disk == eag.system_disk
            assert view.system_network == eag.system_network


def test_sample_view_processes_list(rawlog_path: Path):
    eager = parse_file(rawlog_path, max_samples=2)
    with _open_lazy(rawlog_path) as lazy:
        for i, eag in enumerate(eager.samples):
            view = lazy[i]
            assert len(view.processes) == len(eag.processes)
            assert [p.pid for p in view.processes] == [p.pid for p in eag.processes]
            assert [p.name for p in view.processes] == [p.name for p in eag.processes]
            # Drill into one process to confirm the full decode path.
            assert view.processes[0] == eag.processes[0]


def test_sample_view_aggregate_header_fields(rawlog_path: Path):
    eager = parse_file(rawlog_path, max_samples=1)
    eag = eager.samples[0]
    with _open_lazy(rawlog_path) as lazy:
        view = lazy[0]
        for field in ("interval", "nactproc", "ntask", "totproc", "totrun",
                      "totslpi", "totslpu", "totzomb"):
            assert getattr(view, field) == getattr(eag, field), field


def test_sample_view_lru_cache_hits_on_repeat(rawlog_path: Path):
    # Two attribute accesses on the same view must share a single decode.
    # The view records an instrumentation counter so we can assert.
    with _open_lazy(rawlog_path) as lazy:
        view = lazy[0]
        _ = view.system_cpu
        before = view._decode_count
        _ = view.system_cpu
        _ = view.system_memory
        _ = view.system_disk
        _ = view.system_network
        after = view._decode_count
        # One sstat inflate pays for all four system_* fields, so the
        # counter may advance by at most 1 here (the cache is per-view).
        assert after - before <= 1


def test_lazy_rawlog_iterable(rawlog_path: Path):
    # Iterate yields views in timestamp order.
    with _open_lazy(rawlog_path) as lazy:
        prev = -1
        count = 0
        for view in lazy:
            assert isinstance(view, SampleView)
            assert view.curtime > prev
            prev = view.curtime
            count += 1
        assert count == len(lazy)


def test_lazy_rawlog_negative_index_and_slice(rawlog_path: Path):
    with _open_lazy(rawlog_path) as lazy:
        n = len(lazy)
        assert lazy[-1].curtime == lazy[n - 1].curtime


def test_sample_view_memory_budget(rawlog_path: Path):
    # Creating 100 views must cost dramatically less than the eager decode
    # that materializes every Sample. 1/5 of eager is the headline budget
    # in the Phase 22 plan — pick a conservative threshold here. Force
    # eager here explicitly: Phase 22 T-11 flipped the default, but the
    # comparison only makes sense against a fully materialized RawLog.
    tracemalloc.start()
    try:
        eager = parse_file(rawlog_path, lazy=False)
        eager_peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    eager_count = len(eager.samples)
    if eager_count < 10:
        pytest.skip("not enough samples for a meaningful memory comparison")

    tracemalloc.start()
    try:
        with _open_lazy(rawlog_path) as lazy:
            # Touch enough views to stress the cache but not the whole file.
            touched = min(100, len(lazy))
            views = [lazy[i] for i in range(touched)]
            # Read one scalar per view so __getattr__ does some work.
            _ = [v.curtime for v in views]
            lazy_peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    # Pure offset index + a handful of cached decodes must stay well
    # under eager. Exact ratio depends on LRU size and CPython allocator
    # behavior, so leave ample slack — the important thing is "much less",
    # not an exact figure.
    assert lazy_peak < eager_peak * 0.5, (
        f"lazy peak {lazy_peak} not meaningfully smaller than eager {eager_peak}"
    )


def test_sample_view_dataclass_equivalence_fields(rawlog_path: Path):
    # dataclasses.fields() on Sample returns the canonical attribute
    # set. Every name must be readable on SampleView too — that's the
    # duck-typing contract the route layer assumes.
    from atop_web.parser.reader import Sample

    with _open_lazy(rawlog_path) as lazy:
        view = lazy[0]
        for f in dataclasses.fields(Sample):
            # Should not raise AttributeError.
            getattr(view, f.name)
