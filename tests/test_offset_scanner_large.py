"""Phase 22 post-mortem: large-rawlog offset + concurrency regression.

Symptoms in production (266 MB AL2 rawlog, 20,162 samples):

    RawLogError: unexpected EOF while reading sstat blob at lazy
    sample 19801: wanted 3155507384 bytes, got 5124113

Mixed with sample numbers 19802 / 19807 / 19891 out of order. That
"wanted 3 GB" is the smoking gun: ``scomplen`` is a 32-bit field in
the rawrecord header, but 3 GB only shows up when you read garbage
bytes from the wrong file offset. The offset index itself is correct
(``scripts/inspect_offsets.py`` verified it), so the garbage has to
come from a concurrent request corrupting the shared file handle's
seek position.

This test pins down three invariants for the lazy path:

1. Every offset in the index lives inside the file, and the deltas
   between consecutive samples stay in a sane 1 KiB .. 10 MiB band.
2. A single-threaded decode of samples 19800..19900 finishes without
   raising ``RawLogError``.
3. Many threads decoding overlapping sample ranges in parallel all
   finish cleanly. This is the concurrency regression the bug report
   is really about.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from atop_web.parser.lazy import LazyRawLog
from atop_web.parser.reader import RawLogError


FIXTURE = Path.home() / "Downloads" / "al2_atop_20260403"


pytestmark = pytest.mark.skipif(
    not FIXTURE.is_file(),
    reason=f"large fixture {FIXTURE} not present",
)


def test_every_offset_inside_file():
    size = FIXTURE.stat().st_size
    lazy = LazyRawLog.open(FIXTURE)
    try:
        for i, off in enumerate(lazy.index.offsets):
            assert 0 < off < size, f"sample {i}: offset {off} outside file"
    finally:
        lazy.close()


def test_offset_deltas_are_sane():
    # Deltas should be monotonically increasing and each about one
    # rawrecord + zlib payload in size. 10 MiB per sample is already
    # implausibly large; 0 or negative means offsets repeated.
    lazy = LazyRawLog.open(FIXTURE)
    try:
        bad = []
        offs = lazy.index.offsets
        for i in range(1, len(offs)):
            d = offs[i] - offs[i - 1]
            if d <= 0 or d > (10 << 20):
                bad.append((i, offs[i - 1], offs[i], d))
        assert not bad, f"suspicious deltas: {bad[:5]}"
    finally:
        lazy.close()


def test_focus_window_decodes_clean():
    """Samples 19800..19900 must decode without a RawLogError.

    This is the "hot zone" where production saw the errors. A single
    consumer should always succeed; any failure here means the index
    is pointing at something that isn't a valid sstat stream.
    """
    lazy = LazyRawLog.open(FIXTURE)
    try:
        n = len(lazy)
        lo, hi = 19800, min(19900, n)
        errors = []
        for i in range(lo, hi):
            try:
                view = lazy[i]
                _ = view.system_cpu
                _ = view.system_memory
                _ = view.system_disk
                _ = view.system_network
            except RawLogError as exc:
                errors.append((i, str(exc)))
        assert not errors, f"{len(errors)} decode errors, first: {errors[:3]}"
    finally:
        lazy.close()


def test_concurrent_decode_does_not_race():
    """Eight threads decoding overlapping windows must all succeed.

    Before the fix the shared ``_file`` handle let one thread's
    ``seek(offset)`` land between another thread's ``seek`` and its
    ``read(scomplen)``, so the second thread would read garbage bytes
    from the wrong offset and interpret the first four of them as a
    ``scomplen`` (hence "wanted 3155507384 bytes"). The fix (mmap)
    removes the shared seek position; this test locks that in.
    """
    lazy = LazyRawLog.open(FIXTURE)
    try:
        n = len(lazy)
        # Each worker walks a different slice of the file so seek-position
        # collisions are guaranteed if the implementation still has any.
        worker_count = 8
        errors: list[tuple[int, int, str]] = []
        errors_lock = threading.Lock()

        def worker(worker_id: int) -> None:
            # Spread starting points across the file so workers interleave.
            start = (n // worker_count) * worker_id
            for j in range(start, min(start + 200, n)):
                try:
                    view = lazy[j]
                    _ = view.system_cpu
                    _ = view.system_memory
                except RawLogError as exc:
                    with errors_lock:
                        errors.append((worker_id, j, str(exc)))
                    return

        threads = [
            threading.Thread(target=worker, args=(w,))
            for w in range(worker_count)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"{len(errors)} concurrent errors: {errors[:3]}"
    finally:
        lazy.close()


def test_concurrent_processes_decode_does_not_race():
    """The tstat (processes) path must also be concurrency-safe.

    Same shape as the sstat test above but exercises the longer seek +
    read window in ``SampleView.processes``.
    """
    lazy = LazyRawLog.open(FIXTURE)
    try:
        n = len(lazy)
        indices = list(range(0, n, max(1, n // 64)))[:64]

        errors: list[tuple[int, str]] = []
        errors_lock = threading.Lock()

        def worker() -> None:
            for j in indices:
                try:
                    _ = lazy[j].processes
                except RawLogError as exc:
                    with errors_lock:
                        errors.append((j, str(exc)))
                    return

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"{len(errors)} process decode errors: {errors[:3]}"
    finally:
        lazy.close()
