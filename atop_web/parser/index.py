"""Lazy-decode support: compact offset index for a rawlog file.

Phase 22 goal: ``RawLog.samples: list[Sample]`` unconditionally materializes
every sample as a Python dataclass graph, which blows up Python heap 10-20x
relative to the rawlog file size. This module replaces that graph with a
single ``SampleIndex`` object that stores the minimum per-sample metadata
needed to seek back into the file and decode one sample on demand.

Storage layout
--------------
All per-sample arrays use ``array.array`` with a fixed typecode (``q``
for signed 8-byte offsets/timestamps, ``I`` for unsigned 4-byte
compressed-payload sizes and ``ndeviat``). Together this comes out to
~28 bytes per sample versus the 2-4 KB that a materialized ``Sample``
plus its sub-dataclasses cost, and it stays contiguous in CPython's
native allocator rather than scattering dict+slot objects across the
Python heap.

Construction
------------
``build_sample_index`` runs one forward pass over the rawrecord headers
using the Phase-22 public ``scan_sample_offsets``. It never inflates a
zlib blob; all costs scale with ``n_samples * rawrecord_size`` (96 B per
sample), which is bounded even on 10 GB rawlogs.
"""

from __future__ import annotations

import array
import bisect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List

if TYPE_CHECKING:
    from atop_web.parser.reader import SampleOffset, VersionSpec


# Signed 8-byte for file offsets (can exceed 2 GiB on large rawlogs) and
# for curtime (unix seconds fit in 4 B but we pick ``q`` so everything
# stays aligned on 8-byte boundaries).
_OFFSET_CODE = "q"
_TIMESTAMP_CODE = "q"
# Compressed payload sizes and ``ndeviat`` are always small positive
# integers. Unsigned 4 B is plenty and halves the memory vs ``q``.
_SIZE_CODE = "I"


@dataclass(slots=True)
class SampleIndex:
    """Compact, C-packed map of one rawlog's sample layout.

    The five parallel arrays are the hot path for every range query. A
    ``spec`` reference travels with the index so the lazy view can reach
    the right decoder without plumbing the spec through every call site.
    """

    offsets: "array.array[int]"
    timestamps: "array.array[int]"
    scomplens: "array.array[int]"
    pcomplens: "array.array[int]"
    ndeviats: "array.array[int]"
    spec: "VersionSpec | None"

    def __len__(self) -> int:
        return len(self.offsets)

    def __iter__(self) -> Iterator[tuple[int, int, int, int, int]]:
        # Zip the typed arrays — yields plain Python ints, not array views.
        return zip(
            self.offsets,
            self.timestamps,
            self.scomplens,
            self.pcomplens,
            self.ndeviats,
        )

    def mem_bytes(self) -> int:
        """Total native-storage bytes held by the typed arrays.

        Does not count Python object overhead (the dataclass header, the
        ``array.array`` headers — a few hundred bytes total, amortized).
        This is the number the memory-budget tests assert on.
        """
        return (
            self.offsets.buffer_info()[1] * self.offsets.itemsize
            + self.timestamps.buffer_info()[1] * self.timestamps.itemsize
            + self.scomplens.buffer_info()[1] * self.scomplens.itemsize
            + self.pcomplens.buffer_info()[1] * self.pcomplens.itemsize
            + self.ndeviats.buffer_info()[1] * self.ndeviats.itemsize
        )

    def slice_by_time(self, start: int, end: int) -> tuple[int, int]:
        """Return ``[lo, hi)`` index range whose timestamps fall in ``[start, end]``.

        Both bounds are inclusive on the timestamp axis (matches how the
        existing API handles range queries) but the returned index range
        is half-open for easy slicing: ``timestamps[lo:hi]``.
        """
        ts = self.timestamps
        lo = bisect.bisect_left(ts, start)
        hi = bisect.bisect_right(ts, end)
        return lo, hi

    # --- boundary / stats helpers used by summary and tool handlers -------

    def first_time(self) -> int | None:
        """``curtime`` of the first sample, or ``None`` on an empty index."""
        return self.timestamps[0] if len(self.timestamps) else None

    def last_time(self) -> int | None:
        """``curtime`` of the last sample, or ``None`` on an empty index."""
        return self.timestamps[-1] if len(self.timestamps) else None

    def median_interval_seconds(self) -> int | None:
        """Median delta between consecutive sample timestamps.

        Mirrors ``atop_web.llm.context._median_interval_seconds`` so the
        summary route's ``recommended_min_range_seconds`` comes out
        identical whether the session is eager or lazy. Ignores non-
        positive deltas (first-frame artifacts in some captures).
        """
        ts = self.timestamps
        n = len(ts)
        if n < 2:
            return None
        deltas: list[int] = []
        for i in range(1, n):
            dt = ts[i] - ts[i - 1]
            if dt > 0:
                deltas.append(dt)
        if not deltas:
            return None
        deltas.sort()
        return deltas[len(deltas) // 2]


def build_sample_index(
    sample_offsets: "List[SampleOffset]",
    spec: "VersionSpec",
) -> SampleIndex:
    """Pack a list of ``SampleOffset`` into typed arrays.

    Doing this in one shot (rather than appending in the scanner's inner
    loop) keeps the scanner's code path untouched and lets both the
    eager and lazy parsers share a single source of offsets.
    """
    n = len(sample_offsets)
    offsets = array.array(_OFFSET_CODE, [0]) * n
    timestamps = array.array(_TIMESTAMP_CODE, [0]) * n
    scomplens = array.array(_SIZE_CODE, [0]) * n
    pcomplens = array.array(_SIZE_CODE, [0]) * n
    ndeviats = array.array(_SIZE_CODE, [0]) * n

    for i, so in enumerate(sample_offsets):
        offsets[i] = so.offset
        timestamps[i] = so.curtime
        scomplens[i] = so.scomplen
        pcomplens[i] = so.pcomplen
        ndeviats[i] = so.ndeviat

    return SampleIndex(
        offsets=offsets,
        timestamps=timestamps,
        scomplens=scomplens,
        pcomplens=pcomplens,
        ndeviats=ndeviats,
        spec=spec,
    )
