"""Pre-aggregated grids over a rawlog's system-wide metrics.

Phase 22 T-04. A rawlog covering a day at 10-second resolution is 8k+
samples. Rendering a whole-day chart by pulling every sample through
lazy decode defeats the point of Phase 22 — it would have to sstat-
inflate 8k times per redraw. Instead we scan the rawlog once at parse
time, bucket every system-wide counter onto 1m / 5m / 1h grids, and
answer wide-window chart queries straight from the grids. Narrower
queries fall through to the lazy decode path (see T-07).

Only system-wide counters live here. Per-process (tstat) aggregates are
out of scope: they drive the processes tab, not the chart, and tstat
dominates per-sample inflate cost — downsampling it inside the same
pass would cost the memory we are trying to save.

Storage shape
-------------
Each grid is a ``Grid`` dataclass holding:

* ``bucket_starts`` — ``array.array("q")`` of bucket-start unix seconds.
* ``series`` — a ``dict[str, array.array]`` keyed by metric name.

All arrays use native typecodes so footprint stays within the plan's
"hundreds of KB" budget even on long captures.
"""

from __future__ import annotations

import array
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from atop_web.parser.lazy import LazyRawLog, SampleView


# Bucket step in seconds per grid name. The order doubles as the
# fallback priority when callers ask for a coarser resolution than 1m.
_GRID_STEPS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "1h": 3600,
}

# Below this sample count we skip building the 5m/1h grids — they add
# cost without giving meaningfully different resolution for a short
# capture. 1m is always built.
_GRID_MIN_SAMPLES: dict[str, int] = {
    "1m": 0,
    "5m": 150,   # ~25 minutes of 10-second samples
    "1h": 1800,  # ~5 hours of 10-second samples
}


# Metric names surfaced to chart callers. Kept in one place so the grid
# builder and the test suite stay in sync.
_METRIC_NAMES: tuple[str, ...] = (
    "cpu_busy",           # fraction in [0, 1]
    "mem_used",           # pages
    "mem_available",      # pages (may be None on atop 2.7)
    "swap_used",          # pages
    "disk_read_sectors",  # sectors
    "disk_write_sectors",
    "net_rx_bytes",
    "net_tx_bytes",
)


# Typecode for metric arrays — float64 covers both integer counters (as
# averages they may be non-integer) and the 0..1 cpu busy fraction.
_METRIC_CODE = "d"
_TIMESTAMP_CODE = "q"


@dataclass
class Grid:
    """Per-grid bucketed metrics."""

    step_seconds: int
    bucket_starts: "array.array[int]"
    series: dict[str, "array.array[float]"]

    def __len__(self) -> int:
        return len(self.bucket_starts)

    def mem_bytes(self) -> int:
        total = self.bucket_starts.buffer_info()[1] * self.bucket_starts.itemsize
        for arr in self.series.values():
            total += arr.buffer_info()[1] * arr.itemsize
        return total


@dataclass
class LookupResult:
    """What the chart route gets back from ``Aggregate.lookup``."""

    hit: bool
    # Populated only when ``hit``; both are the same length.
    bucket_starts: Optional[list[int]] = None
    series: Optional[list[Optional[float]]] = None


@dataclass
class Aggregate:
    """All grids for one rawlog."""

    grids: dict[str, Grid] = field(default_factory=dict)

    def bytes_footprint(self) -> int:
        return sum(g.mem_bytes() for g in self.grids.values())

    def lookup(self, metric: str, grid: str, start: int, end: int) -> LookupResult:
        """Answer ``[start, end]`` from the named grid if it can.

        A query is only a hit when the window is wide enough that every
        sample in it fits inside at least one bucket of ``grid`` — sub-
        bucket windows force the route to fall back to lazy decode so
        the user sees original-resolution data. That test is:
        ``(end - start) >= step_seconds``.
        """
        g = self.grids.get(grid)
        if g is None:
            return LookupResult(hit=False)
        if metric not in g.series:
            return LookupResult(hit=False)
        if end - start < g.step_seconds:
            return LookupResult(hit=False)
        # Clip to bucket starts that intersect [start, end].
        starts: list[int] = []
        values: list[Optional[float]] = []
        source_starts = g.bucket_starts
        source_series = g.series[metric]
        for i, t in enumerate(source_starts):
            if t + g.step_seconds <= start:
                continue
            if t > end:
                break
            v = source_series[i]
            starts.append(t)
            # NaN encodes "no decoded sample in this bucket"; surface as None.
            values.append(None if v != v else v)
        return LookupResult(hit=True, bucket_starts=starts, series=values)


# ---------------------------------------------------------------------------
# Builder


def _bucket_floor(curtime: int, step: int) -> int:
    return (curtime // step) * step


def _extract_sample_metrics(view: "SampleView") -> dict[str, Optional[float]]:
    """Pull one row of chart metrics from a single SampleView.

    Returns a dict keyed by metric name. ``None`` means "metric is not
    available for this sample" — the aggregator skips those values in
    the bucket average instead of poisoning it.
    """
    out: dict[str, Optional[float]] = {name: None for name in _METRIC_NAMES}

    # CPU busy fraction: (utime+stime+ntime+Itime+Stime+steal+guest) /
    # (hertz * nrcpu * interval). itime/wtime are idle-ish; leaving them
    # out is atop's convention for "busy %".
    cpu = view.system_cpu
    interval = view.interval if view.interval and view.interval > 0 else 0
    if cpu is not None and cpu.hertz > 0 and cpu.nrcpu > 0 and interval > 0:
        all_cpu = cpu.all
        busy_ticks = (
            all_cpu.utime
            + all_cpu.stime
            + all_cpu.ntime
            + all_cpu.Itime
            + all_cpu.Stime
            + all_cpu.steal
            + all_cpu.guest
        )
        budget = cpu.hertz * cpu.nrcpu * interval
        if budget > 0:
            frac = busy_ticks / budget
            if frac < 0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0
            out["cpu_busy"] = frac

    mem = view.system_memory
    if mem is not None:
        used = mem.physmem - mem.freemem - mem.cachemem - mem.buffermem - mem.slabmem
        if used < 0:
            used = 0
        out["mem_used"] = float(used)
        out["mem_available"] = (
            float(mem.availablemem) if mem.availablemem is not None else None
        )
        swap_used = mem.totswap - mem.freeswap
        if swap_used < 0:
            swap_used = 0
        out["swap_used"] = float(swap_used)

    disk = view.system_disk
    if disk is not None:
        rs = sum(d.nrsect for d in disk.disks)
        ws = sum(d.nwsect for d in disk.disks)
        out["disk_read_sectors"] = float(rs)
        out["disk_write_sectors"] = float(ws)

    net = view.system_network
    if net is not None:
        rx = sum(i.rbyte for i in net.interfaces)
        tx = sum(i.sbyte for i in net.interfaces)
        out["net_rx_bytes"] = float(rx)
        out["net_tx_bytes"] = float(tx)

    return out


def _integer_metric(name: str) -> bool:
    # mem_used / mem_available / swap_used / disk / net are integer
    # counters and we want the aggregate to round to int when averaging.
    # cpu_busy is a fraction and must stay float.
    return name != "cpu_busy"


class _GridBuilder:
    """Running-sum accumulator for one grid."""

    def __init__(self, step: int) -> None:
        self.step = step
        self.bucket_starts: list[int] = []
        # For each bucket: metric -> (sum, count). Stored as parallel
        # dicts to avoid a per-bucket Python dict allocation.
        self._sums: list[dict[str, float]] = []
        self._counts: list[dict[str, int]] = []
        self._current_start: int | None = None

    def _start_bucket(self, start: int) -> None:
        self.bucket_starts.append(start)
        self._sums.append({name: 0.0 for name in _METRIC_NAMES})
        self._counts.append({name: 0 for name in _METRIC_NAMES})
        self._current_start = start

    def add(self, curtime: int, metrics: dict[str, Optional[float]]) -> None:
        start = _bucket_floor(curtime, self.step)
        if self._current_start != start:
            self._start_bucket(start)
        sums = self._sums[-1]
        counts = self._counts[-1]
        for name, v in metrics.items():
            if v is None:
                continue
            sums[name] += v
            counts[name] += 1

    def finalize(self) -> Grid:
        starts = array.array(_TIMESTAMP_CODE, self.bucket_starts)
        series: dict[str, array.array] = {
            name: array.array(_METRIC_CODE, []) for name in _METRIC_NAMES
        }
        for i in range(len(self.bucket_starts)):
            sums = self._sums[i]
            counts = self._counts[i]
            for name in _METRIC_NAMES:
                c = counts[name]
                if c == 0:
                    series[name].append(float("nan"))
                else:
                    mean = sums[name] / c
                    if _integer_metric(name):
                        mean = float(int(mean))
                    series[name].append(mean)
        return Grid(step_seconds=self.step, bucket_starts=starts, series=series)


def build_aggregate(lazy_rawlog: "LazyRawLog") -> Aggregate:
    """Walk the lazy rawlog once and bucket every sample onto every grid.

    Picks which grids to populate based on the sample count. The 1-minute
    grid is always populated so the chart route has a reliable default;
    coarser grids only show up when they add real resolution.
    """
    n = len(lazy_rawlog)
    active_grids = [
        (name, _GridBuilder(step))
        for name, step in _GRID_STEPS.items()
        if n >= _GRID_MIN_SAMPLES[name]
    ]
    if not active_grids:
        return Aggregate(grids={})

    for view in lazy_rawlog:
        metrics = _extract_sample_metrics(view)
        for _, builder in active_grids:
            builder.add(view.curtime, metrics)

    agg = Aggregate()
    for name, builder in active_grids:
        agg.grids[name] = builder.finalize()
    return agg
