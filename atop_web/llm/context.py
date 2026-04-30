"""Context builders for the chat endpoint.

The chat endpoint serves two investigation modes:

* ``all`` mode (no ``time_range``): we hand the model aggregate statistics
  for the whole capture plus a list of *spike candidates* (CPU p95 windows,
  memory spikes, disk bursts) so it can decide which spans deserve a closer
  look. The system prompt instructs the model to emit
  ``<range start=... end=... reason=.../>`` tags for problem spans and the
  SSE layer forwards those tags as ``range_hint`` events.

* ``range`` mode (``time_range`` provided): we filter samples to the
  requested window and produce a compact per window summary (averages,
  maxima, top processes) so the model focuses on that slice.

Both builders reuse ``briefing._fit_to_budget`` to stay within the token
budget. ``None`` counters (availablemem, inflight) are passed through as
``null`` so the model sees "not measured" instead of ``0``, matching the
Phase 16 briefing convention.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from statistics import mean
from typing import Any

from atop_web.llm import briefing
from atop_web.parser.reader import RawLog, Sample


def _iso(epoch: int | None) -> str | None:
    """Format an epoch second integer as ISO8601 UTC with a ``Z`` suffix.

    LLMs are notoriously bad at arithmetic on 10 digit epoch timestamps;
    the Phase 19 fix is to hand them pre formatted strings so the model
    only has to echo values back instead of computing them. ``None`` in
    ``None`` out so missing counters stay null in the JSON payload.
    """
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(int(epoch), tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except (OSError, ValueError, OverflowError):
        return None


MAX_CONTEXT_CHARS = briefing.MAX_INPUT_CHARS
TOP_N = 5
SPIKE_P95_FRACTION = 0.95
MAX_SPIKE_WINDOWS = 6
# Minimum samples we want inside any range suggestion. With typical atop
# intervals of 60s..600s this keeps a ``<range/>`` selection wide enough
# to actually contain data instead of landing between two samples.
MIN_SAMPLES_PER_RANGE = 20
# Absolute floor for the suggested min range: even on very dense captures
# (eg. 5s intervals) we want at least a 10 minute window for readability.
MIN_RANGE_FLOOR_SECONDS = 600


def _median_interval_seconds(samples: list[Sample]) -> int | None:
    """Return the median interval between samples in seconds.

    ``Sample.interval`` is the per sample value atop wrote, but it can be
    0 on the very first frame of a capture; we prefer timestamp deltas
    so a single malformed sample does not skew the estimate. Returns
    ``None`` when there are fewer than 2 samples, which lets callers
    decide whether to fall back to a default.
    """
    if len(samples) < 2:
        return None
    deltas: list[int] = []
    for prev, curr in zip(samples, samples[1:]):
        dt = curr.curtime - prev.curtime
        if dt > 0:
            deltas.append(dt)
    if not deltas:
        return None
    deltas.sort()
    mid = len(deltas) // 2
    return deltas[mid]


def _recommended_min_range(interval_seconds: int | None) -> int:
    """Recommended minimum ``end - start`` for a ``<range/>`` suggestion."""
    if not interval_seconds or interval_seconds <= 0:
        return MIN_RANGE_FLOOR_SECONDS
    return max(interval_seconds * MIN_SAMPLES_PER_RANGE, MIN_RANGE_FLOOR_SECONDS)


def _expand_spike_window(
    ts: int, interval_seconds: int | None, min_range_seconds: int
) -> tuple[int, int]:
    """Center a spike sample and widen it to ``min_range_seconds``.

    Using the single sample timestamp as both ``start`` and ``end`` leaves
    the model with a zero width range, which in turn can end up narrower
    than the capture interval and produce "no samples in range" results
    when the user clicks the badge.
    """
    half = max(min_range_seconds // 2, interval_seconds or 0)
    return ts - half, ts + half


def _filter_samples(
    rawlog: RawLog, start: int | None, end: int | None
) -> list[Sample]:
    if start is None and end is None:
        return list(rawlog.samples)
    lo = start if start is not None else -(1 << 62)
    hi = end if end is not None else (1 << 62)
    return [s for s in rawlog.samples if lo <= s.curtime <= hi]


def _sample_cpu_pct(sample: Sample, hertz: int, ncpu: int) -> float | None:
    cpu = sample.system_cpu
    if cpu is None or hertz <= 0 or ncpu <= 0 or sample.interval <= 0:
        return None
    a = cpu.all
    busy = (
        a.utime + a.stime + a.ntime + a.Itime + a.Stime + a.steal + a.guest
    )
    # busy excludes wtime (iowait) and itime (idle); the model can still ask
    # for iowait separately from the detailed fields.
    denom = hertz * sample.interval * ncpu
    if denom <= 0:
        return None
    return round(busy / denom * 100.0, 2)


def _sample_mem_used_mib(
    sample: Sample, pagesize: int
) -> float | None:
    m = sample.system_memory
    if m is None or pagesize <= 0:
        return None
    used_pages = max(m.physmem - m.freemem - m.cachemem - m.buffermem - m.slabmem, 0)
    return round(used_pages * pagesize / (1024 * 1024), 1)


def _sample_disk_total_mibs(sample: Sample, prev: Sample | None) -> float | None:
    disk = sample.system_disk
    if disk is None:
        return None
    if prev is None or prev.system_disk is None:
        return 0.0
    dt = max(1, sample.curtime - prev.curtime)
    prev_by_name = {d.name: d for d in prev.system_disk.disks}
    total_sect = 0
    for d in disk.disks:
        pd = prev_by_name.get(d.name)
        if pd is None:
            continue
        total_sect += max(d.nrsect - pd.nrsect, 0) + max(d.nwsect - pd.nwsect, 0)
    return round(total_sect * 512 / (1024 * 1024) / dt, 2)


def _summarize_window(samples: list[Sample], hertz: int, pagesize: int) -> dict:
    """Average/max/min aggregates for a slice, skipping ``None`` readings."""
    if not samples:
        return {}
    ncpu_candidates = [s.nrcpu for s in samples if s.nrcpu]
    ncpu = ncpu_candidates[-1] if ncpu_candidates else 1

    cpu_pcts: list[float] = []
    for s in samples:
        pct = _sample_cpu_pct(s, hertz, ncpu or 1)
        if pct is not None:
            cpu_pcts.append(pct)

    mem_used: list[float] = []
    mem_avail: list[float] = []
    for s in samples:
        used = _sample_mem_used_mib(s, pagesize)
        if used is not None:
            mem_used.append(used)
        m = s.system_memory
        if m is not None and m.availablemem is not None:
            mem_avail.append(round(m.availablemem * pagesize / (1024 * 1024), 1))

    disk_mibs: list[float] = []
    prev: Sample | None = None
    for s in samples:
        val = _sample_disk_total_mibs(s, prev)
        prev = s
        if val is not None:
            disk_mibs.append(val)

    def agg(values: list[float]) -> dict | None:
        if not values:
            return None
        return {
            "avg": round(mean(values), 2),
            "max": round(max(values), 2),
            "min": round(min(values), 2),
            "samples": len(values),
        }

    return {
        "cpu_pct": agg(cpu_pcts),
        "mem_used_mib": agg(mem_used),
        "mem_available_mib": agg(mem_avail) if mem_avail else None,
        "disk_total_mibps": agg(disk_mibs),
    }


def _top_processes(samples: list[Sample], key: str, top_n: int = TOP_N) -> list[dict]:
    """Aggregate the top ``top_n`` processes across ``samples`` by ``key``.

    ``key`` is ``"cpu"`` (utime+stime summed) or ``"rss"`` (max RSS seen).
    Cmdline is intentionally dropped to avoid leaking secrets.
    """
    totals: dict[tuple[int, str], dict[str, Any]] = {}
    for s in samples:
        for p in s.processes:
            k = (p.pid, p.name)
            entry = totals.setdefault(
                k,
                {
                    "pid": p.pid,
                    "name": p.name,
                    "cpu_ticks": 0,
                    "rmem_kb_max": 0,
                    "samples": 0,
                },
            )
            entry["cpu_ticks"] += p.utime + p.stime
            if p.rmem_kb > entry["rmem_kb_max"]:
                entry["rmem_kb_max"] = p.rmem_kb
            entry["samples"] += 1
    if key == "cpu":
        ranked = sorted(totals.values(), key=lambda e: e["cpu_ticks"], reverse=True)
    elif key == "rss":
        ranked = sorted(totals.values(), key=lambda e: e["rmem_kb_max"], reverse=True)
    else:
        raise ValueError(f"unsupported top_processes key: {key!r}")
    return ranked[:top_n]


def build_range_context(
    rawlog: RawLog, start: int | None, end: int | None
) -> dict:
    """Compact summary for ``range`` mode.

    ``start`` / ``end`` are epoch seconds (UTC). Either or both may be
    ``None`` - missing bounds default to the extremes of the capture.
    """
    subset = _filter_samples(rawlog, start, end)
    hertz = rawlog.header.hertz or 100
    pagesize = rawlog.header.pagesize or 4096
    interval_seconds = _median_interval_seconds(rawlog.samples)
    min_range_seconds = _recommended_min_range(interval_seconds)
    capture_start = rawlog.samples[0].curtime if rawlog.samples else None
    capture_end = rawlog.samples[-1].curtime if rawlog.samples else None
    capture_duration = (
        capture_end - capture_start
        if capture_start is not None and capture_end is not None
        else 0
    )
    payload: dict[str, Any] = {
        "mode": "range",
        "capture": {
            "hostname": rawlog.header.nodename,
            "aversion": rawlog.header.aversion,
            "hertz": hertz,
            "pagesize": pagesize,
            "sample_count_total": len(rawlog.samples),
            "start": _iso(capture_start),
            "end": _iso(capture_end),
            "start_epoch": capture_start,
            "end_epoch": capture_end,
            "duration_seconds": capture_duration,
            "interval_seconds": interval_seconds,
            "recommended_min_range_seconds": min_range_seconds,
        },
        "range": {
            "start": _iso(start),
            "end": _iso(end),
            "sample_count": len(subset),
            "first": _iso(subset[0].curtime) if subset else None,
            "last": _iso(subset[-1].curtime) if subset else None,
        },
    }
    if not subset:
        payload["note"] = "no samples in requested range"
        return payload
    payload["aggregate"] = _summarize_window(subset, hertz, pagesize)
    payload["top_processes_by_cpu"] = _top_processes(subset, "cpu")
    payload["top_processes_by_rss"] = _top_processes(subset, "rss")
    return payload


def _detect_spikes(rawlog: RawLog) -> list[dict]:
    """Return up to ``MAX_SPIKE_WINDOWS`` candidate windows worth the model's attention.

    We use simple signals so the model can still explain away false
    positives: CPU above the p95 of the capture, memory used exceeding the
    p95 of used MiB, and disk throughput above the p95 of total MiB/s. Each
    window is centered on the spike sample and widened to
    ``recommended_min_range_seconds`` so the ``<range/>`` suggestion the
    model echoes back has at least ~20 samples in it.
    """
    samples = rawlog.samples
    if len(samples) < 3:
        return []
    hertz = rawlog.header.hertz or 100
    pagesize = rawlog.header.pagesize or 4096
    ncpu = next((s.nrcpu for s in reversed(samples) if s.nrcpu), None) or 1
    interval_seconds = _median_interval_seconds(samples)
    min_range_seconds = _recommended_min_range(interval_seconds)

    cpu_pcts = [
        (s.curtime, _sample_cpu_pct(s, hertz, ncpu)) for s in samples
    ]
    mem_used = [
        (s.curtime, _sample_mem_used_mib(s, pagesize)) for s in samples
    ]
    disk_vals: list[tuple[int, float | None]] = []
    prev: Sample | None = None
    for s in samples:
        disk_vals.append((s.curtime, _sample_disk_total_mibs(s, prev)))
        prev = s

    def p95(values: list[float]) -> float | None:
        clean = sorted(v for v in values if v is not None)
        if len(clean) < 5:
            return None
        idx = int(len(clean) * SPIKE_P95_FRACTION)
        if idx >= len(clean):
            idx = len(clean) - 1
        return clean[idx]

    cpu_vals = [v for _, v in cpu_pcts if v is not None]
    mem_vals = [v for _, v in mem_used if v is not None]
    dsk_vals = [v for _, v in disk_vals if v is not None]

    cpu_th = p95(cpu_vals)
    mem_th = p95(mem_vals)
    dsk_th = p95(dsk_vals)

    def expanded(ts: int) -> tuple[int, int]:
        return _expand_spike_window(ts, interval_seconds, min_range_seconds)

    windows: list[dict] = []
    for ts, v in cpu_pcts:
        if cpu_th is not None and v is not None and v >= cpu_th and v > 50.0:
            s0, s1 = expanded(ts)
            windows.append({
                "start": _iso(s0),
                "end": _iso(s1),
                "center": _iso(ts),
                "signal": "cpu_high",
                "value": v,
                "threshold": round(cpu_th, 2),
            })
    for ts, v in mem_used:
        if mem_th is not None and v is not None and v >= mem_th:
            s0, s1 = expanded(ts)
            windows.append({
                "start": _iso(s0),
                "end": _iso(s1),
                "center": _iso(ts),
                "signal": "mem_high",
                "value": v,
                "threshold": round(mem_th, 1),
            })
    for ts, v in disk_vals:
        if dsk_th is not None and v is not None and v >= dsk_th and v > 1.0:
            s0, s1 = expanded(ts)
            windows.append({
                "start": _iso(s0),
                "end": _iso(s1),
                "center": _iso(ts),
                "signal": "disk_burst",
                "value": v,
                "threshold": round(dsk_th, 2),
            })

    windows.sort(key=lambda w: w["value"], reverse=True)
    return windows[:MAX_SPIKE_WINDOWS]


def build_all_context(rawlog: RawLog) -> dict:
    """Compact summary for ``all`` mode: full capture stats + spike hints."""
    samples = rawlog.samples
    hertz = rawlog.header.hertz or 100
    pagesize = rawlog.header.pagesize or 4096
    interval_seconds = _median_interval_seconds(samples)
    min_range_seconds = _recommended_min_range(interval_seconds)
    capture_start = samples[0].curtime if samples else None
    capture_end = samples[-1].curtime if samples else None
    payload: dict[str, Any] = {
        "mode": "all",
        "capture": {
            "hostname": rawlog.header.nodename,
            "aversion": rawlog.header.aversion,
            "hertz": hertz,
            "pagesize": pagesize,
            "sample_count": len(samples),
            "start": _iso(capture_start),
            "end": _iso(capture_end),
            "start_epoch": capture_start,
            "end_epoch": capture_end,
            "duration_seconds": (
                capture_end - capture_start if samples else 0
            ),
            "interval_seconds": interval_seconds,
            "recommended_min_range_seconds": min_range_seconds,
        },
    }
    if not samples:
        payload["note"] = "no samples"
        return payload
    payload["aggregate"] = _summarize_window(samples, hertz, pagesize)
    payload["top_processes_by_cpu"] = _top_processes(samples, "cpu")
    payload["top_processes_by_rss"] = _top_processes(samples, "rss")
    payload["spike_candidates"] = _detect_spikes(rawlog)
    return payload


def serialize_context(payload: dict) -> tuple[str, bool]:
    """Serialize ``payload`` to JSON, shrinking lists until it fits the budget.

    Returns ``(text, truncated)``. The lists we trim are, in order: the
    spike candidates (whole list), the two process tables (top N halved),
    then a hard character cap as a last resort.
    """
    truncated = False
    while True:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= MAX_CONTEXT_CHARS:
            return text, truncated
        spikes = payload.get("spike_candidates")
        if isinstance(spikes, list) and len(spikes) > 0:
            payload["spike_candidates"] = spikes[: len(spikes) // 2]
            truncated = True
            continue
        shrunk = False
        for key in ("top_processes_by_cpu", "top_processes_by_rss"):
            rows = payload.get(key)
            if isinstance(rows, list) and len(rows) > 1:
                payload[key] = rows[: max(1, len(rows) // 2)]
                truncated = True
                shrunk = True
        if shrunk:
            continue
        # Hard cap so we never exceed the budget.
        return text[: MAX_CONTEXT_CHARS - 32] + '..."TRUNCATED"', True
