"""Provider agnostic tool (function calling) plumbing.

The chat router hands the LLM a list of :class:`ToolSpec` objects that
describe on demand calculations over the parsed atop rawlog. Providers
translate those specs into their native format (Bedrock ``toolConfig``,
OpenAI ``tools``, Anthropic ``tools``, Ollama ``tools``, ...). When the
model decides to call one, the provider yields a :class:`ToolUseRequest`
event; the router executes the matching handler locally and feeds the
resulting :class:`ToolResult` back in the next turn.

Design goals:

* Keep the spec and event types free of any vendor detail so we can
  implement Bedrock now (Phase 20) and OpenAI / Anthropic / Ollama /
  Gemini later (Phase 21) without reshaping the chat router.
* Pre compute nothing. Handlers run lazily against the rawlog closure
  so we never explode the prompt with metric combinations the user may
  never ask about.
* Return human units (percentage, MiB, IOPS, packets per second) plus
  ISO8601 timestamps; never leak ``cpu_ticks`` or other raw counters
  into the tool result.
"""

from __future__ import annotations

import fnmatch
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from atop_web.parser.reader import RawLog, Sample


# ---------------------------------------------------------------------------
# Provider neutral types
# ---------------------------------------------------------------------------


ToolHandler = Callable[[dict], dict]


@dataclass
class ToolSpec:
    """Describes one tool the model may call.

    ``input_schema`` is a JSON Schema object (draft 2020-12 compatible
    subset); providers lower it to whatever shape their API wants. The
    ``handler`` receives the already parsed argument dict and returns a
    JSON serializable dict.
    """

    name: str
    description: str
    input_schema: dict
    handler: ToolHandler

    def call(self, args: dict) -> dict:
        return self.handler(args or {})


@dataclass
class ToolCall:
    """One tool invocation the model emitted during a turn."""

    call_id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    """Result of a local handler, fed back into the next model turn."""

    call_id: str
    name: str
    content: dict
    is_error: bool = False


# Provider events emitted by ``LLMProvider.chat_with_tools``. Streaming
# intent is preserved: text deltas arrive as the model writes them, tool
# use requests arrive on their own frame, and ``Stop`` closes the turn.


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolUseRequest:
    call_id: str
    name: str
    arguments: dict


@dataclass
class Stop:
    reason: str  # "end_turn" | "tool_use" | "max_tokens" | "stop_sequence" | ...


ProviderEvent = TextDelta | ToolUseRequest | Stop


# ---------------------------------------------------------------------------
# Helpers used by handlers
# ---------------------------------------------------------------------------


def _iso(epoch: int | float | None) -> str | None:
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(int(epoch), tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except (OSError, ValueError, OverflowError):
        return None


def _parse_iso_or_epoch(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Plain int string: accept as epoch seconds.
        if text.lstrip("-").isdigit():
            return int(text)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return int(datetime.fromisoformat(text).timestamp())
        except ValueError:
            return None
    return None


def _subset(samples: list[Sample], start: int | None, end: int | None) -> list[Sample]:
    if start is None and end is None:
        return samples
    lo = start if start is not None else -(1 << 62)
    hi = end if end is not None else (1 << 62)
    return [s for s in samples if lo <= s.curtime <= hi]


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    idx = q * (len(xs) - 1)
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return xs[int(idx)]
    frac = idx - low
    return xs[low] * (1 - frac) + xs[high] * frac


def _sample_cpu_pct(sample: Sample, hertz: int, ncpu: int) -> float | None:
    cpu = sample.system_cpu
    if cpu is None or hertz <= 0 or ncpu <= 0 or sample.interval <= 0:
        return None
    a = cpu.all
    busy = a.utime + a.stime + a.ntime + a.Itime + a.Stime + a.steal + a.guest
    denom = hertz * sample.interval * ncpu
    if denom <= 0:
        return None
    return round(busy / denom * 100.0, 3)


def _sample_mem_used_mib(sample: Sample, pagesize: int) -> float | None:
    m = sample.system_memory
    if m is None or pagesize <= 0:
        return None
    used_pages = max(m.physmem - m.freemem - m.cachemem - m.buffermem - m.slabmem, 0)
    return round(used_pages * pagesize / (1024 * 1024), 2)


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
    return round(total_sect * 512 / (1024 * 1024) / dt, 3)


def _sample_net_total_kbps(sample: Sample, prev: Sample | None) -> float | None:
    net = sample.system_network
    if net is None:
        return None
    if prev is None or prev.system_network is None:
        return 0.0
    dt = max(1, sample.curtime - prev.curtime)
    prev_by_name = {i.name: i for i in prev.system_network.interfaces}
    total_bytes = 0
    for i in net.interfaces:
        pi = prev_by_name.get(i.name)
        if pi is None:
            continue
        total_bytes += max(i.rbyte - pi.rbyte, 0) + max(i.sbyte - pi.sbyte, 0)
    return round(total_bytes / 1000 / dt, 3)


def _per_sample_metric(
    metric: str, samples: list[Sample], hertz: int, pagesize: int, ncpu: int
) -> list[tuple[int, float]]:
    """Compute ``metric`` for every sample; returns (ts, value) pairs."""
    metric = metric.lower()
    out: list[tuple[int, float]] = []
    if metric in ("cpu", "cpu_pct"):
        for s in samples:
            v = _sample_cpu_pct(s, hertz, ncpu)
            if v is not None:
                out.append((s.curtime, v))
    elif metric in ("mem", "memory", "mem_used_mib"):
        for s in samples:
            v = _sample_mem_used_mib(s, pagesize)
            if v is not None:
                out.append((s.curtime, v))
    elif metric in ("disk", "disk_mibps", "disk_total_mibps"):
        prev: Sample | None = None
        for s in samples:
            v = _sample_disk_total_mibs(s, prev)
            prev = s
            if v is not None:
                out.append((s.curtime, v))
    elif metric in ("net", "network", "net_kbps"):
        prev = None
        for s in samples:
            v = _sample_net_total_kbps(s, prev)
            prev = s
            if v is not None:
                out.append((s.curtime, v))
    elif metric in ("load1", "load5", "load15", "lavg1", "lavg5", "lavg15"):
        attr = {
            "load1": "lavg1", "lavg1": "lavg1",
            "load5": "lavg5", "lavg5": "lavg5",
            "load15": "lavg15", "lavg15": "lavg15",
        }[metric]
        for s in samples:
            if s.system_cpu is not None:
                out.append((s.curtime, float(getattr(s.system_cpu, attr))))
    else:
        raise ValueError(
            f"unsupported metric: {metric!r}. "
            "Use cpu, mem, disk, net, load1, load5 or load15."
        )
    return out


SUPPORTED_METRICS = (
    "cpu", "mem", "disk", "net", "load1", "load5", "load15",
)


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------


def build_tool_specs(rawlog: RawLog) -> list[ToolSpec]:
    """Return the 7 tool specs bound to ``rawlog`` via closures.

    Each handler captures ``rawlog`` so the router only has to pass
    argument dicts. ``hertz``, ``pagesize`` and the derived ``ncpu`` are
    resolved once so handlers do not re walk the header for every call.
    """
    samples_all = rawlog.samples
    hertz = rawlog.header.hertz or 100
    pagesize = rawlog.header.pagesize or 4096
    ncpu = (
        next((s.nrcpu for s in reversed(samples_all) if s.nrcpu), None)
        or (
            samples_all[0].system_cpu.nrcpu
            if samples_all and samples_all[0].system_cpu
            else 1
        )
        or 1
    )
    capture_start = samples_all[0].curtime if samples_all else None
    capture_end = samples_all[-1].curtime if samples_all else None

    def _range_args(args: dict) -> tuple[int | None, int | None]:
        return (
            _parse_iso_or_epoch(args.get("start")),
            _parse_iso_or_epoch(args.get("end")),
        )

    # ----- 1. get_metric_stats -------------------------------------------

    def get_metric_stats(args: dict) -> dict:
        metric = str(args.get("metric", "")).strip()
        if not metric:
            return {"error": "metric is required"}
        start, end = _range_args(args)
        subset = _subset(samples_all, start, end)
        if not subset:
            return {
                "metric": metric,
                "count": 0,
                "note": "no samples in range",
            }
        try:
            series = _per_sample_metric(metric, subset, hertz, pagesize, ncpu)
        except ValueError as exc:
            return {"error": str(exc), "metric": metric}
        if not series:
            return {
                "metric": metric,
                "count": 0,
                "note": "metric not available (likely null on this capture)",
            }
        values = [v for _, v in series]
        max_ts, max_val = max(series, key=lambda kv: kv[1])
        min_ts, min_val = min(series, key=lambda kv: kv[1])
        avg = round(sum(values) / len(values), 3)
        p95 = _percentile(values, 0.95)
        unit = {
            "cpu": "percent",
            "cpu_pct": "percent",
            "mem": "mib",
            "memory": "mib",
            "mem_used_mib": "mib",
            "disk": "mib_per_second",
            "disk_mibps": "mib_per_second",
            "disk_total_mibps": "mib_per_second",
            "net": "kb_per_second",
            "network": "kb_per_second",
            "net_kbps": "kb_per_second",
            "load1": "load_average",
            "lavg1": "load_average",
            "load5": "load_average",
            "lavg5": "load_average",
            "load15": "load_average",
            "lavg15": "load_average",
        }[metric.lower()]
        return {
            "metric": metric,
            "unit": unit,
            "count": len(values),
            "range_start": _iso(subset[0].curtime),
            "range_end": _iso(subset[-1].curtime),
            "max": {"value": round(max_val, 3), "at": _iso(max_ts)},
            "min": {"value": round(min_val, 3), "at": _iso(min_ts)},
            "avg": avg,
            "p95": round(p95, 3) if p95 is not None else None,
        }

    # ----- 2. get_top_processes ------------------------------------------

    def get_top_processes(args: dict) -> dict:
        metric = str(args.get("metric", "cpu")).strip().lower()
        if metric not in ("cpu", "rss", "mem", "disk", "net"):
            return {"error": f"unsupported process metric: {metric!r}"}
        limit = int(args.get("limit", 5) or 5)
        limit = max(1, min(limit, 50))
        start, end = _range_args(args)
        subset = _subset(samples_all, start, end)
        if not subset:
            return {
                "metric": metric,
                "limit": limit,
                "processes": [],
                "note": "no samples in range",
            }
        # Aggregate ``cpu`` as percentage averaged across the window so
        # the model gets a human readable number rather than raw ticks.
        agg: dict[tuple[int, str], dict] = {}
        window_intervals = 0
        for s in subset:
            if s.interval > 0:
                window_intervals += s.interval
            for p in s.processes:
                key = (p.pid, p.name)
                entry = agg.setdefault(
                    key,
                    {
                        "pid": p.pid,
                        "name": p.name,
                        "utime": 0,
                        "stime": 0,
                        "rmem_kb_max": 0,
                        "rsz": 0,
                        "wsz": 0,
                        "tcp_bytes": 0,
                        "udp_bytes": 0,
                        "samples": 0,
                    },
                )
                entry["utime"] += p.utime
                entry["stime"] += p.stime
                entry["rmem_kb_max"] = max(entry["rmem_kb_max"], p.rmem_kb)
                entry["rsz"] += max(p.rsz, 0)
                entry["wsz"] += max(p.wsz, 0)
                entry["tcp_bytes"] += max(p.tcpsnd, 0) + max(p.tcprcv, 0)
                entry["udp_bytes"] += max(p.udpsnd, 0) + max(p.udprcv, 0)
                entry["samples"] += 1
        items = list(agg.values())

        if metric == "cpu":
            denom = max(hertz * max(window_intervals, 1) * ncpu, 1)
            for item in items:
                ticks = item["utime"] + item["stime"]
                item["cpu_pct"] = round(ticks / denom * 100.0, 3)
            items.sort(key=lambda e: e["cpu_pct"], reverse=True)
        elif metric in ("rss", "mem"):
            for item in items:
                item["rss_mib"] = round(item["rmem_kb_max"] / 1024.0, 2)
            items.sort(key=lambda e: e["rss_mib"], reverse=True)
        elif metric == "disk":
            for item in items:
                item["disk_mib_total"] = round(
                    (item["rsz"] + item["wsz"]) * 512 / (1024 * 1024), 3
                )
            items.sort(key=lambda e: e["disk_mib_total"], reverse=True)
        elif metric == "net":
            for item in items:
                item["net_mib_total"] = round(
                    (item["tcp_bytes"] + item["udp_bytes"]) / (1024 * 1024), 3
                )
            items.sort(key=lambda e: e["net_mib_total"], reverse=True)

        shaped = []
        for item in items[:limit]:
            out: dict[str, Any] = {
                "pid": item["pid"],
                "name": item["name"],
                "samples": item["samples"],
            }
            if metric == "cpu":
                out["cpu_pct"] = item["cpu_pct"]
            elif metric in ("rss", "mem"):
                out["rss_mib"] = item["rss_mib"]
            elif metric == "disk":
                out["disk_mib_total"] = item["disk_mib_total"]
            elif metric == "net":
                out["net_mib_total"] = item["net_mib_total"]
            shaped.append(out)
        return {
            "metric": metric,
            "limit": limit,
            "range_start": _iso(subset[0].curtime),
            "range_end": _iso(subset[-1].curtime),
            "processes": shaped,
        }

    # ----- 3. find_spikes -------------------------------------------------

    def find_spikes(args: dict) -> dict:
        metric = str(args.get("metric", "cpu")).strip().lower()
        threshold = args.get("threshold_pct")
        window_seconds = args.get("window_seconds") or 300
        try:
            window_seconds = max(30, int(window_seconds))
        except (TypeError, ValueError):
            window_seconds = 300
        try:
            series = _per_sample_metric(
                metric, samples_all, hertz, pagesize, ncpu
            )
        except ValueError as exc:
            return {"error": str(exc), "metric": metric}
        if not series:
            return {"metric": metric, "spikes": [], "note": "no samples"}
        values = [v for _, v in series]
        if threshold is None:
            # Default threshold: p95 of the series (same heuristic used by
            # the Level 1 briefing) so the model gets meaningful hits
            # without having to guess a number.
            threshold = _percentile(values, 0.95) or 0.0
        else:
            try:
                threshold = float(threshold)
            except (TypeError, ValueError):
                return {"error": "threshold_pct must be a number"}
        half = window_seconds // 2
        capture_lo = samples_all[0].curtime if samples_all else 0
        capture_hi = samples_all[-1].curtime if samples_all else 0
        spikes: list[dict] = []
        for ts, v in series:
            if v >= threshold:
                spikes.append(
                    {
                        "center": _iso(ts),
                        "start": _iso(max(capture_lo, ts - half)),
                        "end": _iso(min(capture_hi, ts + half)),
                        "value": round(v, 3),
                    }
                )
        spikes.sort(key=lambda s: s["value"], reverse=True)
        return {
            "metric": metric,
            "threshold": round(float(threshold), 3),
            "window_seconds": window_seconds,
            "spike_count": len(spikes),
            "spikes": spikes[:20],
        }

    # ----- 4. get_process_count ------------------------------------------

    def get_process_count(args: dict) -> dict:
        pattern = args.get("pattern")
        start, end = _range_args(args)
        subset = _subset(samples_all, start, end)
        if not subset:
            return {"count_max": 0, "count_avg": 0, "note": "no samples in range"}
        regex = None
        if pattern:
            try:
                # ``pattern`` is interpreted as a glob first (familiar to
                # operators), then falls back to a regex if glob gives no
                # hits on the first sample. We compile it either way.
                regex = re.compile(fnmatch.translate(str(pattern)))
            except re.error:
                return {"error": f"invalid pattern: {pattern!r}"}
        counts_total: list[int] = []
        counts_match: list[int] = []
        for s in subset:
            procs = s.processes
            counts_total.append(len(procs))
            if regex is None:
                counts_match.append(len(procs))
            else:
                counts_match.append(sum(1 for p in procs if regex.match(p.name)))
        return {
            "range_start": _iso(subset[0].curtime),
            "range_end": _iso(subset[-1].curtime),
            "samples": len(subset),
            "pattern": pattern,
            "count_max": max(counts_match),
            "count_min": min(counts_match),
            "count_avg": round(sum(counts_match) / len(counts_match), 2),
            "total_process_max": max(counts_total),
            "total_process_avg": round(sum(counts_total) / len(counts_total), 2),
        }

    # ----- 5. get_samples_in_range ---------------------------------------

    def get_samples_in_range(args: dict) -> dict:
        start, end = _range_args(args)
        if start is None or end is None:
            return {"error": "start and end are required"}
        if end < start:
            return {"error": "end must be >= start"}
        requested = args.get("metrics") or ["cpu", "mem"]
        if isinstance(requested, str):
            requested = [requested]
        requested = [str(m).strip().lower() for m in requested if m]
        for m in requested:
            if m not in SUPPORTED_METRICS:
                return {"error": f"unsupported metric: {m!r}"}
        subset = _subset(samples_all, start, end)
        # Cap the response so a wide range does not blow the token
        # budget; the caller can narrow with ``start``/``end``.
        MAX_ROWS = 60
        step = max(1, len(subset) // MAX_ROWS) if subset else 1
        kept = subset[::step][:MAX_ROWS]
        # Disk / net need a ``prev`` sample for deltas; use the sample
        # right before each kept one so the first row is not zero when
        # the capture contains data outside the range.
        prev_of: dict[int, Sample | None] = {}
        for i, s in enumerate(kept):
            orig_idx = subset.index(s)
            prev_of[id(s)] = subset[orig_idx - 1] if orig_idx > 0 else None
        rows: list[dict] = []
        for s in kept:
            row: dict[str, Any] = {"at": _iso(s.curtime)}
            for m in requested:
                if m == "cpu":
                    row["cpu_pct"] = _sample_cpu_pct(s, hertz, ncpu)
                elif m == "mem":
                    row["mem_used_mib"] = _sample_mem_used_mib(s, pagesize)
                elif m == "disk":
                    row["disk_mibps"] = _sample_disk_total_mibs(s, prev_of[id(s)])
                elif m == "net":
                    row["net_kbps"] = _sample_net_total_kbps(s, prev_of[id(s)])
                elif m in ("load1", "lavg1"):
                    row["load1"] = (
                        s.system_cpu.lavg1 if s.system_cpu else None
                    )
                elif m in ("load5", "lavg5"):
                    row["load5"] = (
                        s.system_cpu.lavg5 if s.system_cpu else None
                    )
                elif m in ("load15", "lavg15"):
                    row["load15"] = (
                        s.system_cpu.lavg15 if s.system_cpu else None
                    )
            rows.append(row)
        return {
            "start": _iso(start),
            "end": _iso(end),
            "metrics": requested,
            "sampled_rows": len(rows),
            "total_rows": len(subset),
            "rows": rows,
        }

    # ----- 6. get_capture_info -------------------------------------------

    def get_capture_info(args: dict) -> dict:
        # Median timestamp delta is a better estimator of the effective
        # sampling rate than ``Sample.interval`` which can be 0 on the
        # first frame of a capture.
        if len(samples_all) >= 2:
            deltas = [
                b.curtime - a.curtime
                for a, b in zip(samples_all, samples_all[1:])
                if b.curtime - a.curtime > 0
            ]
            deltas.sort()
            interval = deltas[len(deltas) // 2] if deltas else None
        else:
            interval = None
        return {
            "hostname": rawlog.header.nodename,
            "kernel": rawlog.header.release,
            "atop_version": rawlog.header.aversion,
            "hertz": hertz,
            "pagesize": pagesize,
            "ncpu": ncpu,
            "sample_count": len(samples_all),
            "start": _iso(capture_start),
            "end": _iso(capture_end),
            "duration_seconds": (
                capture_end - capture_start
                if capture_start is not None and capture_end is not None
                else 0
            ),
            "interval_seconds": interval,
        }

    # ----- 7. compare_ranges ---------------------------------------------

    def compare_ranges(args: dict) -> dict:
        ra = args.get("range_a") or {}
        rb = args.get("range_b") or {}
        metric = str(args.get("metric", "cpu")).strip().lower()

        def _one(block: dict) -> dict:
            s = _parse_iso_or_epoch(block.get("start"))
            e = _parse_iso_or_epoch(block.get("end"))
            sub = _subset(samples_all, s, e)
            if not sub:
                return {"count": 0, "note": "no samples", "start": _iso(s), "end": _iso(e)}
            try:
                series = _per_sample_metric(metric, sub, hertz, pagesize, ncpu)
            except ValueError as exc:
                return {"error": str(exc)}
            if not series:
                return {"count": 0, "note": "metric not available"}
            values = [v for _, v in series]
            return {
                "count": len(values),
                "start": _iso(sub[0].curtime),
                "end": _iso(sub[-1].curtime),
                "avg": round(sum(values) / len(values), 3),
                "max": round(max(values), 3),
                "p95": round(_percentile(values, 0.95) or 0.0, 3),
            }

        a = _one(ra)
        b = _one(rb)
        delta: dict[str, Any] = {}
        for field in ("avg", "max", "p95"):
            if field in a and field in b:
                delta[f"{field}_delta"] = round(b[field] - a[field], 3)
                if a[field]:
                    delta[f"{field}_pct_change"] = round(
                        (b[field] - a[field]) / a[field] * 100.0, 2
                    )
        return {"metric": metric, "range_a": a, "range_b": b, "delta": delta}

    return [
        ToolSpec(
            name="get_metric_stats",
            description=(
                "Aggregate statistics (max, min, avg, p95) for a single "
                "system metric over an optional time range. Returns human "
                "units (percent for cpu, MiB for memory, MiB/s for disk, "
                "KB/s for net, load_average for loadN)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "One of: cpu, mem, disk, net, load1, load5, load15.",
                        "enum": list(SUPPORTED_METRICS),
                    },
                    "start": {"type": "string", "description": "ISO8601 UTC lower bound (optional)."},
                    "end": {"type": "string", "description": "ISO8601 UTC upper bound (optional)."},
                },
                "required": ["metric"],
            },
            handler=get_metric_stats,
        ),
        ToolSpec(
            name="get_top_processes",
            description=(
                "Top N processes ranked by a metric over a time range. "
                "``cpu`` returns percent of total capacity across the "
                "window (not raw ticks); ``rss`` returns peak RSS in "
                "MiB; ``disk`` and ``net`` return MiB totals."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["cpu", "rss", "mem", "disk", "net"],
                        "description": "Ranking dimension.",
                    },
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                "required": ["metric"],
            },
            handler=get_top_processes,
        ),
        ToolSpec(
            name="find_spikes",
            description=(
                "Find timestamps where a metric exceeds a threshold. "
                "If ``threshold_pct`` is omitted, the p95 of the "
                "observed metric is used. Returns windows centered on "
                "each spike, widened to ``window_seconds``."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": list(SUPPORTED_METRICS)},
                    "threshold_pct": {
                        "type": "number",
                        "description": "Threshold in the metric's native unit (percent for cpu, MiB for mem, ...).",
                    },
                    "window_seconds": {"type": "integer", "minimum": 30},
                },
                "required": ["metric"],
            },
            handler=find_spikes,
        ),
        ToolSpec(
            name="get_process_count",
            description=(
                "Count processes matching ``pattern`` (glob) per sample "
                "over a time range. Returns min/avg/max counts and "
                "total process counts as context."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern against process name, e.g. 'nginx*'. Omit for total counts.",
                    },
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                },
            },
            handler=get_process_count,
        ),
        ToolSpec(
            name="get_samples_in_range",
            description=(
                "Dump per sample metric values inside a time range "
                "(capped to 60 rows). Use this when the user asks for "
                "a trend or pattern, not just an aggregate."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "ISO8601 UTC."},
                    "end": {"type": "string", "description": "ISO8601 UTC."},
                    "metrics": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(SUPPORTED_METRICS),
                        },
                    },
                },
                "required": ["start", "end"],
            },
            handler=get_samples_in_range,
        ),
        ToolSpec(
            name="get_capture_info",
            description=(
                "Return basic metadata about the capture: hostname, "
                "kernel, atop version, sample count, wall clock start "
                "and end, median interval_seconds."
            ),
            input_schema={
                "type": "object",
                "properties": {},
            },
            handler=get_capture_info,
        ),
        ToolSpec(
            name="compare_ranges",
            description=(
                "Compare two time ranges on the same metric. Returns "
                "avg/max/p95 for each range and absolute + percentage "
                "deltas. Useful for before/after questions."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "range_a": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "string"},
                            "end": {"type": "string"},
                        },
                        "required": ["start", "end"],
                    },
                    "range_b": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "string"},
                            "end": {"type": "string"},
                        },
                        "required": ["start", "end"],
                    },
                    "metric": {"type": "string", "enum": list(SUPPORTED_METRICS)},
                },
                "required": ["range_a", "range_b", "metric"],
            },
            handler=compare_ranges,
        ),
    ]
