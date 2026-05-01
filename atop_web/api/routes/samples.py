"""GET /api/samples: CPU, memory, disk and network time series."""

from __future__ import annotations

from fastapi import APIRouter, Query

from atop_web.api.cache import get_response_cache
from atop_web.api.sessions import get_store
from atop_web.api.timerange import parse_iso_epoch

router = APIRouter()


# These helpers used to be typed to ``Sample`` directly; Phase 22 has
# them operate on any object whose ``processes`` list carries the atop
# per-process fields (``utime``, ``rmem_kb``, ``rsz``, ...). Eager
# ``Sample`` and lazy ``SampleView`` both satisfy that contract.


def _cpu_ticks(sample) -> int:
    return sum(p.utime + p.stime for p in sample.processes)


def _rmem_kb(sample) -> int:
    return sum(max(p.rmem_kb, 0) for p in sample.processes if p.isproc)


def _vmem_kb(sample) -> int:
    return sum(max(p.vmem_kb, 0) for p in sample.processes if p.isproc)


def _disk_read_sectors(sample) -> int:
    return sum(max(p.rsz, 0) for p in sample.processes)


def _disk_write_sectors(sample) -> int:
    return sum(max(p.wsz, 0) for p in sample.processes)


def _net_tcp_send(sample) -> int:
    return sum(max(p.tcpsnd, 0) for p in sample.processes)


def _net_tcp_recv(sample) -> int:
    return sum(max(p.tcprcv, 0) for p in sample.processes)


def _net_udp_send(sample) -> int:
    return sum(max(p.udpsnd, 0) for p in sample.processes)


def _net_udp_recv(sample) -> int:
    return sum(max(p.udprcv, 0) for p in sample.processes)


@router.get("/samples")
def samples(
    session: str = Query(...),
    from_: str | None = Query(None, alias="from", description="ISO8601 lower bound"),
    to: str | None = Query(None, description="ISO8601 upper bound"),
) -> dict:
    sess = get_store().require(session)

    from_epoch = parse_iso_epoch(from_, field="from")
    to_epoch = parse_iso_epoch(to, field="to")

    cache = get_response_cache()
    key = (session, "samples", from_epoch, to_epoch)

    def _build():
        return _samples_impl(sess, session, from_epoch, to_epoch)

    return cache.get_or_compute(key, _build)


def _samples_impl(sess, session: str, from_epoch: int | None, to_epoch: int | None) -> dict:
    # Streaming accumulator: one pass over the window, every per-sample
    # number folded into its array, and the tstat payload on each
    # SampleView dropped as soon as we are done with it. Without this
    # drop, decoding 20k samples through ``.processes`` would keep
    # hundreds of thousands of Process dataclasses alive at once (one
    # per process per sample) and OOMKill the container.
    timeline: list[int] = []
    intervals: list[int] = []
    nrcpu_series: list[int | None] = []
    cpu_ticks: list[int] = []
    rmem: list[int] = []
    vmem: list[int] = []
    dsk_r: list[int] = []
    dsk_w: list[int] = []
    tcp_s: list[int] = []
    tcp_r: list[int] = []
    udp_s: list[int] = []
    udp_r: list[int] = []
    ntask: list[int] = []
    totproc: list[int] = []

    count = 0
    # Phase 24: use the bounded-cardinality chart iterator. Without it
    # an ALL-range request walks every raw sample, decodes tstat for
    # each of them, and ends up taking several minutes on a multi-day
    # capture. ``_chart_window_iter`` caps the emitted count to
    # ``_CHART_TARGET_POINTS`` so the response fits in a few seconds.
    for s in _chart_window_iter(sess, from_epoch, to_epoch):
        timeline.append(s.curtime)
        intervals.append(s.interval)
        nrcpu_series.append(s.nrcpu)
        cpu_ticks.append(_cpu_ticks(s))
        rmem.append(_rmem_kb(s))
        vmem.append(_vmem_kb(s))
        dsk_r.append(_disk_read_sectors(s))
        dsk_w.append(_disk_write_sectors(s))
        tcp_s.append(_net_tcp_send(s))
        tcp_r.append(_net_tcp_recv(s))
        udp_s.append(_net_udp_send(s))
        udp_r.append(_net_udp_recv(s))
        ntask.append(s.ntask)
        totproc.append(s.totproc)
        _drop_view_caches(s)
        count += 1

    # ``ncpu`` varies per sample in theory but in practice the value is
    # constant throughout a capture. Surface the last observed non null value
    # so the UI can convert CPU ticks to percentage without re-fetching the
    # per sample nrcpu series.
    ncpu = next((v for v in reversed(nrcpu_series) if v), None)

    return {
        "session": session,
        "count": count,
        "total": sess.sample_count(),
        "range": {"from": from_epoch, "to": to_epoch},
        "meta": {
            "hertz": sess.rawlog.header.hertz,
            "ncpu": ncpu,
        },
        "timeline": timeline,
        "intervals": intervals,
        "nrcpu": nrcpu_series,
        "cpu": {"ticks": cpu_ticks},
        "mem": {"rss_kb": rmem, "vsz_kb": vmem},
        "dsk": {"read_sectors": dsk_r, "write_sectors": dsk_w},
        "net": {
            "tcp_sent": tcp_s,
            "tcp_recv": tcp_r,
            "udp_sent": udp_s,
            "udp_recv": udp_r,
        },
        "tasks": {"ntask": ntask, "totproc": totproc},
    }


def _iter_window(sess, from_epoch: int | None, to_epoch: int | None):
    """Yield samples in the window without materializing them all at once.

    For lazy sessions we use the offset index's bisect to find the
    index range, then yield one view at a time. For eager sessions
    we fall back to the list-based filter.
    """
    if sess.is_lazy and sess.index is not None:
        if from_epoch is None and to_epoch is None:
            n = len(sess.rawlog)
            for i in range(n):
                yield sess.rawlog[i]
            return
        lo = from_epoch if from_epoch is not None else -(1 << 62)
        hi = to_epoch if to_epoch is not None else (1 << 62)
        lo_i, hi_i = sess.index.slice_by_time(lo, hi)
        for i in range(lo_i, hi_i):
            yield sess.rawlog[i]
        return
    samples_list = sess.rawlog.samples
    if from_epoch is None and to_epoch is None:
        for s in samples_list:
            yield s
        return
    lo = from_epoch if from_epoch is not None else -(1 << 62)
    hi = to_epoch if to_epoch is not None else (1 << 62)
    for s in samples_list:
        if lo <= s.curtime <= hi:
            yield s


# Windows this wide or wider are always downsampled by ``_chart_window_iter``.
# Below the threshold we keep per-sample resolution so short-range tooltips
# still show every data point the capture recorded.
_CHART_DOWNSAMPLE_MIN_WINDOW_SECONDS = 60

# Upper bound on how many samples the chart endpoints emit regardless of
# the window. Phase 24 fix: on a multi-day capture at 60-second resolution
# a fixed 60-second bucket still emits thousands of points, which forced
# the chart endpoints to sstat-inflate thousands of samples per request
# and time out the dashboard. Capping at ``_CHART_TARGET_POINTS`` and
# growing ``bucket_step`` with the window keeps decoder cost bounded.
_CHART_TARGET_POINTS = 600


def _chart_window_iter(
    sess,
    from_epoch: int | None,
    to_epoch: int | None,
    *,
    bucket_step: int = 60,
    target_points: int = _CHART_TARGET_POINTS,
):
    """Yield samples for a chart response at bounded cardinality.

    Phase 23 + 24 chart fast path: over a wide window the four
    ``/api/samples/system_*`` endpoints don't need every raw sample to
    draw their line. A single representative per bucket (the first
    sample whose ``curtime`` lands inside that bucket) gives the
    aggregate grid resolution without ever materializing all samples,
    and the bucket step grows with the window so the response never
    exceeds ``target_points`` regardless of capture length.

    The effective bucket is the larger of:

    - ``bucket_step`` (default 60 s, the minimum useful resolution for
      operator charts), and
    - ``ceil(window_seconds / target_points)``, so an ALL-range query
      on a week-long capture emits about ``target_points`` samples
      instead of thousands.

    Narrow windows (shorter than ``_CHART_DOWNSAMPLE_MIN_WINDOW_SECONDS``)
    fall through to the per-sample ``_iter_window`` so short-range
    requests keep their fine resolution.

    The tstat / sstat caches on each yielded view are *not* dropped
    here. Callers still have to call ``_drop_view_caches`` once they
    are done with the view.
    """
    if (
        from_epoch is not None
        and to_epoch is not None
        and to_epoch - from_epoch < _CHART_DOWNSAMPLE_MIN_WINDOW_SECONDS
    ):
        yield from _iter_window(sess, from_epoch, to_epoch)
        return

    window_seconds = _estimate_window_seconds(sess, from_epoch, to_epoch)
    effective_step = bucket_step
    if window_seconds is not None and target_points > 0:
        min_step = _ceil_div(window_seconds, target_points)
        if min_step > effective_step:
            effective_step = min_step

    last_bucket: int | None = None
    for view in _iter_window(sess, from_epoch, to_epoch):
        bucket = view.curtime // effective_step
        if bucket == last_bucket:
            continue
        last_bucket = bucket
        yield view


def _estimate_window_seconds(
    sess,
    from_epoch: int | None,
    to_epoch: int | None,
) -> int | None:
    """Return window width in seconds, filling in missing bounds from ``sess``."""
    if from_epoch is not None and to_epoch is not None:
        return max(0, to_epoch - from_epoch)
    first = last = None
    if getattr(sess, "is_lazy", False) and getattr(sess, "index", None) is not None:
        first = sess.index.first_time()
        last = sess.index.last_time()
    else:
        samples = getattr(getattr(sess, "rawlog", None), "samples", None)
        if samples:
            first = samples[0].curtime
            last = samples[-1].curtime
    if first is None or last is None:
        return None
    lo = from_epoch if from_epoch is not None else first
    hi = to_epoch if to_epoch is not None else last
    return max(0, hi - lo)


def _ceil_div(num: int, den: int) -> int:
    if den <= 0:
        return num
    return -(-num // den)


def _drop_view_caches(view) -> None:
    """Release tstat / sstat caches on a SampleView once we're done.

    The LRU keeps the view object alive so a follow-up request for the
    same sample does not re-decode, but we do not need the ``processes``
    list or the ``_bundle`` after the /samples route has already rolled
    them into its output arrays. ``getattr`` dance keeps this a no-op
    on eager ``Sample`` dataclasses, which have no ``_processes_cache``
    attribute.
    """
    if hasattr(view, "_processes_cache"):
        try:
            view._processes_cache = None
        except AttributeError:
            pass
    if hasattr(view, "_bundle"):
        try:
            view._bundle = None
        except AttributeError:
            pass


def _serialize_percpu(p) -> dict:
    return {
        "cpunr": p.cpunr,
        "stime": p.stime,
        "utime": p.utime,
        "ntime": p.ntime,
        "itime": p.itime,
        "wtime": p.wtime,
        "Itime": p.Itime,
        "Stime": p.Stime,
        "steal": p.steal,
        "guest": p.guest,
    }


@router.get("/samples/system_cpu")
def system_cpu(
    session: str = Query(...),
    from_: str | None = Query(None, alias="from", description="ISO8601 lower bound"),
    to: str | None = Query(None, description="ISO8601 upper bound"),
) -> dict:
    """Time series of system wide CPU counters decoded from sstat.cpustat.

    Values are raw clock ticks per field (user, system, idle, ...). Combined
    with ``hertz`` from the rawheader and each sample's ``interval`` (seconds),
    clients can derive the usual "CPU %" view without re-fetching the raw
    layout. Samples whose cpustat could not be decoded are reported via
    ``missing_samples`` and omitted from the array.
    """
    sess = get_store().require(session)
    from_epoch = parse_iso_epoch(from_, field="from")
    to_epoch = parse_iso_epoch(to, field="to")
    cache = get_response_cache()
    key = (session, "system_cpu", from_epoch, to_epoch)
    return cache.get_or_compute(
        key,
        lambda: _system_cpu_impl(sess, session, from_epoch, to_epoch),
    )


def _system_cpu_impl(sess, session: str, from_epoch: int | None, to_epoch: int | None) -> dict:
    entries: list[dict] = []
    missing = 0
    hertz = sess.rawlog.header.hertz
    ncpu_last: int | None = None

    for s in _chart_window_iter(sess, from_epoch, to_epoch):
        cpu = s.system_cpu
        _drop_view_caches(s)
        if cpu is None:
            missing += 1
            continue
        ncpu_last = cpu.nrcpu
        entries.append(
            {
                "curtime": s.curtime,
                "interval": s.interval,
                "nrcpu": cpu.nrcpu,
                "devint": cpu.devint,
                "csw": cpu.csw,
                "nprocs": cpu.nprocs,
                "lavg1": cpu.lavg1,
                "lavg5": cpu.lavg5,
                "lavg15": cpu.lavg15,
                "all": _serialize_percpu(cpu.all),
                "cpus": [_serialize_percpu(p) for p in cpu.cpus],
            }
        )

    return {
        "session": session,
        "hertz": hertz,
        "ncpu": ncpu_last,
        "range": {"from": from_epoch, "to": to_epoch},
        "count": len(entries),
        "missing_samples": missing,
        "samples": entries,
    }


def _serialize_interface(i) -> dict:
    return {
        "name": i.name,
        "type": i.type,
        "speed_mbps": i.speed_mbps,
        "duplex": i.duplex,
        "rbyte": i.rbyte,
        "rpack": i.rpack,
        "rerrs": i.rerrs,
        "rdrop": i.rdrop,
        "sbyte": i.sbyte,
        "spack": i.spack,
        "serrs": i.serrs,
        "sdrop": i.sdrop,
    }


@router.get("/samples/system_network")
def system_network(
    session: str = Query(...),
    from_: str | None = Query(None, alias="from", description="ISO8601 lower bound"),
    to: str | None = Query(None, description="ISO8601 upper bound"),
) -> dict:
    """Time series of per interface network counters decoded from sstat.intfstat.

    Values are monotonically increasing kernel counters (bytes, packets,
    errors, drops) per interface. Clients compute per sample deltas and
    divide by ``interval`` to derive throughput and packets per second.
    """
    sess = get_store().require(session)
    from_epoch = parse_iso_epoch(from_, field="from")
    to_epoch = parse_iso_epoch(to, field="to")
    cache = get_response_cache()
    key = (session, "system_network", from_epoch, to_epoch)
    return cache.get_or_compute(
        key,
        lambda: _system_network_impl(sess, session, from_epoch, to_epoch),
    )


def _system_network_impl(sess, session: str, from_epoch: int | None, to_epoch: int | None) -> dict:
    entries: list[dict] = []
    missing = 0

    for s in _chart_window_iter(sess, from_epoch, to_epoch):
        net = s.system_network
        _drop_view_caches(s)
        if net is None:
            entries.append(
                {
                    "curtime": s.curtime,
                    "interval": s.interval,
                    "nrintf": 0,
                    "interfaces": [],
                }
            )
            missing += 1
            continue
        entries.append(
            {
                "curtime": s.curtime,
                "interval": s.interval,
                "nrintf": net.nrintf,
                "interfaces": [_serialize_interface(i) for i in net.interfaces],
            }
        )

    # Union of interface names across the window so the UI can build a
    # dropdown without re-scanning the series.
    seen: list[str] = []
    for entry in entries:
        for i in entry["interfaces"]:
            if i["name"] not in seen:
                seen.append(i["name"])

    return {
        "session": session,
        "range": {"from": from_epoch, "to": to_epoch},
        "count": len(entries),
        "missing_samples": missing,
        "interfaces": seen,
        "samples": entries,
    }


def _serialize_disk(d) -> dict:
    return {
        "name": d.name,
        "nread": d.nread,
        "nrsect": d.nrsect,
        "nwrite": d.nwrite,
        "nwsect": d.nwsect,
        "io_ms": d.io_ms,
        "avque": d.avque,
        "ndisc": d.ndisc,
        "ndsect": d.ndsect,
        "inflight": d.inflight,
        "kind": d.kind,
    }


@router.get("/samples/system_disk")
def system_disk(
    session: str = Query(...),
    from_: str | None = Query(None, alias="from", description="ISO8601 lower bound"),
    to: str | None = Query(None, description="ISO8601 upper bound"),
) -> dict:
    """Time series of per device disk counters decoded from sstat.dskstat.

    Values are monotonically increasing kernel counters (reads, sectors, I/O
    milliseconds) per block device. Clients compute per sample deltas and
    divide by ``interval`` to derive IOPS or throughput.
    """
    sess = get_store().require(session)
    from_epoch = parse_iso_epoch(from_, field="from")
    to_epoch = parse_iso_epoch(to, field="to")
    cache = get_response_cache()
    key = (session, "system_disk", from_epoch, to_epoch)
    return cache.get_or_compute(
        key,
        lambda: _system_disk_impl(sess, session, from_epoch, to_epoch),
    )


def _system_disk_impl(sess, session: str, from_epoch: int | None, to_epoch: int | None) -> dict:
    entries: list[dict] = []
    missing = 0

    for s in _chart_window_iter(sess, from_epoch, to_epoch):
        disk = s.system_disk
        _drop_view_caches(s)
        if disk is None:
            missing += 1
            continue
        entries.append(
            {
                "curtime": s.curtime,
                "interval": s.interval,
                "disks": [_serialize_disk(d) for d in disk.disks],
                "mdds": [_serialize_disk(d) for d in disk.mdds],
                "lvms": [_serialize_disk(d) for d in disk.lvms],
            }
        )

    # Advertise the union of device names actually present across samples so
    # the UI can build the device picker without re-scanning the series.
    def _collect_names(kind: str) -> list[str]:
        seen: list[str] = []
        for entry in entries:
            for d in entry[kind]:
                if d["name"] not in seen:
                    seen.append(d["name"])
        return seen

    return {
        "session": session,
        "range": {"from": from_epoch, "to": to_epoch},
        "count": len(entries),
        "missing_samples": missing,
        "devices": {
            "disks": _collect_names("disks"),
            "mdds": _collect_names("mdds"),
            "lvms": _collect_names("lvms"),
        },
        "samples": entries,
    }


@router.get("/samples/system_memory")
def system_memory(
    session: str = Query(...),
    from_: str | None = Query(None, alias="from", description="ISO8601 lower bound"),
    to: str | None = Query(None, description="ISO8601 upper bound"),
) -> dict:
    """Time series of system wide memory counters decoded from sstat.memstat.

    See ``atop_web/parser/reader.py`` for the offset derivation. All counters
    are in OS pages; ``pagesize`` is returned alongside so the client can
    derive MiB or GiB. Samples whose sstat layout could not be decoded are
    omitted from the ``samples`` array but still counted via
    ``missing_samples``.
    """
    sess = get_store().require(session)
    from_epoch = parse_iso_epoch(from_, field="from")
    to_epoch = parse_iso_epoch(to, field="to")
    cache = get_response_cache()
    key = (session, "system_memory", from_epoch, to_epoch)
    return cache.get_or_compute(
        key,
        lambda: _system_memory_impl(sess, session, from_epoch, to_epoch),
    )


def _system_memory_impl(sess, session: str, from_epoch: int | None, to_epoch: int | None) -> dict:
    entries: list[dict] = []
    missing = 0
    pagesize = sess.rawlog.header.pagesize or 0
    swap_configured = False

    for s in _chart_window_iter(sess, from_epoch, to_epoch):
        mem = s.system_memory
        _drop_view_caches(s)
        if mem is None:
            missing += 1
            continue
        if mem.totswap > 0 or mem.freeswap > 0 or mem.swapcached > 0:
            swap_configured = True
        if mem.pagesize and not pagesize:
            pagesize = mem.pagesize
        entries.append(
            {
                "curtime": s.curtime,
                "physmem": mem.physmem,
                "freemem": mem.freemem,
                "buffermem": mem.buffermem,
                "slabmem": mem.slabmem,
                "cachemem": mem.cachemem,
                "availablemem": mem.availablemem,
                "totswap": mem.totswap,
                "freeswap": mem.freeswap,
                "swapcached": mem.swapcached,
            }
        )

    return {
        "session": session,
        "pagesize": pagesize,
        "range": {"from": from_epoch, "to": to_epoch},
        "count": len(entries),
        "missing_samples": missing,
        "swap_configured": swap_configured,
        "samples": entries,
    }
