"""GET /api/processes: per sample tstat process list.

The response includes derived units (cpu_pct, rmem_mb, vmem_mb, dsk_*_mb)
alongside the raw atop counters so the UI does not have to redo the math.
Sorting supports every column the UI exposes and is stable: ties fall back
to pid to keep row order deterministic across reloads.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from atop_web.api.cache import get_response_cache
from atop_web.api.sessions import get_store
from atop_web.api.timerange import parse_iso_epoch
from atop_web.parser.reader import Process

router = APIRouter()


SORTABLE = ("pid", "name", "state", "nthr", "cpu", "rmem", "vmem", "dsk", "net")

SECTORS_TO_MB = 512 / (1024 * 1024)


def _cpu_pct(p: Process, hertz: int | None, interval_sec: int, ncpu: int | None) -> float | None:
    if not hertz or hertz <= 0 or interval_sec <= 0 or not ncpu or ncpu <= 0:
        return None
    total_ticks = max(0, p.utime + p.stime)
    denom = hertz * interval_sec * ncpu
    if denom <= 0:
        return None
    return round(total_ticks / denom * 100.0, 4)


def _serialize(
    p: Process,
    *,
    hertz: int | None,
    interval_sec: int,
    ncpu: int | None,
) -> dict:
    total_ticks = p.utime + p.stime
    return {
        "pid": p.pid,
        "tgid": p.tgid,
        "ppid": p.ppid,
        "name": p.name,
        "cmdline": p.cmdline,
        "state": p.state,
        "nthr": p.nthr,
        "isproc": p.isproc,
        # raw counters (kept verbatim for power users / regressions)
        "cpu_ticks": total_ticks,
        "utime": p.utime,
        "stime": p.stime,
        "rmem_kb": p.rmem_kb,
        "vmem_kb": p.vmem_kb,
        "dsk_read_sectors": p.rsz,
        "dsk_write_sectors": p.wsz,
        "dsk_rio": p.rio,
        "dsk_wio": p.wio,
        "tcp_sent": p.tcpsnd,
        "tcp_recv": p.tcprcv,
        "udp_sent": p.udpsnd,
        "udp_recv": p.udprcv,
        # derived units
        "cpu_pct": _cpu_pct(p, hertz, interval_sec, ncpu),
        "rmem_mb": round(max(p.rmem_kb, 0) / 1024.0, 4),
        "vmem_mb": round(max(p.vmem_kb, 0) / 1024.0, 4),
        "dsk_read_mb": round(max(p.rsz, 0) * SECTORS_TO_MB, 4),
        "dsk_write_mb": round(max(p.wsz, 0) * SECTORS_TO_MB, 4),
    }


def _sort_key(sort_by: str):
    if sort_by == "pid":
        return lambda p: (p.pid, p.pid)
    if sort_by == "name":
        return lambda p: (p.name.lower(), p.pid)
    if sort_by == "state":
        return lambda p: (p.state or "", p.pid)
    if sort_by == "nthr":
        return lambda p: (p.nthr, p.pid)
    if sort_by == "cpu":
        return lambda p: (p.utime + p.stime, p.pid)
    if sort_by == "rmem":
        return lambda p: (p.rmem_kb, p.pid)
    if sort_by == "vmem":
        return lambda p: (p.vmem_kb, p.pid)
    if sort_by == "dsk":
        return lambda p: (p.rsz + p.wsz, p.pid)
    if sort_by == "net":
        return lambda p: (
            p.tcpsnd + p.tcprcv + p.udpsnd + p.udprcv,
            p.pid,
        )
    raise HTTPException(status_code=400, detail=f"unsupported sort_by: {sort_by}")


@router.get("/processes")
def processes(
    session: str = Query(...),
    time: int | None = Query(None, description="epoch seconds; defaults to the last sample in range"),
    index: int | None = Query(None, ge=0, description="sample index within the filtered range"),
    limit: int = Query(200, ge=1, le=10000),
    sort_by: str = Query("cpu", description="column to sort by"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    from_: str | None = Query(None, alias="from", description="ISO8601 lower bound"),
    to: str | None = Query(None, description="ISO8601 upper bound"),
) -> dict:
    if sort_by not in SORTABLE:
        raise HTTPException(
            status_code=400,
            detail=f"sort_by must be one of {list(SORTABLE)!r}",
        )

    sess = get_store().require(session)
    if sess.sample_count() == 0:
        raise HTTPException(status_code=404, detail="no samples in session")

    from_epoch = parse_iso_epoch(from_, field="from")
    to_epoch = parse_iso_epoch(to, field="to")

    cache = get_response_cache()
    cache_key = (
        session,
        "processes",
        from_epoch,
        to_epoch,
        time,
        index,
        limit,
        sort_by,
        order,
    )

    def _build_processes():
        return _processes_impl(
            sess, session, from_epoch, to_epoch, time, index, limit, sort_by, order
        )

    try:
        return cache.get_or_compute(cache_key, _build_processes)
    except HTTPException:
        raise


def _processes_impl(
    sess,
    session: str,
    from_epoch: int | None,
    to_epoch: int | None,
    time: int | None,
    index: int | None,
    limit: int,
    sort_by: str,
    order: str,
) -> dict:
    subset = sess.samples_in_range(from_epoch, to_epoch)
    if not subset:
        raise HTTPException(status_code=404, detail="no samples in range")

    if index is not None:
        if index >= len(subset):
            raise HTTPException(status_code=404, detail="index out of range")
        sample = subset[index]
    elif time is not None:
        sample = min(subset, key=lambda s: abs(s.curtime - time))
    else:
        sample = subset[-1]

    hertz = sess.rawlog.header.hertz
    interval_sec = sample.interval
    ncpu = sample.nrcpu

    sort_callable = _sort_key(sort_by)
    reverse = order == "desc"
    # ``sorted`` is stable; the key appends pid as the secondary component
    # so ties resolve consistently regardless of the input order.
    procs = sorted(sample.processes, key=sort_callable, reverse=reverse)[:limit]

    return {
        "session": session,
        "curtime": sample.curtime,
        "interval": interval_sec,
        "ndeviat": sample.ndeviat,
        "nactproc": sample.nactproc,
        "sort_by": sort_by,
        "order": order,
        "count": len(procs),
        "meta": {
            "hertz": hertz,
            "ncpu": ncpu,
            "interval_sec": interval_sec,
        },
        "processes": [
            _serialize(p, hertz=hertz, interval_sec=interval_sec, ncpu=ncpu)
            for p in procs
        ],
    }
