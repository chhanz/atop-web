"""GET /api/samples: CPU, memory, disk and network time series."""

from __future__ import annotations

from fastapi import APIRouter, Query

from atop_web.api.sessions import get_store
from atop_web.api.timerange import filter_samples, parse_iso_epoch
from atop_web.parser.reader import Sample

router = APIRouter()


def _cpu_ticks(sample: Sample) -> int:
    return sum(p.utime + p.stime for p in sample.processes)


def _rmem_kb(sample: Sample) -> int:
    return sum(max(p.rmem_kb, 0) for p in sample.processes if p.isproc)


def _vmem_kb(sample: Sample) -> int:
    return sum(max(p.vmem_kb, 0) for p in sample.processes if p.isproc)


def _disk_read_sectors(sample: Sample) -> int:
    return sum(max(p.rsz, 0) for p in sample.processes)


def _disk_write_sectors(sample: Sample) -> int:
    return sum(max(p.wsz, 0) for p in sample.processes)


def _net_tcp_send(sample: Sample) -> int:
    return sum(max(p.tcpsnd, 0) for p in sample.processes)


def _net_tcp_recv(sample: Sample) -> int:
    return sum(max(p.tcprcv, 0) for p in sample.processes)


def _net_udp_send(sample: Sample) -> int:
    return sum(max(p.udpsnd, 0) for p in sample.processes)


def _net_udp_recv(sample: Sample) -> int:
    return sum(max(p.udprcv, 0) for p in sample.processes)


@router.get("/samples")
def samples(
    session: str = Query(...),
    from_: str | None = Query(None, alias="from", description="ISO8601 lower bound"),
    to: str | None = Query(None, description="ISO8601 upper bound"),
) -> dict:
    sess = get_store().require(session)
    rawlog = sess.rawlog

    from_epoch = parse_iso_epoch(from_, field="from")
    to_epoch = parse_iso_epoch(to, field="to")
    subset = filter_samples(rawlog.samples, from_epoch, to_epoch)

    timeline = [s.curtime for s in subset]
    intervals = [s.interval for s in subset]
    nrcpu_series = [s.nrcpu for s in subset]

    cpu_ticks = [_cpu_ticks(s) for s in subset]
    rmem = [_rmem_kb(s) for s in subset]
    vmem = [_vmem_kb(s) for s in subset]
    dsk_r = [_disk_read_sectors(s) for s in subset]
    dsk_w = [_disk_write_sectors(s) for s in subset]
    tcp_s = [_net_tcp_send(s) for s in subset]
    tcp_r = [_net_tcp_recv(s) for s in subset]
    udp_s = [_net_udp_send(s) for s in subset]
    udp_r = [_net_udp_recv(s) for s in subset]
    ntask = [s.ntask for s in subset]
    totproc = [s.totproc for s in subset]

    # ``ncpu`` varies per sample in theory but in practice the value is
    # constant throughout a capture. Surface the last observed non null value
    # so the UI can convert CPU ticks to percentage without re-fetching the
    # per sample nrcpu series.
    ncpu = next((v for v in reversed(nrcpu_series) if v), None)

    return {
        "session": session,
        "count": len(subset),
        "total": len(rawlog.samples),
        "range": {"from": from_epoch, "to": to_epoch},
        "meta": {
            "hertz": rawlog.header.hertz,
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
    rawlog = sess.rawlog

    from_epoch = parse_iso_epoch(from_, field="from")
    to_epoch = parse_iso_epoch(to, field="to")
    subset = filter_samples(rawlog.samples, from_epoch, to_epoch)

    entries: list[dict] = []
    missing = 0
    hertz = rawlog.header.hertz
    ncpu_last: int | None = None

    for s in subset:
        cpu = s.system_cpu
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
    rawlog = sess.rawlog

    from_epoch = parse_iso_epoch(from_, field="from")
    to_epoch = parse_iso_epoch(to, field="to")
    subset = filter_samples(rawlog.samples, from_epoch, to_epoch)

    entries: list[dict] = []
    missing = 0

    for s in subset:
        net = s.system_network
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
    rawlog = sess.rawlog

    from_epoch = parse_iso_epoch(from_, field="from")
    to_epoch = parse_iso_epoch(to, field="to")
    subset = filter_samples(rawlog.samples, from_epoch, to_epoch)

    entries: list[dict] = []
    missing = 0

    for s in subset:
        disk = s.system_disk
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
    rawlog = sess.rawlog

    from_epoch = parse_iso_epoch(from_, field="from")
    to_epoch = parse_iso_epoch(to, field="to")
    subset = filter_samples(rawlog.samples, from_epoch, to_epoch)

    entries: list[dict] = []
    missing = 0
    pagesize = rawlog.header.pagesize or 0
    swap_configured = False

    for s in subset:
        mem = s.system_memory
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
