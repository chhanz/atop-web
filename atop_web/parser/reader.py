"""atop rawlog reader with multi version dispatch.

This module loads a per atop version CDEF describing the on disk layout
and iterates over samples in a rawlog file. The parser reads:

- the fixed sized ``rawheader`` at the start of the file
- a sequence of samples, each composed of a ``rawrecord`` header, a zlib
  compressed ``sstat`` blob, ``ndeviat`` zlib compressed ``tstat``
  structures, and (on atop >= 2.12) optional cgroup payloads

Two atop rawlog revisions are supported:

* atop 2.12.x (AL2023 default, native build)
* atop 2.7.x  (AL2 EPEL default)

The right CDEF is chosen by reading ``tstatlen`` and ``sstatlen`` from the
``rawheader`` and looking them up in ``VERSION_TABLE``. Unknown combinations
raise ``RawLogError`` rather than silently skipping fields.

System level statistics (sstat) depend on a large number of per host sized
arrays declared in photosyst.h (MAXCPU, MAXDSK, etc.) that differ between
atop builds. The reader therefore decompresses sstat and decodes only the
fields we need, using per version offsets measured against rawlogs produced
by the target OS toolchain (not ``sizeof`` on the current host).

Per process tstat statistics are decoded in full by the CDEF layout.
"""

from __future__ import annotations

import io
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

from dissect.cstruct import cstruct

from atop_web.parser.decompress import DecompressError, inflate

if TYPE_CHECKING:
    from atop_web.parser.index import SampleIndex


def _lazy_default() -> bool:
    """Return the default ``lazy=`` value for parse entry points.

    Reads ``ATOP_LAZY`` fresh on every call so tests (and operators)
    can flip the gate at runtime without reloading the module.
    """
    return os.environ.get("ATOP_LAZY", "1") != "0"

MAGIC = 0xFEEDBEEF

_LAYOUTS_DIR = Path(__file__).parent / "layouts"


class RawLogError(ValueError):
    """Raised for structural errors in a rawlog file."""


# ---------------------------------------------------------------------------
# Version layout specifications
#
# Each atop rawlog revision is captured in a VersionSpec. Offsets and sizes
# come from measuring a real rawlog written by that atop version on the
# target OS toolchain. Do not recompute them from ``sizeof`` on the current
# host: MAXCPU / MAXDSK / glibc padding differ between AL2 and AL2023, so a
# native compile gives wrong numbers for cross-build rawlogs.


@dataclass(frozen=True)
class VersionSpec:
    """Static layout description for one atop rawlog revision."""

    name: str
    cdef_filename: str
    tstat_size: int
    sstat_size: int

    # Whether struct rawrecord carries the cgroup fields (ccomplen /
    # coriglen / ncgpids / icomplen). atop 2.12 does; atop 2.7 does not.
    record_has_cgroup_fields: bool

    # memstat layout inside sstat.
    memstat_offset: int
    memstat_physmem_idx: int
    memstat_freemem_idx: int
    memstat_buffermem_idx: int
    memstat_slabmem_idx: int
    memstat_cachemem_idx: int
    memstat_totswap_idx: int
    memstat_freeswap_idx: int
    memstat_swapcached_idx: int
    # 2.7 has no availablemem; set to ``None`` to skip decoding.
    memstat_availablemem_idx: Optional[int]
    # Number of count_t fields we need to read past ``memstat_offset``.
    memstat_min_tail_fields: int

    # cpustat layout.
    cpustat_max_cpu: int
    percpu_size: int
    percpu_all_offset: int
    percpu_array_offset: int

    # dskstat layout.
    dskstat_offset: int
    perdsk_size: int
    # 2.7 perdsk is 112 B (cfuture[2]); 2.12 is 128 B (inflight + cfuture[3]).
    perdsk_has_inflight: bool
    perdsk_name_size: int
    dsk_array_offset: int
    mdd_array_offset: int
    lvm_array_offset: int
    max_dsk: int
    max_mdd: int
    max_lvm: int

    # intfstat layout.
    intfstat_offset: int
    perintf_size: int
    perintf_name_size: int
    perintf_array_offset: int
    perintf_max: int

    # Lazily constructed cstruct for this version (cached).
    cs_holder: dict = field(default_factory=dict)

    @property
    def cs(self) -> cstruct:
        cs = self.cs_holder.get("cs")
        if cs is None:
            cs = cstruct(endian="<")
            cs.load((_LAYOUTS_DIR / self.cdef_filename).read_text())
            self.cs_holder["cs"] = cs
        return cs

    @property
    def rawheader_size(self) -> int:
        return len(self.cs.rawheader)

    @property
    def rawrecord_size(self) -> int:
        return len(self.cs.rawrecord)

    @property
    def dskstat_size(self) -> int:
        return self.lvm_array_offset + self.perdsk_size * self.max_lvm

    @property
    def intfstat_size(self) -> int:
        return self.perintf_array_offset + self.perintf_size * self.perintf_max

    @property
    def memstat_min_total(self) -> int:
        return self.memstat_offset + self.memstat_min_tail_fields * 8


# atop 2.12.x (AL2023 native). memstat / dskstat / intfstat offsets were
# measured against ``atop_20260427``: memstat by locating ``physmem`` =
# sysconf(_PHYS_PAGES), dskstat by scanning for "nvme0n1", intfstat by
# scanning for "docker0" and stepping back.
SPEC_2_12 = VersionSpec(
    name="atop_2_12",
    cdef_filename="atop_2_12.cdef",
    tstat_size=968,
    sstat_size=1_064_016,
    record_has_cgroup_fields=True,
    memstat_offset=344_312,
    memstat_physmem_idx=0,
    memstat_freemem_idx=1,
    memstat_buffermem_idx=2,
    memstat_slabmem_idx=3,
    memstat_cachemem_idx=4,
    memstat_totswap_idx=6,
    memstat_freeswap_idx=7,
    memstat_swapcached_idx=26,
    memstat_availablemem_idx=43,
    memstat_min_tail_fields=44,
    cpustat_max_cpu=2048,
    percpu_size=168,
    percpu_all_offset=80,
    percpu_array_offset=248,
    dskstat_offset=601_688,
    perdsk_size=128,
    perdsk_has_inflight=True,
    perdsk_name_size=32,
    dsk_array_offset=16,
    mdd_array_offset=16 + 128 * 1024,
    lvm_array_offset=16 + 128 * 1024 + 128 * 256,
    max_dsk=1024,
    max_mdd=256,
    max_lvm=2048,
    intfstat_offset=345_664,
    perintf_size=272,
    perintf_name_size=16,
    perintf_array_offset=8,
    perintf_max=128,
)


# atop 2.7.x (AL2 EPEL, gcc 7.3.1 / glibc 2.26). Offsets measured from
# struct definitions in /tmp/atop-al2-src/atop-2.7.1/photosyst.h with AL2
# alignment. See the Phase 16 design notes for the derivation.
#
# sstat layout (2.7):
#   cpu       @      0  (344_312 B, same as 2.12: MAXCPU=2048, percpu=168)
#   mem       @ 344_312 (336 B: 42 count_t)
#   net       @ 344_648 (936 B: struct netstat)
#   intf      @ 345_584 (34_824 B: 8 + 272 * 128)
#   memnuma   @ 380_408 (90_120 B: 8 + 88 * MAXNUMA=1024)
#   cpunuma   @ 470_528 (81_928 B: 8 + 80 * MAXNUMA=1024)
#   dsk       @ 552_456 (372_752 B: 12 + 4 + 112 * (1024 + 256 + 2048))
#   nfs       @ 925_208 (17_192 B)
#   cfs       @ 942_400 (7_176 B)
#   psi       @ 949_576 (128 B)
#   gpu       @ 949_704 (2_824 B: 8 + 88 * MAXGPU=32)
#   ifb       @ 952_528 (1_800 B: 8 + 56 * MAXIBPORT=32)
#   www       @ 954_328 (32 B)
# Total = 954_360.
#
# memstat (2.7) field indices (see photosyst.h struct memstat):
#   0 physmem, 1 freemem, 2 buffermem, 3 slabmem, 4 cachemem, 5 cachedrt,
#   6 totswap, 7 freeswap, 8..14 pgscans..committed, 15..17 shmem*,
#   18 slabreclaim, 19..21 *hugepage*, 22 vmwballoon, 23 zfsarcsize,
#   24 swapcached, 25 ksmsharing, 26 ksmshared, 27..32 zsw*..numamigrate,
#   33..41 cfuture[0..8]  -> 42 count_t = 336 B.
# There is no availablemem in 2.7. Note the swapcached index (24) differs
# from 2.12 (26): 2.12 inserts two extra page counters earlier in the
# struct, so the same name lives two slots further down.
#
# dskstat (2.7):
#   0   int ndsk
#   4   int nmdd
#   8   int nlvm
#   12  int _pad (compiler inserts 4 B to align count_t in perdsk)
#   16  perdsk dsk[MAXDSK=1024]      (112 B each)
#   16 + 112*1024 = 114_704 perdsk mdd[MAXMDD=256]   (112 B each)
#   114_704 + 112*256 = 143_376 perdsk lvm[MAXLVM=2048] (112 B each)
# Total = 143_376 + 112*2048 = 372_752.
#
# perdsk (2.7): 112 B = name[32] + 9 count_t (nread..ndsect) + cfuture[2].
# There is no inflight field.
SPEC_2_7 = VersionSpec(
    name="atop_2_7",
    cdef_filename="atop_2_7.cdef",
    tstat_size=840,
    sstat_size=954_360,
    record_has_cgroup_fields=False,
    memstat_offset=344_312,
    memstat_physmem_idx=0,
    memstat_freemem_idx=1,
    memstat_buffermem_idx=2,
    memstat_slabmem_idx=3,
    memstat_cachemem_idx=4,
    memstat_totswap_idx=6,
    memstat_freeswap_idx=7,
    memstat_swapcached_idx=24,
    memstat_availablemem_idx=None,
    memstat_min_tail_fields=42,
    cpustat_max_cpu=2048,
    percpu_size=168,
    percpu_all_offset=80,
    percpu_array_offset=248,
    dskstat_offset=552_456,
    perdsk_size=112,
    perdsk_has_inflight=False,
    perdsk_name_size=32,
    dsk_array_offset=16,
    mdd_array_offset=16 + 112 * 1024,
    lvm_array_offset=16 + 112 * 1024 + 112 * 256,
    max_dsk=1024,
    max_mdd=256,
    max_lvm=2048,
    intfstat_offset=345_584,
    perintf_size=272,
    perintf_name_size=16,
    perintf_array_offset=8,
    perintf_max=128,
)


# Map (tstatlen, sstatlen) -> VersionSpec. These two fields in the rawheader
# together uniquely identify the atop rawlog revision for the versions we
# support. Keep the table small: fallbacks / fuzzy matching risk silently
# decoding a future revision as an older one.
VERSION_TABLE: dict[tuple[int, int], VersionSpec] = {
    (SPEC_2_12.tstat_size, SPEC_2_12.sstat_size): SPEC_2_12,
    (SPEC_2_7.tstat_size, SPEC_2_7.sstat_size): SPEC_2_7,
}


def _select_spec(tstatlen: int, sstatlen: int) -> VersionSpec:
    spec = VERSION_TABLE.get((tstatlen, sstatlen))
    if spec is None:
        supported = ", ".join(
            f"{s.name} (tstat={s.tstat_size}, sstat={s.sstat_size})"
            for s in (SPEC_2_12, SPEC_2_7)
        )
        raise RawLogError(
            f"unsupported atop version: tstat={tstatlen}, sstat={sstatlen}. "
            f"Supported: {supported}"
        )
    return spec


# Module level constants kept for backwards compatibility with existing tests
# and callers that import them directly. They reflect the 2.12 layout.
RAWHEADER_SIZE = SPEC_2_12.rawheader_size
RAWRECORD_SIZE = SPEC_2_12.rawrecord_size
TSTAT_SIZE = SPEC_2_12.tstat_size

# Offsets exposed for compatibility with existing 2.12 tests. The actual
# decoders consult the selected VersionSpec at runtime.
MEMSTAT_OFFSET = SPEC_2_12.memstat_offset
DSKSTAT_OFFSET = SPEC_2_12.dskstat_offset
INTFSTAT_OFFSET = SPEC_2_12.intfstat_offset


# ---------------------------------------------------------------------------
# Dataclasses returned to callers


@dataclass
class Header:
    magic: int
    aversion_raw: int
    aversion: str
    rawheadlen: int
    rawreclen: int
    hertz: int
    pidwidth: int
    sstatlen: int
    tstatlen: int
    pagesize: int
    supportflags: int
    osrel: int
    osvers: int
    ossub: int
    cstatlen: int
    sysname: str
    nodename: str
    release: str
    version: str
    machine: str
    domainname: str


@dataclass
class Process:
    """Subset of tstat fields we surface to the API layer."""

    pid: int
    tgid: int
    ppid: int
    name: str
    state: str
    cmdline: str
    nthr: int
    isproc: bool
    utime: int
    stime: int
    rmem_kb: int
    vmem_kb: int
    rio: int
    wio: int
    rsz: int
    wsz: int
    tcpsnd: int
    tcprcv: int
    udpsnd: int
    udprcv: int


@dataclass
class PerCpu:
    """Per CPU tick counters decoded from sstat.cpustat."""

    cpunr: int
    stime: int
    utime: int
    ntime: int
    itime: int
    wtime: int
    Itime: int
    Stime: int
    steal: int
    guest: int


@dataclass
class SystemCpu:
    """System wide CPU counters decoded from sstat.cpustat."""

    hertz: int
    nrcpu: int
    devint: int
    csw: int
    nprocs: int
    lavg1: float
    lavg5: float
    lavg15: float
    all: PerCpu
    cpus: list[PerCpu] = field(default_factory=list)


@dataclass
class PerInterface:
    """Per network interface counters decoded from sstat.intfstat."""

    name: str
    type: str  # 'e' (ether), 'w' (wireless), 'v' (virtual), '?' (unknown)
    speed_mbps: int
    duplex: int  # 0 = half, 1 = full, -1 = unknown
    rbyte: int
    rpack: int
    rerrs: int
    rdrop: int
    sbyte: int
    spack: int
    serrs: int
    sdrop: int


@dataclass
class SystemNetwork:
    """System wide network interface counters decoded from sstat.intfstat."""

    nrintf: int
    interfaces: list[PerInterface] = field(default_factory=list)


@dataclass
class DiskDevice:
    """Per block device counters decoded from sstat.dskstat.

    ``inflight`` is ``None`` on atop rawlogs that do not ship the field
    (atop 2.7 writes no inflight column). Callers must not substitute a
    zero: zero means "no outstanding I/O", not "the field is missing".
    """

    name: str
    nread: int
    nrsect: int
    nwrite: int
    nwsect: int
    io_ms: int
    avque: int
    ndisc: int
    ndsect: int
    inflight: Optional[int]
    kind: str  # one of "dsk", "mdd", "lvm"


@dataclass
class SystemDisk:
    """System wide disk counters decoded from sstat.dskstat."""

    disks: list[DiskDevice] = field(default_factory=list)
    mdds: list[DiskDevice] = field(default_factory=list)
    lvms: list[DiskDevice] = field(default_factory=list)


@dataclass
class SystemMemory:
    """System wide memory counters decoded from sstat.memstat.

    All page counters are raw values as stored by atop (number of OS pages,
    not bytes). ``availablemem`` is ``None`` on atop rawlogs that do not
    ship the field (atop 2.7 predates the kernel ``MemAvailable`` counter).
    Callers must not approximate a replacement value: a calculated number
    can easily be mistaken for the real counter. Display "N/A" instead.
    """

    pagesize: int
    physmem: int
    freemem: int
    buffermem: int
    slabmem: int
    cachemem: int
    totswap: int
    freeswap: int
    swapcached: int
    availablemem: Optional[int]


@dataclass
class Sample:
    curtime: int
    interval: int
    ndeviat: int
    nactproc: int
    ntask: int
    totproc: int
    totrun: int
    totslpi: int
    totslpu: int
    totzomb: int
    nrcpu: int | None = None
    system_memory: SystemMemory | None = None
    system_cpu: SystemCpu | None = None
    system_disk: SystemDisk | None = None
    system_network: SystemNetwork | None = None
    processes: list[Process] = field(default_factory=list)


@dataclass
class RawLog:
    header: Header
    samples: list[Sample]
    spec: VersionSpec | None = None
    # Phase 22: populated when ``parse_stream(..., lazy=True)`` is used.
    # Eager mode leaves it ``None`` so existing callers see no change.
    index: "SampleIndex | None" = None


@dataclass(slots=True)
class SampleOffset:
    """Byte layout of a single sample inside a rawlog file.

    Produced by ``scan_sample_offsets`` in one forward pass over the
    rawrecord headers. Carries the fields a lazy decoder needs to seek
    to, and bound the read of, any one sample without re-walking the
    entire file. ``sstatlen`` / ``tstatlen`` mirror the fixed per-version
    struct sizes so downstream code does not need to thread the spec
    alongside the offsets.
    """

    offset: int
    scomplen: int
    pcomplen: int
    curtime: int
    ndeviat: int
    sstatlen: int
    tstatlen: int


# ---------------------------------------------------------------------------
# Low level helpers


def _decode_cstring(raw: bytes) -> str:
    end = raw.find(b"\x00")
    if end >= 0:
        raw = raw[:end]
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return raw.decode("latin-1", errors="replace")


def _as_bytes(value) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, int):
        return bytes([value & 0xFF])
    if isinstance(value, (list, tuple)):
        return bytes(value)
    return bytes(str(value), "latin-1")


def _as_int(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, bytes):
        return value[0] if value else 0
    return int(value)


def _build_header(raw, spec: VersionSpec) -> Header:
    uts = raw.utsname
    version_bits = raw.aversion
    major = (version_bits >> 8) & 0x7F
    minor = version_bits & 0xFF
    # atop 2.7 rawheader has no pidwidth / cstatlen fields; expose zeros so
    # downstream code can rely on the shape without special casing.
    pidwidth = getattr(raw, "pidwidth", 0) or 0
    cstatlen = getattr(raw, "cstatlen", 0) or 0
    return Header(
        magic=raw.magic,
        aversion_raw=raw.aversion,
        aversion=f"{major}.{minor}",
        rawheadlen=raw.rawheadlen,
        rawreclen=raw.rawreclen,
        hertz=raw.hertz,
        pidwidth=pidwidth,
        sstatlen=raw.sstatlen,
        tstatlen=raw.tstatlen,
        pagesize=raw.pagesize,
        supportflags=raw.supportflags,
        osrel=raw.osrel,
        osvers=raw.osvers,
        ossub=raw.ossub,
        cstatlen=cstatlen,
        sysname=_decode_cstring(_as_bytes(uts.sysname)),
        nodename=_decode_cstring(_as_bytes(uts.nodename)),
        release=_decode_cstring(_as_bytes(uts.release)),
        version=_decode_cstring(_as_bytes(uts.version)),
        machine=_decode_cstring(_as_bytes(uts.machine)),
        domainname=_decode_cstring(_as_bytes(uts.domainname)),
    )


def _build_process(t) -> Process:
    gen = t.gen
    state_byte = _as_bytes(gen.state)
    return Process(
        pid=gen.pid,
        tgid=gen.tgid,
        ppid=gen.ppid,
        name=_decode_cstring(_as_bytes(gen.name)),
        state=_decode_cstring(state_byte) if state_byte and state_byte != b"\x00" else "",
        cmdline=_decode_cstring(_as_bytes(gen.cmdline)),
        nthr=gen.nthr,
        isproc=bool(_as_int(gen.isproc)),
        utime=t.cpu.utime,
        stime=t.cpu.stime,
        rmem_kb=t.mem.rmem,
        vmem_kb=t.mem.vmem,
        rio=t.dsk.rio,
        wio=t.dsk.wio,
        rsz=t.dsk.rsz,
        wsz=t.dsk.wsz,
        tcpsnd=t.net.tcpsnd,
        tcprcv=t.net.tcprcv,
        udpsnd=t.net.udpsnd,
        udprcv=t.net.udprcv,
    )


# ---------------------------------------------------------------------------
# sstat decoders (version aware)


def _read_percpu(sstat_bytes: bytes, offset: int, spec: VersionSpec) -> PerCpu | None:
    if offset + spec.percpu_size > len(sstat_bytes):
        return None
    cpunr = struct.unpack_from("<i", sstat_bytes, offset)[0]
    stime = struct.unpack_from("<q", sstat_bytes, offset + 8)[0]
    utime = struct.unpack_from("<q", sstat_bytes, offset + 16)[0]
    ntime = struct.unpack_from("<q", sstat_bytes, offset + 24)[0]
    itime = struct.unpack_from("<q", sstat_bytes, offset + 32)[0]
    wtime = struct.unpack_from("<q", sstat_bytes, offset + 40)[0]
    Itime = struct.unpack_from("<q", sstat_bytes, offset + 48)[0]
    Stime = struct.unpack_from("<q", sstat_bytes, offset + 56)[0]
    steal = struct.unpack_from("<q", sstat_bytes, offset + 64)[0]
    guest = struct.unpack_from("<q", sstat_bytes, offset + 72)[0]
    ticks = (stime, utime, ntime, itime, wtime, Itime, Stime, steal, guest)
    if any(t < 0 for t in ticks):
        return None
    return PerCpu(
        cpunr=cpunr,
        stime=stime,
        utime=utime,
        ntime=ntime,
        itime=itime,
        wtime=wtime,
        Itime=Itime,
        Stime=Stime,
        steal=steal,
        guest=guest,
    )


def _decode_system_cpu(
    sstat_bytes: bytes,
    hertz: int,
    sstatlen_claim: int,
    spec: VersionSpec | None = None,
) -> SystemCpu | None:
    """Decode cpustat from the decompressed sstat blob.

    Returns ``None`` when the sstat layout does not match a known atop
    revision, so callers can fall back gracefully. When ``spec`` is
    provided the caller has already chosen the layout; otherwise we look
    it up via ``sstatlen_claim`` and the callers' ``tstatlen`` assumption
    (not available here, so we fall back to the 2.12 spec for legacy
    tests whose claim matches 2.12).
    """
    if hertz <= 0:
        return None
    if spec is None:
        spec = _spec_for_sstatlen(sstatlen_claim)
        if spec is None:
            return None
    if sstatlen_claim != spec.sstat_size:
        return None
    if len(sstat_bytes) < spec.percpu_array_offset + spec.percpu_size:
        return None

    try:
        nrcpu = struct.unpack_from("<q", sstat_bytes, 0)[0]
        devint = struct.unpack_from("<q", sstat_bytes, 8)[0]
        csw = struct.unpack_from("<q", sstat_bytes, 16)[0]
        nprocs = struct.unpack_from("<q", sstat_bytes, 24)[0]
        lavg1 = struct.unpack_from("<f", sstat_bytes, 32)[0]
        lavg5 = struct.unpack_from("<f", sstat_bytes, 36)[0]
        lavg15 = struct.unpack_from("<f", sstat_bytes, 40)[0]
    except struct.error:
        return None

    if nrcpu < 1 or nrcpu > spec.cpustat_max_cpu:
        return None
    if devint < 0 or csw < 0 or nprocs < 0:
        return None

    max_fit = (len(sstat_bytes) - spec.percpu_array_offset) // spec.percpu_size
    if max_fit < nrcpu:
        return None

    all_cpu = _read_percpu(sstat_bytes, spec.percpu_all_offset, spec)
    if all_cpu is None:
        return None

    cpus: list[PerCpu] = []
    for i in range(nrcpu):
        off = spec.percpu_array_offset + spec.percpu_size * i
        p = _read_percpu(sstat_bytes, off, spec)
        if p is None:
            return None
        cpus.append(p)

    return SystemCpu(
        hertz=hertz,
        nrcpu=nrcpu,
        devint=devint,
        csw=csw,
        nprocs=nprocs,
        lavg1=lavg1,
        lavg5=lavg5,
        lavg15=lavg15,
        all=all_cpu,
        cpus=cpus,
    )


def _read_perdsk_array(
    sstat_bytes: bytes,
    base_off: int,
    count: int,
    max_count: int,
    kind: str,
    spec: VersionSpec,
) -> list[DiskDevice] | None:
    if count < 0 or count > max_count:
        return None
    end = base_off + spec.perdsk_size * count
    if end > len(sstat_bytes):
        return None

    devices: list[DiskDevice] = []
    for i in range(count):
        off = base_off + spec.perdsk_size * i
        raw_name = sstat_bytes[off : off + spec.perdsk_name_size]
        end_idx = raw_name.find(b"\x00")
        if end_idx >= 0:
            raw_name = raw_name[:end_idx]
        if not raw_name:
            break
        try:
            name = raw_name.decode("utf-8", errors="replace")
        except Exception:
            name = raw_name.decode("latin-1", errors="replace")

        name_end = spec.perdsk_name_size
        try:
            nread = struct.unpack_from("<q", sstat_bytes, off + name_end + 0)[0]
            nrsect = struct.unpack_from("<q", sstat_bytes, off + name_end + 8)[0]
            nwrite = struct.unpack_from("<q", sstat_bytes, off + name_end + 16)[0]
            nwsect = struct.unpack_from("<q", sstat_bytes, off + name_end + 24)[0]
            io_ms = struct.unpack_from("<q", sstat_bytes, off + name_end + 32)[0]
            avque = struct.unpack_from("<q", sstat_bytes, off + name_end + 40)[0]
            ndisc = struct.unpack_from("<q", sstat_bytes, off + name_end + 48)[0]
            ndsect = struct.unpack_from("<q", sstat_bytes, off + name_end + 56)[0]
            if spec.perdsk_has_inflight:
                inflight = struct.unpack_from(
                    "<q", sstat_bytes, off + name_end + 64
                )[0]
            else:
                inflight = None
        except struct.error:
            return None

        if nread < 0 or nrsect < 0 or nwrite < 0 or nwsect < 0 or io_ms < 0:
            return None

        devices.append(
            DiskDevice(
                name=name,
                nread=nread,
                nrsect=nrsect,
                nwrite=nwrite,
                nwsect=nwsect,
                io_ms=io_ms,
                avque=avque,
                ndisc=ndisc,
                ndsect=ndsect,
                inflight=inflight,
                kind=kind,
            )
        )
    return devices


def _decode_system_disk(
    sstat_bytes: bytes,
    sstatlen_claim: int,
    spec: VersionSpec | None = None,
) -> SystemDisk | None:
    """Decode dskstat from the decompressed sstat blob."""
    if spec is None:
        spec = _spec_for_sstatlen(sstatlen_claim)
        if spec is None:
            return None
    if sstatlen_claim != spec.sstat_size:
        return None
    if len(sstat_bytes) < spec.dskstat_offset + spec.dskstat_size:
        return None

    try:
        ndsk = struct.unpack_from("<i", sstat_bytes, spec.dskstat_offset + 0)[0]
        nmdd = struct.unpack_from("<i", sstat_bytes, spec.dskstat_offset + 4)[0]
        nlvm = struct.unpack_from("<i", sstat_bytes, spec.dskstat_offset + 8)[0]
    except struct.error:
        return None

    if ndsk < 0 or ndsk > spec.max_dsk:
        return None
    if nmdd < 0 or nmdd > spec.max_mdd:
        return None
    if nlvm < 0 or nlvm > spec.max_lvm:
        return None

    disks = _read_perdsk_array(
        sstat_bytes,
        spec.dskstat_offset + spec.dsk_array_offset,
        ndsk,
        spec.max_dsk,
        "dsk",
        spec,
    )
    if disks is None:
        return None
    mdds = _read_perdsk_array(
        sstat_bytes,
        spec.dskstat_offset + spec.mdd_array_offset,
        nmdd,
        spec.max_mdd,
        "mdd",
        spec,
    )
    if mdds is None:
        return None
    lvms = _read_perdsk_array(
        sstat_bytes,
        spec.dskstat_offset + spec.lvm_array_offset,
        nlvm,
        spec.max_lvm,
        "lvm",
        spec,
    )
    if lvms is None:
        return None

    return SystemDisk(disks=disks, mdds=mdds, lvms=lvms)


def _decode_system_network(
    sstat_bytes: bytes,
    sstatlen_claim: int,
    spec: VersionSpec | None = None,
) -> SystemNetwork | None:
    """Decode intfstat from the decompressed sstat blob."""
    if spec is None:
        spec = _spec_for_sstatlen(sstatlen_claim)
        if spec is None:
            return None
    if sstatlen_claim != spec.sstat_size:
        return None
    if len(sstat_bytes) < spec.intfstat_offset + spec.intfstat_size:
        return None

    try:
        nrintf = struct.unpack_from("<i", sstat_bytes, spec.intfstat_offset)[0]
    except struct.error:
        return None

    if nrintf < 1 or nrintf > spec.perintf_max:
        return None

    interfaces: list[PerInterface] = []
    for i in range(nrintf):
        base = spec.intfstat_offset + spec.perintf_array_offset + spec.perintf_size * i
        raw_name = sstat_bytes[base : base + spec.perintf_name_size]
        name_end = raw_name.find(b"\x00")
        if name_end >= 0:
            raw_name = raw_name[:name_end]
        if not raw_name:
            return None
        try:
            name = raw_name.decode("utf-8", errors="replace")
        except Exception:
            name = raw_name.decode("latin-1", errors="replace")

        try:
            rbyte = struct.unpack_from("<q", sstat_bytes, base + 16)[0]
            rpack = struct.unpack_from("<q", sstat_bytes, base + 24)[0]
            rerrs = struct.unpack_from("<q", sstat_bytes, base + 32)[0]
            rdrop = struct.unpack_from("<q", sstat_bytes, base + 40)[0]
            sbyte = struct.unpack_from("<q", sstat_bytes, base + 112)[0]
            spack = struct.unpack_from("<q", sstat_bytes, base + 120)[0]
            serrs = struct.unpack_from("<q", sstat_bytes, base + 128)[0]
            sdrop = struct.unpack_from("<q", sstat_bytes, base + 136)[0]
            type_byte = sstat_bytes[base + 208 : base + 209]
            speed = struct.unpack_from("<q", sstat_bytes, base + 216)[0]
            duplex_byte = sstat_bytes[base + 232 : base + 233]
        except struct.error:
            return None

        if rbyte < 0 or rpack < 0 or sbyte < 0 or spack < 0:
            return None

        iface_type = "?"
        if type_byte:
            ch = chr(type_byte[0]) if type_byte[0] else "?"
            if ch in ("e", "w", "v", "?"):
                iface_type = ch
            elif type_byte[0] == 0:
                iface_type = "?"
            else:
                iface_type = ch

        duplex = -1
        if duplex_byte:
            duplex = duplex_byte[0]
            if duplex not in (0, 1):
                duplex = -1

        interfaces.append(
            PerInterface(
                name=name,
                type=iface_type,
                speed_mbps=max(speed, 0),
                duplex=duplex,
                rbyte=rbyte,
                rpack=rpack,
                rerrs=rerrs,
                rdrop=rdrop,
                sbyte=sbyte,
                spack=spack,
                serrs=serrs,
                sdrop=sdrop,
            )
        )

    return SystemNetwork(nrintf=nrintf, interfaces=interfaces)


def _decode_system_memory(
    sstat_bytes: bytes,
    pagesize: int,
    sstatlen_claim: int,
    spec: VersionSpec | None = None,
) -> SystemMemory | None:
    """Decode memstat fields from the decompressed sstat blob.

    Returns ``None`` when the sstat layout does not match a known atop
    revision. ``availablemem`` is ``None`` when the selected revision does
    not carry the field (atop 2.7).
    """
    if pagesize <= 0:
        return None
    if spec is None:
        spec = _spec_for_sstatlen(sstatlen_claim)
        if spec is None:
            return None
    if sstatlen_claim != spec.sstat_size:
        return None
    if len(sstat_bytes) < spec.memstat_min_total:
        return None

    def _read(idx: int) -> int:
        return struct.unpack_from(
            "<q", sstat_bytes, spec.memstat_offset + idx * 8
        )[0]

    try:
        physmem = _read(spec.memstat_physmem_idx)
    except struct.error:
        return None

    if physmem <= 0 or physmem > (1 << 40):
        return None

    if spec.memstat_availablemem_idx is not None:
        available = _read(spec.memstat_availablemem_idx)
    else:
        available = None

    return SystemMemory(
        pagesize=pagesize,
        physmem=physmem,
        freemem=_read(spec.memstat_freemem_idx),
        buffermem=_read(spec.memstat_buffermem_idx),
        slabmem=_read(spec.memstat_slabmem_idx),
        cachemem=_read(spec.memstat_cachemem_idx),
        totswap=_read(spec.memstat_totswap_idx),
        freeswap=_read(spec.memstat_freeswap_idx),
        swapcached=_read(spec.memstat_swapcached_idx),
        availablemem=available,
    )


def _spec_for_sstatlen(sstatlen: int) -> VersionSpec | None:
    """Look up a spec by sstat length alone.

    Used by the test oriented decoder entry points that are called without
    a spec argument. The ``(tstatlen, sstatlen)`` tuple is the normal key;
    here we only have sstatlen, so we scan the version table and take the
    unique match. If two revisions ever share the same sstat size we will
    need to pass the spec explicitly.
    """
    matches = [s for s in VERSION_TABLE.values() if s.sstat_size == sstatlen]
    if len(matches) == 1:
        return matches[0]
    return None


# ---------------------------------------------------------------------------
# Stream level parsing


def _read_exact(stream: io.BufferedReader, size: int, what: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise RawLogError(
            f"unexpected EOF while reading {what}: wanted {size} bytes, got {len(data)}"
        )
    return data


# Offsets inside rawheader that are stable across atop 2.7 and 2.12:
#   0    uint32  magic
#   4    uint16  aversion
#   10   uint16  rawheadlen
#   12   uint16  rawreclen
#   14   uint16  hertz
#   16   uint16  [pidwidth on 2.12] / [sfuture[0] on 2.7]
#   28   uint32  sstatlen
#   32   uint32  tstatlen
# This is what makes an early version probe possible before we commit to a
# CDEF: the 12 byte block at offsets 16..27 differs (pidwidth vs sfuture),
# but sstatlen and tstatlen land at the same absolute positions.
_PROBE_HEADER_SIZE = 36


def _probe_header(stream: io.BufferedReader) -> tuple[int, int, int, int]:
    """Peek the first 36 bytes and return (magic, rawheadlen, tstatlen, sstatlen)."""
    stream.seek(0)
    buf = _read_exact(stream, _PROBE_HEADER_SIZE, "rawheader probe")
    magic = struct.unpack_from("<I", buf, 0)[0]
    rawheadlen = struct.unpack_from("<H", buf, 10)[0]
    sstatlen = struct.unpack_from("<I", buf, 28)[0]
    tstatlen = struct.unpack_from("<I", buf, 32)[0]
    return magic, rawheadlen, tstatlen, sstatlen


def _parse_header(stream: io.BufferedReader) -> tuple[Header, VersionSpec]:
    magic, rawheadlen, tstatlen, sstatlen = _probe_header(stream)
    if magic != MAGIC:
        raise RawLogError(
            f"invalid rawlog magic: expected 0x{MAGIC:08X}, got 0x{magic:08X}"
        )
    spec = _select_spec(tstatlen, sstatlen)

    if rawheadlen != spec.rawheader_size:
        raise RawLogError(
            f"rawheader length mismatch: file claims {rawheadlen}, "
            f"{spec.name} CDEF is {spec.rawheader_size}"
        )

    stream.seek(0)
    raw = spec.cs.rawheader(stream)
    if raw.rawreclen != spec.rawrecord_size:
        raise RawLogError(
            f"rawrecord length mismatch: file claims {raw.rawreclen}, "
            f"{spec.name} CDEF is {spec.rawrecord_size}"
        )
    if raw.tstatlen != spec.tstat_size:
        raise RawLogError(
            f"tstat length mismatch: file claims {raw.tstatlen}, "
            f"{spec.name} CDEF is {spec.tstat_size}"
        )
    return _build_header(raw, spec), spec


def scan_sample_offsets(
    stream: io.BufferedReader,
    max_samples: int | None,
    spec: VersionSpec,
) -> list[SampleOffset]:
    """Walk rawrecord headers, record offsets, skip past payloads.

    This is the public entry point used by both the eager decode path and
    the Phase 22 lazy index builder. It reads only the fixed-size record
    header of each sample (no zlib inflate), which keeps the first pass
    bounded by ``n_samples * rawrecord_size`` bytes regardless of the
    total rawlog size.

    The stream must be positioned immediately after the rawheader — use
    ``_parse_header`` first. On return, the stream is positioned at the
    first byte past the last sample it read (or EOF when ``max_samples``
    is None and the file runs out).
    """
    offsets: list[SampleOffset] = []
    sstatlen = spec.sstat_size
    tstatlen = spec.tstat_size
    while True:
        if max_samples is not None and len(offsets) >= max_samples:
            break
        pos = stream.tell()
        head = stream.read(spec.rawrecord_size)
        if not head:
            break
        if len(head) != spec.rawrecord_size:
            raise RawLogError(
                f"truncated rawrecord at sample {len(offsets)}: got {len(head)} bytes"
            )
        rec = spec.cs.rawrecord(head)
        skip = rec.scomplen + rec.pcomplen
        if spec.record_has_cgroup_fields:
            skip += rec.ccomplen + rec.icomplen
        stream.seek(skip, io.SEEK_CUR)
        offsets.append(
            SampleOffset(
                offset=pos,
                scomplen=rec.scomplen,
                pcomplen=rec.pcomplen,
                curtime=rec.curtime,
                ndeviat=rec.ndeviat,
                sstatlen=sstatlen,
                tstatlen=tstatlen,
            )
        )
    return offsets


def _scan_sample_offsets(
    stream: io.BufferedReader, max_samples: int | None, spec: VersionSpec
) -> list[SampleOffset]:
    """Backwards-compatible shim for the pre-Phase-22 private spelling.

    Kept so in-flight callers keep working while T-02/T-03 migrate to the
    public name. Removed in T-11 once nothing imports this.
    """
    return scan_sample_offsets(stream, max_samples, spec)


ProgressCallback = "typing.Callable[[str, int, int | None, int | None], None]"


@dataclass(slots=True)
class _DecodedSystemBundle:
    """What one inflated sstat blob yields after decoding.

    The lazy view caches this per sample so a single ``__getattr__`` for
    ``system_cpu`` also primes ``system_memory`` / ``system_disk`` /
    ``system_network`` from the same inflate — sstat inflate is the
    dominant per-sample cost and we only want to pay it once.
    """

    nrcpu: int | None
    system_memory: SystemMemory | None
    system_cpu: SystemCpu | None
    system_disk: SystemDisk | None
    system_network: SystemNetwork | None


def _decode_sstat_bundle(
    sstat_bytes: bytes,
    spec: VersionSpec,
    pagesize: int,
    sstatlen: int,
    hertz: int,
) -> _DecodedSystemBundle:
    """Decode all four system_* sub-dataclasses from one inflated sstat blob."""
    nrcpu: int | None = None
    if len(sstat_bytes) >= 8:
        candidate = int.from_bytes(sstat_bytes[:8], byteorder="little", signed=True)
        if 1 <= candidate <= 8192:
            nrcpu = candidate
    return _DecodedSystemBundle(
        nrcpu=nrcpu,
        system_memory=_decode_system_memory(sstat_bytes, pagesize, sstatlen, spec),
        system_cpu=_decode_system_cpu(sstat_bytes, hertz, sstatlen, spec),
        system_disk=_decode_system_disk(sstat_bytes, sstatlen, spec),
        system_network=_decode_system_network(sstat_bytes, sstatlen, spec),
    )


def _decode_processes(
    pdata: bytes,
    spec: VersionSpec,
    ndeviat: int,
) -> list[Process]:
    tstatlen = spec.tstat_size
    expected = tstatlen * ndeviat
    if len(pdata) != expected:
        raise RawLogError(
            f"tstat payload size mismatch: got {len(pdata)} bytes, expected {expected}"
        )
    processes: list[Process] = []
    for j in range(ndeviat):
        chunk = pdata[j * tstatlen : (j + 1) * tstatlen]
        t = spec.cs.tstat(chunk)
        processes.append(_build_process(t))
    return processes


def _decode_samples(
    stream: io.BufferedReader,
    spec: VersionSpec,
    pagesize: int,
    sstatlen: int,
    hertz: int,
    offsets: list[SampleOffset],
    progress_cb,
) -> list[Sample]:
    samples: list[Sample] = []
    total = len(offsets)

    PROGRESS_LO, PROGRESS_HI = 15, 85
    SPAN = PROGRESS_HI - PROGRESS_LO

    def _progress_at(fraction: float) -> int:
        fraction = max(0.0, min(1.0, fraction))
        return int(PROGRESS_LO + SPAN * fraction)

    for i, so in enumerate(offsets):
        stream.seek(so.offset, io.SEEK_SET)
        head = stream.read(spec.rawrecord_size)
        if len(head) != spec.rawrecord_size:
            raise RawLogError(
                f"truncated rawrecord at sample {i}: got {len(head)} bytes"
            )
        rec = spec.cs.rawrecord(head)

        scomp = _read_exact(stream, rec.scomplen, f"sstat blob of sample {i}")
        try:
            sstat_bytes = inflate(scomp)
        except DecompressError as exc:
            raise RawLogError(f"sstat inflate failed at sample {i}: {exc}") from exc

        bundle = _decode_sstat_bundle(sstat_bytes, spec, pagesize, sstatlen, hertz)

        if progress_cb is not None and total > 0:
            progress_cb(
                "decoding_sstat",
                i + 1,
                total,
                _progress_at((i + 0.5) / total),
            )

        processes: list[Process] = []
        if rec.pcomplen > 0:
            pcomp = _read_exact(stream, rec.pcomplen, f"tstat blob of sample {i}")
            try:
                pdata = inflate(pcomp)
            except DecompressError as exc:
                raise RawLogError(
                    f"tstat inflate failed at sample {i}: {exc}"
                ) from exc
            processes = _decode_processes(pdata, spec, rec.ndeviat)

        if progress_cb is not None and total > 0:
            progress_cb(
                "decoding_tstat",
                i + 1,
                total,
                _progress_at((i + 1) / total),
            )

        samples.append(
            Sample(
                curtime=rec.curtime,
                interval=rec.interval,
                ndeviat=rec.ndeviat,
                nactproc=rec.nactproc,
                ntask=rec.ntask,
                totproc=rec.totproc,
                totrun=rec.totrun,
                totslpi=rec.totslpi,
                totslpu=rec.totslpu,
                totzomb=rec.totzomb,
                nrcpu=bundle.nrcpu,
                system_memory=bundle.system_memory,
                system_cpu=bundle.system_cpu,
                system_disk=bundle.system_disk,
                system_network=bundle.system_network,
                processes=processes,
            )
        )
    return samples


def parse_stream(
    stream: io.BufferedReader,
    *,
    max_samples: int | None = None,
    progress_cb=None,
    lazy: bool | None = None,
) -> RawLog:
    """Parse a rawlog from an already open binary stream.

    ``lazy`` controls which decode path runs. When left as ``None`` the
    value comes from ``ATOP_LAZY`` (default ``"1"``/lazy). Flipping the
    env to ``"0"`` restores the pre-Phase-22 eager decode as a rollback
    escape hatch.

    Lazy mode returns a ``RawLog`` shell carrying only the header, the
    version spec and a ``SampleIndex``. The ``samples`` list is empty in
    that shell; callers that want per-sample access wrap the result
    with ``LazyRawLog``. The higher-level entry points
    (``parse_file`` / ``parse_bytes``) handle that wrapping for you.
    """
    if lazy is None:
        lazy = _lazy_default()
    if progress_cb is not None:
        progress_cb("header", 0, None, 10)

    header, spec = _parse_header(stream)
    offsets = scan_sample_offsets(stream, max_samples, spec)

    if progress_cb is not None:
        progress_cb("scanning", len(offsets), len(offsets), 15)

    if lazy:
        from atop_web.parser.index import build_sample_index

        index = build_sample_index(offsets, spec)
        if progress_cb is not None:
            progress_cb("index_built", len(index), len(index), 85)
        return RawLog(header=header, samples=[], spec=spec, index=index)

    samples = _decode_samples(
        stream,
        spec,
        header.pagesize,
        header.sstatlen,
        header.hertz,
        offsets,
        progress_cb,
    )

    if progress_cb is not None:
        progress_cb("building_samples", len(samples), len(samples), 85)

    return RawLog(header=header, samples=samples, spec=spec)


def parse_bytes(
    data: bytes,
    *,
    max_samples: int | None = None,
    progress_cb=None,
    lazy: bool | None = None,
):
    """Parse a rawlog given its bytes; lazy-aware by default.

    When the effective ``lazy`` is truthy we spool the bytes to a temp
    file and open a ``LazyRawLog`` against it — the caller then gets a
    proper file-backed lazy rawlog rather than a shell with no handle.
    """
    if lazy is None:
        lazy = _lazy_default()
    if lazy:
        import tempfile

        from atop_web.parser.lazy import LazyRawLog

        tmpdir = os.environ.get("TMPDIR") or tempfile.gettempdir()
        tf = tempfile.NamedTemporaryFile(
            mode="w+b", dir=tmpdir, prefix="atop_parse_", delete=False
        )
        try:
            tf.write(data)
            tf.flush()
        finally:
            tf.close()
        return LazyRawLog.open(Path(tf.name))
    return parse_stream(
        io.BytesIO(data),
        max_samples=max_samples,
        progress_cb=progress_cb,
        lazy=False,
    )


def parse_file(
    path: str | Path,
    *,
    max_samples: int | None = None,
    progress_cb=None,
    lazy: bool | None = None,
):
    """Parse a rawlog given its path; lazy-aware by default.

    Lazy mode returns a ``LazyRawLog`` that keeps the file handle
    open for the life of the session. Eager mode still decodes the
    full capture into ``list[Sample]`` and returns the legacy
    ``RawLog`` dataclass — kept for tests and the ``ATOP_LAZY=0``
    rollback path.
    """
    path = Path(path)
    if lazy is None:
        lazy = _lazy_default()
    if lazy and max_samples is None:
        from atop_web.parser.lazy import LazyRawLog

        return LazyRawLog.open(path)
    # Eager: ``max_samples`` still needs the full decode loop to honor
    # the cap, so the stream path is what we want here.
    with path.open("rb") as fh:
        return parse_stream(
            fh, max_samples=max_samples, progress_cb=progress_cb, lazy=False
        )
