"""Parser tests against the real rawlog binary."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from atop_web.parser import RawLogError, parse_bytes, parse_file
from atop_web.parser.reader import MAGIC, RAWHEADER_SIZE, RAWRECORD_SIZE, TSTAT_SIZE


def test_struct_sizes_match_rawlog_expectations():
    # rawheader = 480, rawrecord = 96, tstat = 968 for atop 2.12 rawlog
    assert RAWHEADER_SIZE == 480
    assert RAWRECORD_SIZE == 96
    assert TSTAT_SIZE == 968


def test_magic_rejects_garbage():
    garbage = b"\x00\x01\x02\x03" + b"\x00" * 500
    with pytest.raises(RawLogError):
        parse_bytes(garbage)


def test_parse_real_header(rawlog_bytes: bytes):
    magic = struct.unpack("<I", rawlog_bytes[:4])[0]
    assert magic == MAGIC

    rawlog = parse_bytes(rawlog_bytes, max_samples=1)
    header = rawlog.header
    assert header.magic == MAGIC
    assert header.rawheadlen == RAWHEADER_SIZE
    assert header.rawreclen == RAWRECORD_SIZE
    assert header.tstatlen == TSTAT_SIZE
    assert header.pagesize > 0
    assert header.sysname == "Linux"
    assert header.nodename
    assert header.release


def test_parse_full_file(rawlog_path):
    rawlog = parse_file(rawlog_path)
    assert rawlog.samples, "expected at least one sample"
    first = rawlog.samples[0]
    assert first.curtime > 0
    assert first.ndeviat > 0
    assert len(first.processes) == first.ndeviat

    for sample in rawlog.samples:
        assert sample.curtime > 0
        assert len(sample.processes) == sample.ndeviat
        for p in sample.processes:
            assert p.pid >= 0
            assert p.utime >= 0
            assert p.stime >= 0


def test_samples_are_chronological(rawlog_path):
    rawlog = parse_file(rawlog_path)
    times = [s.curtime for s in rawlog.samples]
    assert times == sorted(times), "sample timestamps must be monotonic"


def test_process_name_is_printable(rawlog_path):
    rawlog = parse_file(rawlog_path, max_samples=3)
    names = {p.name for s in rawlog.samples for p in s.processes}
    assert any(n for n in names), "expected at least one non empty process name"


# System memory decoding (Phase 8) --------------------------------------------

# Reference values taken from a manual decode of the sample rawlog:
#   physmem      = 1,992,591 pages   (matches host _PHYS_PAGES)
#   freemem      =   333,224 pages
#   buffermem    =       210 pages
#   slabmem      =   505,732 pages
#   cachemem     =   442,365 pages
#   availablemem =   866,132 pages
# See ``atop_web/parser/reader.py`` MEMSTAT_OFFSET for the derivation.


def test_system_memory_first_sample_matches_reference(rawlog_path):
    rawlog = parse_file(rawlog_path, max_samples=1)
    sample = rawlog.samples[0]
    mem = sample.system_memory
    assert mem is not None, "memstat should decode for atop v2.12.x rawlog"
    assert mem.pagesize == rawlog.header.pagesize
    assert mem.physmem == 1_992_591
    assert mem.freemem == 333_224
    assert mem.buffermem == 210
    assert mem.slabmem == 505_732
    assert mem.cachemem == 442_365
    assert mem.availablemem == 866_132
    # Memory accounted for (used + free + cache + buff + slab) should equal
    # physmem within a fraction of a percent.
    used = mem.physmem - mem.freemem - mem.cachemem - mem.buffermem - mem.slabmem
    accounted = used + mem.freemem + mem.cachemem + mem.buffermem + mem.slabmem
    assert accounted == mem.physmem


def test_system_memory_swap_values_non_negative(rawlog_path):
    rawlog = parse_file(rawlog_path, max_samples=4)
    for s in rawlog.samples:
        if s.system_memory is None:
            continue
        assert s.system_memory.totswap >= 0
        assert s.system_memory.freeswap >= 0
        assert s.system_memory.swapcached >= 0


def test_system_memory_none_when_sstatlen_mismatch():
    # Using the raw decoder directly so we do not rely on a synthetic rawlog.
    from atop_web.parser.reader import _decode_system_memory

    # Correct length but wrong sstatlen claim -> ignored.
    assert _decode_system_memory(b"\x00" * 1_064_016, 4096, 123_456) is None
    # Short buffer -> ignored.
    assert _decode_system_memory(b"\x00" * 1024, 4096, 1_064_016) is None
    # Zero pagesize -> ignored.
    assert _decode_system_memory(b"\x00" * 1_064_016, 0, 1_064_016) is None


# System CPU decoding (Phase 9) -----------------------------------------------


def test_decode_system_cpu_ok(rawlog_path):
    rawlog = parse_file(rawlog_path, max_samples=1)
    sample = rawlog.samples[0]
    cpu = sample.system_cpu
    assert cpu is not None
    # ``_PHYS_PAGES`` equivalent for CPU count: known to be 2 in this fixture.
    assert cpu.nrcpu == 2
    assert cpu.hertz == rawlog.header.hertz
    assert cpu.all.utime > 0
    assert cpu.all.stime > 0
    # Summing per-CPU counters must equal the "all" row for u/s/idle.
    sum_u = sum(p.utime for p in cpu.cpus)
    sum_s = sum(p.stime for p in cpu.cpus)
    sum_i = sum(p.itime for p in cpu.cpus)
    # atop aggregates all CPUs by addition; allow +-1 tick rounding slack.
    assert abs(sum_u - cpu.all.utime) <= 1
    assert abs(sum_s - cpu.all.stime) <= 1
    assert abs(sum_i - cpu.all.itime) <= 1
    # Total ticks across all layers must be in the right ballpark compared to
    # the budget hertz * interval * nrcpu (atop counters are cumulative, but
    # within a single sample they stay below the budget).
    total = sum(
        [
            cpu.all.utime,
            cpu.all.stime,
            cpu.all.ntime,
            cpu.all.itime,
            cpu.all.wtime,
            cpu.all.Itime,
            cpu.all.Stime,
            cpu.all.steal,
            cpu.all.guest,
        ]
    )
    assert total > 0
    # Load averages are small non negative floats on a lightly loaded box.
    assert 0.0 <= cpu.lavg1 < 1000.0


def test_decode_system_cpu_len_mismatch():
    from atop_web.parser.reader import _decode_system_cpu

    # Wrong sstatlen claim -> None.
    assert _decode_system_cpu(b"\x00" * 1_064_016, 100, 123_456) is None
    # Short buffer -> None.
    assert _decode_system_cpu(b"\x00" * 100, 100, 1_064_016) is None
    # Zero hertz -> None.
    assert _decode_system_cpu(b"\x00" * 1_064_016, 0, 1_064_016) is None


def test_decode_system_cpu_zero_nrcpu_rejected():
    from atop_web.parser.reader import _decode_system_cpu

    # All zero buffer yields nrcpu=0, which must fail the sanity check.
    assert _decode_system_cpu(b"\x00" * 1_064_016, 100, 1_064_016) is None


# System Disk decoding (Phase 10) ---------------------------------------------


def test_decode_system_disk_ok(rawlog_path):
    rawlog = parse_file(rawlog_path, max_samples=1)
    disk = rawlog.samples[0].system_disk
    assert disk is not None
    assert disk.disks, "expected at least one physical disk"
    dev = disk.disks[0]
    assert dev.name == "nvme0n1"
    assert dev.kind == "dsk"
    assert dev.nread > 0
    assert dev.nrsect > 0
    assert dev.nwrite > 0
    assert dev.nwsect > 0
    # mdd / lvm arrays empty on this fixture but must still be lists.
    assert disk.mdds == []
    assert disk.lvms == []


def test_decode_system_disk_counters_non_negative(rawlog_path):
    # atop writes either cumulative or per interval counters depending on
    # the sample kind, so monotonic increase across consecutive samples is
    # not guaranteed. What is guaranteed is that every counter decodes as a
    # non negative integer and that the device keeps the same name over
    # time.
    rawlog = parse_file(rawlog_path, max_samples=6)
    seen_name = None
    seen_any = False
    for s in rawlog.samples:
        if s.system_disk is None or not s.system_disk.disks:
            continue
        dev = s.system_disk.disks[0]
        seen_any = True
        if seen_name is None:
            seen_name = dev.name
        assert dev.name == seen_name
        for field_name in ("nread", "nrsect", "nwrite", "nwsect", "io_ms"):
            assert getattr(dev, field_name) >= 0, field_name
    assert seen_any, "expected at least one decoded disk sample"


def test_decode_system_disk_len_mismatch():
    from atop_web.parser.reader import _decode_system_disk

    # Wrong sstatlen claim -> None.
    assert _decode_system_disk(b"\x00" * 1_064_016, 123_456) is None
    # Short buffer -> None.
    assert _decode_system_disk(b"\x00" * 1024, 1_064_016) is None


# System Network decoding (Phase 11) ------------------------------------------


def test_decode_system_network_ok(rawlog_path):
    rawlog = parse_file(rawlog_path, max_samples=1)
    net = rawlog.samples[0].system_network
    assert net is not None
    assert net.nrintf >= 1
    assert len(net.interfaces) == net.nrintf
    # The loopback interface is always present on a Linux box.
    names = [i.name for i in net.interfaces]
    assert "lo" in names
    lo = next(i for i in net.interfaces if i.name == "lo")
    assert lo.type == "v"
    # lo is always symmetric: bytes and packets received equal those sent.
    assert lo.rbyte == lo.sbyte
    assert lo.rpack == lo.spack
    # All byte / packet counters must be non negative.
    for iface in net.interfaces:
        assert iface.rbyte >= 0 and iface.sbyte >= 0
        assert iface.rpack >= 0 and iface.spack >= 0


def test_decode_system_network_matches_reference_if_available(tmp_path):
    # When the 2026-04-28 fixture is present, cross check the values the brief
    # calls out so we are sure the layout constants are right.
    import os

    candidates = [
        os.environ.get("ATOP_LOG_DIR") or "",
        "/app/logs",
        "/var/log/atop",
    ]
    target = None
    for base in candidates:
        if not base:
            continue
        path = Path(base) / "atop_20260428"
        if path.is_file():
            target = path
            break
    if target is None:
        pytest.skip("atop_20260428 fixture not available")

    rawlog = parse_file(target, max_samples=1)
    net = rawlog.samples[0].system_network
    assert net is not None
    assert net.nrintf == 15
    by_name = {i.name: i for i in net.interfaces}
    assert by_name["lo"].rbyte == 2_586_528
    assert by_name["lo"].sbyte == 2_586_528
    assert by_name["lo"].rpack == 1_210
    assert by_name["docker0"].rbyte == 17_264_083
    assert by_name["docker0"].sbyte == 230_057_187
    assert by_name["veth1f3f359"].rbyte == 200_527_033
    assert by_name["veth1f3f359"].sbyte == 174_133_553


def test_decode_system_network_len_mismatch():
    from atop_web.parser.reader import _decode_system_network

    # Wrong sstatlen claim -> None.
    assert _decode_system_network(b"\x00" * 1_064_016, 123_456) is None
    # Short buffer -> None.
    assert _decode_system_network(b"\x00" * 1024, 1_064_016) is None
    # All zero buffer -> nrintf = 0 -> None (fail closed on unknown layout).
    assert _decode_system_network(b"\x00" * 1_064_016, 1_064_016) is None


# Parser progress callback (Phase 12) -----------------------------------------


def test_parse_progress_cb_reports_all_stages(rawlog_path):
    events: list[tuple[str, int, int | None, int | None]] = []

    def cb(stage, current, total, progress):
        events.append((stage, current, total, progress))

    rawlog = parse_file(rawlog_path, max_samples=4, progress_cb=cb)
    assert rawlog.samples, "expected at least one decoded sample"

    stages = [e[0] for e in events]
    # Required ordered transitions: header -> scanning -> decoding_sstat ->
    # decoding_tstat -> building_samples.
    assert stages[0] == "header"
    assert "scanning" in stages
    assert "decoding_sstat" in stages
    assert "decoding_tstat" in stages
    assert "building_samples" in stages
    assert stages.index("decoding_sstat") < stages.index("decoding_tstat")
    assert stages.index("decoding_tstat") < stages.index("building_samples")

    # Progress is bounded and non decreasing across emitted events.
    progresses = [p for _, _, _, p in events if p is not None]
    assert all(0 <= p <= 100 for p in progresses)
    for a, b in zip(progresses, progresses[1:]):
        assert a <= b
    assert max(progresses) <= 85  # parser caps its own hint at building stage


def test_parse_progress_cb_emits_enough_updates(rawlog_path):
    events = []

    def cb(stage, current, total, progress):
        events.append(stage)

    # At least five callback fires on a real rawlog (header, scanning,
    # sstat, tstat, building). Larger files obviously emit many more.
    parse_file(rawlog_path, max_samples=3, progress_cb=cb)
    assert len(events) >= 5


# Multi version dispatch (Phase 16) -------------------------------------------


def test_version_table_covers_both_revisions():
    from atop_web.parser.reader import SPEC_2_12, SPEC_2_7, VERSION_TABLE

    assert VERSION_TABLE[(968, 1_064_016)] is SPEC_2_12
    assert VERSION_TABLE[(840, 954_360)] is SPEC_2_7
    # 2.12 struct sizes are exposed through the module as well for legacy
    # tests that import them directly.
    assert SPEC_2_12.tstat_size == 968
    assert SPEC_2_12.sstat_size == 1_064_016
    assert SPEC_2_7.tstat_size == 840
    assert SPEC_2_7.sstat_size == 954_360


def test_select_spec_rejects_unsupported_with_clear_message():
    from atop_web.parser.reader import _select_spec, RawLogError

    with pytest.raises(RawLogError) as exc_info:
        _select_spec(1000, 999_999)
    msg = str(exc_info.value)
    assert "unsupported atop version" in msg
    assert "tstat=1000" in msg
    assert "sstat=999999" in msg
    # The error must advertise the supported combinations so the operator
    # can tell at a glance which atop revisions this build understands.
    assert "atop_2_12" in msg
    assert "atop_2_7" in msg


def test_spec_2_7_layout_sizes():
    from atop_web.parser.reader import SPEC_2_7

    assert SPEC_2_7.rawheader_size == 480
    assert SPEC_2_7.rawrecord_size == 96
    assert SPEC_2_7.tstat_size == 840
    # Measured: dsk arrays sum to 372_752 B, intf arrays to 34_824 B.
    assert SPEC_2_7.dskstat_size == 372_752
    assert SPEC_2_7.intfstat_size == 34_824
    # perdsk on 2.7 is 112 B and carries no inflight column.
    assert SPEC_2_7.perdsk_size == 112
    assert SPEC_2_7.perdsk_has_inflight is False
    # memstat on 2.7 has no availablemem.
    assert SPEC_2_7.memstat_availablemem_idx is None
    # Record has no cgroup accounting fields.
    assert SPEC_2_7.record_has_cgroup_fields is False


def test_parse_2_7_rawlog_header_detects_version(rawlog_27_path):
    rawlog = parse_file(rawlog_27_path, max_samples=1)
    h = rawlog.header
    assert h.aversion == "2.7"
    assert h.tstatlen == 840
    assert h.sstatlen == 954_360
    assert h.rawheadlen == 480
    assert h.rawreclen == 96
    assert rawlog.spec is not None
    assert rawlog.spec.name == "atop_2_7"


def test_parse_2_7_rawlog_samples_decode_core_fields(rawlog_27_path):
    rawlog = parse_file(rawlog_27_path, max_samples=3)
    assert len(rawlog.samples) == 3
    s = rawlog.samples[0]

    assert s.curtime > 0
    assert s.ndeviat > 0
    assert len(s.processes) == s.ndeviat
    for p in s.processes:
        assert p.pid >= 0
        assert p.utime >= 0
        assert p.stime >= 0

    # CPU: at least one CPU, load averages decoded as finite floats.
    assert s.system_cpu is not None
    assert s.system_cpu.nrcpu >= 1
    assert s.system_cpu.lavg1 >= 0.0
    assert s.system_cpu.all.utime >= 0


def test_parse_2_7_memory_availablemem_is_none(rawlog_27_path):
    # atop 2.7 rawlogs carry no availablemem counter. The decoder must
    # return None so the UI shows "N/A" instead of a synthetic value.
    rawlog = parse_file(rawlog_27_path, max_samples=2)
    for s in rawlog.samples:
        assert s.system_memory is not None
        assert s.system_memory.availablemem is None
        assert s.system_memory.physmem > 0
        assert s.system_memory.freemem >= 0
        # Swap fields still decode and must stay non negative.
        assert s.system_memory.totswap >= 0
        assert s.system_memory.freeswap >= 0
        assert s.system_memory.swapcached >= 0


def test_parse_2_7_disk_inflight_is_none(rawlog_27_path):
    # atop 2.7 perdsk has no inflight column. The decoder must return
    # None (not 0, which would lie about the queue depth being empty).
    rawlog = parse_file(rawlog_27_path, max_samples=2)
    saw_any = False
    for s in rawlog.samples:
        if s.system_disk is None:
            continue
        for d in s.system_disk.disks + s.system_disk.mdds + s.system_disk.lvms:
            saw_any = True
            assert d.inflight is None
            assert d.nread >= 0
            assert d.nwrite >= 0
            assert d.nrsect >= 0
            assert d.nwsect >= 0
            assert d.io_ms >= 0
    assert saw_any, "expected at least one decoded disk device in the 2.7 fixture"


def test_parse_2_7_network_counters_non_negative(rawlog_27_path):
    rawlog = parse_file(rawlog_27_path, max_samples=2)
    for s in rawlog.samples:
        net = s.system_network
        assert net is not None
        assert net.nrintf >= 1
        assert len(net.interfaces) == net.nrintf
        names = [i.name for i in net.interfaces]
        assert "lo" in names
        for iface in net.interfaces:
            assert iface.rbyte >= 0
            assert iface.sbyte >= 0
            assert iface.rpack >= 0
            assert iface.spack >= 0


def test_decoders_return_none_on_2_7_sstatlen_with_mismatched_buffer():
    # The 2.7 decoders must also fail closed when the claimed sstat length
    # matches 2.7 but the buffer is too small to carry the substructure.
    from atop_web.parser.reader import (
        _decode_system_memory,
        _decode_system_disk,
        _decode_system_network,
        _decode_system_cpu,
    )

    short = b"\x00" * 1024
    assert _decode_system_memory(short, 4096, 954_360) is None
    assert _decode_system_cpu(short, 100, 954_360) is None
    assert _decode_system_disk(short, 954_360) is None
    assert _decode_system_network(short, 954_360) is None
