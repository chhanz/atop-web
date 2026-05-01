"""Unit tests for the public offset scanner (Phase 22 T-01).

The scanner walks rawrecord headers without inflating any payload, and
returns a list of ``SampleOffset`` records that downstream code (Phase 22
lazy index) will use to seek into the file on demand. The tests here lock
down the invariants that the lazy path depends on:

* The scanner exposes a stable public name (``scan_sample_offsets``).
* Every emitted ``SampleOffset`` carries byte offset, scomp/pcomp sizes,
  sample metadata (curtime / ndeviat / sstatlen / tstatlen) and matches
  the eager ``parse_file`` decode 1:1.
* The legacy private name (``_scan_sample_offsets``) is kept as a shim so
  in-flight callers do not break while T-02/T-03 migrate over.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from atop_web.parser import parse_file
from atop_web.parser.reader import (
    SampleOffset,
    _parse_header,
    _scan_sample_offsets,
    scan_sample_offsets,
)


def _open_and_skip_header(path: Path):
    fh = path.open("rb")
    _, spec = _parse_header(fh)
    return fh, spec


def test_public_name_is_exported():
    # The public name is what T-02 imports. If this import ever breaks we
    # want to see a crisp test failure, not a downstream NameError.
    from atop_web.parser import reader

    assert callable(reader.scan_sample_offsets)


def test_legacy_private_shim_still_callable(rawlog_path: Path):
    # Backwards compatibility: keep the underscore spelling working until
    # all callers migrate (T-11 removes the shim).
    fh, spec = _open_and_skip_header(rawlog_path)
    try:
        result = _scan_sample_offsets(fh, None, spec)
    finally:
        fh.close()
    assert result, "expected at least one sample offset"
    # Each entry is a SampleOffset (not a bare tuple). The shim adapts.
    assert all(isinstance(o, SampleOffset) for o in result)


def test_scan_matches_eager_curtime_and_ndeviat(rawlog_path: Path):
    # Ground truth from the eager parser — curtime and ndeviat must match
    # one-to-one with what the public scanner returns.
    eager = parse_file(rawlog_path)
    fh, spec = _open_and_skip_header(rawlog_path)
    try:
        offsets = scan_sample_offsets(fh, None, spec)
    finally:
        fh.close()

    assert len(offsets) == len(eager.samples)
    for i, (so, sam) in enumerate(zip(offsets, eager.samples)):
        assert so.curtime == sam.curtime, f"curtime mismatch at sample {i}"
        assert so.ndeviat == sam.ndeviat, f"ndeviat mismatch at sample {i}"


def test_scan_offsets_seek_to_rawrecord_start(rawlog_path: Path):
    # The offset must point at the rawrecord header. Re-reading
    # rawrecord_size bytes from that offset must yield a struct whose
    # curtime / ndeviat equal the ones reported by the scanner. This is
    # the contract that lazy decode (T-03) relies on.
    fh, spec = _open_and_skip_header(rawlog_path)
    try:
        offsets = scan_sample_offsets(fh, 5, spec)
        for so in offsets:
            fh.seek(so.offset, io.SEEK_SET)
            head = fh.read(spec.rawrecord_size)
            rec = spec.cs.rawrecord(head)
            assert rec.curtime == so.curtime
            assert rec.ndeviat == so.ndeviat
            assert rec.scomplen == so.scomplen
            assert rec.pcomplen == so.pcomplen
    finally:
        fh.close()


def test_scan_populates_sstat_and_tstat_sizes(rawlog_path: Path):
    # sstatlen and tstatlen come from the version spec (they are fixed per
    # atop revision) — the scanner surfaces them on each SampleOffset so
    # the lazy decoder does not have to reach back into the spec.
    fh, spec = _open_and_skip_header(rawlog_path)
    try:
        offsets = scan_sample_offsets(fh, 3, spec)
    finally:
        fh.close()
    for so in offsets:
        assert so.sstatlen == spec.sstat_size
        assert so.tstatlen == spec.tstat_size


def test_scan_respects_max_samples(rawlog_path: Path):
    fh, spec = _open_and_skip_header(rawlog_path)
    try:
        offsets = scan_sample_offsets(fh, 2, spec)
    finally:
        fh.close()
    assert len(offsets) == 2


def test_sample_offset_dataclass_fields():
    # Lock down the field set and its order — the aggregate builder (T-04)
    # and the lazy view (T-03) unpack these positionally in hot paths.
    from dataclasses import fields

    names = [f.name for f in fields(SampleOffset)]
    # These must all be present; order is not part of the public contract
    # but the names are.
    for expected in ("offset", "scomplen", "pcomplen", "curtime", "ndeviat",
                     "sstatlen", "tstatlen"):
        assert expected in names, f"SampleOffset missing field: {expected}"


def test_scan_works_on_2_7_rawlog(rawlog_27_path: Path):
    # The 2.7 rawrecord has no cgroup fields — scanner must still produce
    # consistent curtime/ndeviat across versions.
    eager = parse_file(rawlog_27_path, max_samples=3)
    fh, spec = _open_and_skip_header(rawlog_27_path)
    try:
        offsets = scan_sample_offsets(fh, 3, spec)
    finally:
        fh.close()
    assert len(offsets) == len(eager.samples)
    for so, sam in zip(offsets, eager.samples):
        assert so.curtime == sam.curtime
        assert so.ndeviat == sam.ndeviat
