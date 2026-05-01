"""Dump the SampleIndex and rawrecord header bytes around a given sample.

Used to diagnose the Phase 22 post-mortem offset bug: around sample
~19800 the lazy decoder started seeing garbage ``ssize``/``psize``
values (3 GB, 1.8 GB, ...), which means either:

* the scanner is writing wrong offsets into the index, or
* the right offset lands in the middle of a payload instead of on the
  next rawrecord header.

Run:

    python scripts/inspect_offsets.py /home/ec2-user/Downloads/al2_atop_20260403
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

from atop_web.parser.lazy import LazyRawLog


def main(path_str: str, focus: int = 19800, span: int = 20) -> None:
    path = Path(path_str)
    size = path.stat().st_size
    print(f"file: {path}  size={size:,} B")

    lazy = LazyRawLog.open(path)
    try:
        n = len(lazy)
        idx = lazy.index
        spec = lazy.spec
        rawrec_size = spec.rawrecord_size
        print(
            f"samples: {n:,}  rawrecord_size={rawrec_size}  "
            f"spec={spec.name}"
        )

        # 1. Bulk scan: every offset must be < file size, delta between
        #    consecutive offsets must be non-negative and sane.
        bad_range = []
        bad_delta = []
        for i, off in enumerate(idx.offsets):
            if off < 0 or off >= size:
                bad_range.append((i, off))
            if i > 0:
                d = idx.offsets[i] - idx.offsets[i - 1]
                if d <= 0 or d > (10 << 20):  # 10 MiB per sample is huge
                    bad_delta.append((i, d, idx.offsets[i - 1], idx.offsets[i]))
        print(f"offsets out of file range: {len(bad_range)}")
        if bad_range:
            print("  first:", bad_range[:5])
        print(f"offsets with suspicious delta: {len(bad_delta)}")
        if bad_delta:
            print("  first:", bad_delta[:5])

        # 2. Focus window: print header fields + a hexdump around each
        #    sample in [focus-span, focus+span).
        lo = max(0, focus - span)
        hi = min(n, focus + span)
        print(f"\n--- focus window: samples {lo}..{hi - 1} ---")
        with path.open("rb") as fh:
            for i in range(lo, hi):
                off = idx.offsets[i]
                ct = idx.timestamps[i]
                sc = idx.scomplens[i]
                pc = idx.pcomplens[i]
                nd = idx.ndeviats[i]
                fh.seek(off)
                head = fh.read(rawrec_size)
                if len(head) != rawrec_size:
                    print(f"[{i:5d}] off={off:,}  SHORT READ ({len(head)} B)")
                    continue
                try:
                    rec = spec.cs.rawrecord(head)
                    rec_curtime = rec.curtime
                    rec_scomp = rec.scomplen
                    rec_pcomp = rec.pcomplen
                    rec_ndev = rec.ndeviat
                except Exception as exc:
                    rec_curtime = rec_scomp = rec_pcomp = rec_ndev = "?"
                    print(f"[{i:5d}] off={off:,} rec parse error: {exc}")
                    continue
                print(
                    f"[{i:5d}] off={off:,} idx(curtime={ct} scomp={sc} pcomp={pc} ndev={nd})  "
                    f"rec(curtime={rec_curtime} scomp={rec_scomp} pcomp={rec_pcomp} ndev={rec_ndev})"
                )
                mismatch = []
                if ct != rec_curtime:
                    mismatch.append(f"curtime idx={ct} rec={rec_curtime}")
                if sc != rec_scomp:
                    mismatch.append(f"scomp idx={sc} rec={rec_scomp}")
                if pc != rec_pcomp:
                    mismatch.append(f"pcomp idx={pc} rec={rec_pcomp}")
                if nd != rec_ndev:
                    mismatch.append(f"ndev idx={nd} rec={rec_ndev}")
                if mismatch:
                    print("    >>> MISMATCH:", "; ".join(mismatch))
                # Hexdump of the first 32 bytes of the rawrecord header.
                print("    hex[0:32]: " + head[:32].hex())
    finally:
        lazy.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/ec2-user/Downloads/al2_atop_20260403")
