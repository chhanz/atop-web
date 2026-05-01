"""On-demand sample decode backed by the offset index.

Phase 22 T-03 (+ post-mortem fix). The route layer and tool handlers
reach for fields like ``sample.curtime``, ``sample.system_cpu.all.utime``
and ``sample.processes[i].pid`` on samples that, in the new lazy path,
were never fully decoded at load time. ``SampleView`` gives them the
same attribute surface by pulling bytes out of an mmap on first access
and caching the decoded sub-dataclasses on the view instance.

Concurrency contract
--------------------
FastAPI routes run in a thread pool, and several requests may land on
the same session at the same time. The original implementation kept a
shared ``io.BufferedReader`` and used ``seek`` + ``read`` pairs, which
is a textbook race: thread A seeks to offset X, thread B seeks to
offset Y, thread A then reads ``scomplen`` bytes starting at Y, reads
garbage, and interprets the first four as a fresh ``scomplen`` field
that might claim "wanted 3 GB". We switched to ``mmap`` slicing, which
has no shared position, so concurrent decodes are safe without a lock
around the I/O path. The LRU dict is still mutated by multiple threads
so ``_lru_lock`` guards it.

Design notes
------------
* ``LazyRawLog`` owns the file handle and an mmap over the whole
  rawlog. It is a context manager; callers close it so the mmap and
  the underlying descriptor go away together.
* ``SampleView`` is lightweight: only the offset-row ints and lazy
  ``_record`` / ``_bundle`` / ``_processes_cache`` slots. The four
  ``system_*`` fields are all primed by a single sstat inflate.
* A small per-rawlog LRU caches recently-accessed ``SampleView``
  instances. Walking the index in chronological order (as the
  aggregate builder does) stays inside the cache and keeps inflate
  work constant.
"""

from __future__ import annotations

import io
import mmap
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

from atop_web.parser.decompress import DecompressError, inflate
from atop_web.parser.reader import (
    Process,
    RawLogError,
    SystemCpu,
    SystemDisk,
    SystemMemory,
    SystemNetwork,
    _decode_processes,
    _decode_sstat_bundle,
    parse_stream,
)

if TYPE_CHECKING:
    from atop_web.parser.index import SampleIndex
    from atop_web.parser.reader import Header, VersionSpec


# Upper bound on how many recently used SampleView instances we keep
# alive on a LazyRawLog. 16 is the Phase-22 headline number: large
# enough that aggregate builders walking the index sequentially do not
# pay the same inflate twice, small enough to stay in a few MB on the
# 266 MB fixture.
_DEFAULT_LRU_SIZE = 16


class SampleView:
    """Attribute-compatible view onto one sample in a lazy rawlog.

    Reads and decodes only the fields that are asked for, at the moment
    they are asked for. Cached on the instance so the second access on
    the same ``SampleView`` never re-inflates.
    """

    __slots__ = (
        "_rawlog",
        "_index_pos",
        "_offset",
        "_curtime",
        "_ndeviat",
        "_scomplen",
        "_pcomplen",
        "_record",
        "_bundle",
        "_processes_cache",
        "_decode_count",
    )

    def __init__(
        self,
        rawlog: "LazyRawLog",
        index_pos: int,
        offset: int,
        curtime: int,
        ndeviat: int,
        scomplen: int,
        pcomplen: int,
    ) -> None:
        self._rawlog = rawlog
        self._index_pos = index_pos
        self._offset = offset
        self._curtime = curtime
        self._ndeviat = ndeviat
        self._scomplen = scomplen
        self._pcomplen = pcomplen
        self._record = None  # cached rawrecord struct
        self._bundle = None  # cached _DecodedSystemBundle
        self._processes_cache: list[Process] | None = None
        self._decode_count = 0  # test instrumentation; counts inflate() calls

    # ----- cheap (index-only) accessors ------------------------------------

    @property
    def curtime(self) -> int:
        return self._curtime

    @property
    def ndeviat(self) -> int:
        return self._ndeviat

    # ----- lazy rawrecord header ------------------------------------------

    def _read_record(self):
        """Slice the rawrecord header out of mmap and cache it.

        mmap slicing returns a fresh ``bytes`` object with no shared
        position, so this is safe to call from several threads at once
        even on the same sample (we may decode twice, but the result
        is the same and both writers land the same value into
        ``self._record``).
        """
        if self._record is not None:
            return self._record
        spec = self._rawlog.spec
        mm = self._rawlog._mmap
        if mm is None:
            raise RawLogError(
                f"lazy rawlog is closed at sample {self._index_pos}"
            )
        start = self._offset
        end = start + spec.rawrecord_size
        if end > len(mm):
            raise RawLogError(
                f"truncated rawrecord at lazy sample {self._index_pos}: "
                f"got {max(0, len(mm) - start)} bytes"
            )
        head = mm[start:end]
        if len(head) != spec.rawrecord_size:
            raise RawLogError(
                f"truncated rawrecord at lazy sample {self._index_pos}: "
                f"got {len(head)} bytes"
            )
        rec = spec.cs.rawrecord(head)
        self._record = rec
        return rec

    # ----- rawrecord-derived scalars --------------------------------------
    # The eager Sample has these at the top level; we surface them through
    # the lazily-decoded rawrecord so the view is field-compatible.

    @property
    def interval(self) -> int:
        return self._read_record().interval

    @property
    def nactproc(self) -> int:
        return self._read_record().nactproc

    @property
    def ntask(self) -> int:
        return self._read_record().ntask

    @property
    def totproc(self) -> int:
        return self._read_record().totproc

    @property
    def totrun(self) -> int:
        return self._read_record().totrun

    @property
    def totslpi(self) -> int:
        return self._read_record().totslpi

    @property
    def totslpu(self) -> int:
        return self._read_record().totslpu

    @property
    def totzomb(self) -> int:
        return self._read_record().totzomb

    # ----- sstat-backed sub-dataclasses -----------------------------------

    def _read_bundle(self):
        if self._bundle is not None:
            return self._bundle
        rec = self._read_record()
        spec = self._rawlog.spec
        mm = self._rawlog._mmap
        if mm is None:
            raise RawLogError(
                f"lazy rawlog is closed at sample {self._index_pos}"
            )
        blob_start = self._offset + spec.rawrecord_size
        blob_end = blob_start + rec.scomplen
        if blob_end > len(mm):
            raise RawLogError(
                f"unexpected EOF while reading sstat blob at lazy sample "
                f"{self._index_pos}: wanted {rec.scomplen} bytes, "
                f"got {max(0, len(mm) - blob_start)}"
            )
        scomp = mm[blob_start:blob_end]
        try:
            sstat_bytes = inflate(scomp)
        except DecompressError as exc:
            raise RawLogError(
                f"sstat inflate failed at lazy sample {self._index_pos}: {exc}"
            ) from exc
        self._decode_count += 1
        bundle = _decode_sstat_bundle(
            sstat_bytes,
            spec,
            self._rawlog.header.pagesize,
            self._rawlog.header.sstatlen,
            self._rawlog.header.hertz,
        )
        self._bundle = bundle
        return bundle

    @property
    def nrcpu(self) -> Optional[int]:
        return self._read_bundle().nrcpu

    @property
    def system_memory(self) -> Optional[SystemMemory]:
        return self._read_bundle().system_memory

    @property
    def system_cpu(self) -> Optional[SystemCpu]:
        return self._read_bundle().system_cpu

    @property
    def system_disk(self) -> Optional[SystemDisk]:
        return self._read_bundle().system_disk

    @property
    def system_network(self) -> Optional[SystemNetwork]:
        return self._read_bundle().system_network

    # ----- processes list --------------------------------------------------

    @property
    def processes(self) -> list[Process]:
        if self._processes_cache is not None:
            return self._processes_cache
        rec = self._read_record()
        if rec.pcomplen <= 0 or self._ndeviat == 0:
            self._processes_cache = []
            return self._processes_cache
        spec = self._rawlog.spec
        mm = self._rawlog._mmap
        if mm is None:
            raise RawLogError(
                f"lazy rawlog is closed at sample {self._index_pos}"
            )
        # tstat blob lives after the rawrecord header and the sstat
        # payload. mmap slicing is position-free so this is independent
        # of any concurrent read in progress.
        blob_start = self._offset + spec.rawrecord_size + rec.scomplen
        blob_end = blob_start + rec.pcomplen
        if blob_end > len(mm):
            raise RawLogError(
                f"unexpected EOF while reading tstat blob at lazy sample "
                f"{self._index_pos}: wanted {rec.pcomplen} bytes, "
                f"got {max(0, len(mm) - blob_start)}"
            )
        pcomp = mm[blob_start:blob_end]
        try:
            pdata = inflate(pcomp)
        except DecompressError as exc:
            raise RawLogError(
                f"tstat inflate failed at lazy sample {self._index_pos}: {exc}"
            ) from exc
        self._decode_count += 1
        self._processes_cache = _decode_processes(pdata, spec, rec.ndeviat)
        return self._processes_cache


@dataclass
class LazyRawLog:
    """File-backed, index-driven view of a rawlog.

    Holds the file handle and an mmap open for the lifetime of the
    object; callers close it via ``close()`` or the context manager.
    mmap slicing gives every thread its own fresh ``bytes`` with no
    shared seek position, so route handlers can decode concurrently.
    """

    header: "Header"
    spec: "VersionSpec"
    index: "SampleIndex"
    _file: io.BufferedReader
    _mmap: "mmap.mmap | None" = None
    _source_path: Path | None = None
    _lru_size: int = _DEFAULT_LRU_SIZE

    def __post_init__(self) -> None:
        # OrderedDict-based LRU: recently used keys are at the end.
        self._lru: "OrderedDict[int, SampleView]" = OrderedDict()
        # Guards the LRU against concurrent route handlers. The I/O
        # path itself does not need a lock because mmap slicing is
        # atomic and position-free, but dict mutation is not.
        self._lru_lock = threading.Lock()

    # --- construction ------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path, *, lru_size: int = _DEFAULT_LRU_SIZE) -> "LazyRawLog":
        """Open ``path``, build the offset index, keep the mmap alive."""
        p = Path(path)
        fh = p.open("rb")
        try:
            rawlog = parse_stream(fh, lazy=True)
            # mmap over the entire file. ``ACCESS_READ`` gives the
            # kernel a read-only page cache mapping; slicing it yields
            # bytes with no shared position so multiple threads can
            # decode concurrently.
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            fh.close()
            raise
        assert rawlog.index is not None
        assert rawlog.spec is not None
        return cls(
            header=rawlog.header,
            spec=rawlog.spec,
            index=rawlog.index,
            _file=fh,
            _mmap=mm,
            _source_path=p,
            _lru_size=lru_size,
        )

    # --- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        with self._lru_lock:
            self._lru.clear()
        if self._mmap is not None:
            try:
                self._mmap.close()
            except (BufferError, ValueError):
                # BufferError: a SampleView still holds a memoryview
                # into the mapping. We already cleared the LRU so any
                # such view is on its way out; let GC finish the job.
                pass
            self._mmap = None
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "LazyRawLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --- sample access -----------------------------------------------------

    @property
    def samples(self) -> "LazyRawLog":
        """Eager compatibility: ``rawlog.samples`` works on lazy rawlogs too.

        Returns ``self`` so downstream code that treats ``rawlog.samples``
        as a sequence (``len``, iteration, indexing, slicing via
        ``samples_in_range``) keeps working without adding an
        ``isinstance`` dance everywhere. This is the same object as the
        ``LazyRawLog`` itself; do not mistake it for an eager list.
        """
        return self

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i):
        n = len(self.index)
        if isinstance(i, slice):
            # Duck-type parity with ``list``: slicing yields a list of
            # views, not another LazyRawLog. Consumers expect a concrete
            # sequence they can ``zip()`` against.
            return [self[j] for j in range(*i.indices(n))]
        if i < 0:
            i += n
        if i < 0 or i >= n:
            raise IndexError(f"sample index {i} out of range (0..{n - 1})")

        # LRU lookup and mutation happen under the lock so concurrent
        # requests cannot corrupt the OrderedDict internals.
        with self._lru_lock:
            cached = self._lru.get(i)
            if cached is not None:
                self._lru.move_to_end(i)
                return cached
            view = SampleView(
                rawlog=self,
                index_pos=i,
                offset=self.index.offsets[i],
                curtime=self.index.timestamps[i],
                ndeviat=self.index.ndeviats[i],
                scomplen=self.index.scomplens[i],
                pcomplen=self.index.pcomplens[i],
            )
            self._lru[i] = view
            if len(self._lru) > self._lru_size:
                self._lru.popitem(last=False)
            return view

    def __iter__(self) -> Iterator[SampleView]:
        for i in range(len(self.index)):
            yield self[i]

    # --- range helpers -----------------------------------------------------

    def slice_by_time(self, start: int, end: int) -> Iterator[SampleView]:
        """Yield views whose curtime is in the inclusive ``[start, end]`` window."""
        lo, hi = self.index.slice_by_time(start, end)
        for i in range(lo, hi):
            yield self[i]
