"""In process session store for parsed rawlog files.

Phase 22 T-05: sessions may now carry either a fully decoded
``RawLog`` (eager, pre-Phase-22 path) or a ``LazyRawLog`` plus a
``SampleIndex`` and pre-aggregate cache (lazy path). The union is
expressed with plain dataclass fields rather than a class hierarchy so
route handlers can dispatch on ``session.is_lazy`` without importing
both parser subsystems. ``clear()`` is the lifecycle hook that closes
the lazy path's file handle — leaking it would pin mmap pages and
a file descriptor on every rawlog re-upload.
"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Dict, Optional, Union

from atop_web.parser import RawLog

if TYPE_CHECKING:
    from atop_web.parser.aggregate import Aggregate
    from atop_web.parser.index import SampleIndex
    from atop_web.parser.lazy import LazyRawLog


@dataclass
class Session:
    session_id: str
    filename: str
    size_bytes: int
    # ``rawlog`` is the eager decode in the legacy path and the lazy
    # object in the new path. Route handlers should go through
    # ``is_lazy`` / the specialised accessors below instead of poking at
    # ``rawlog.samples`` directly.
    rawlog: Union[RawLog, "LazyRawLog"]
    is_lazy: bool = False
    # Lazy-only state. Populated by ``SessionStore.create_lazy``; ``None``
    # on eager sessions. ``source_path`` / ``file_handle`` are tracked
    # separately from ``LazyRawLog`` so ``clear()`` can close them even
    # if the LazyRawLog reference has already gone out of scope.
    source_path: Optional[Path] = None
    file_handle: Optional[BinaryIO] = None
    index: Optional["SampleIndex"] = None
    aggregate: Optional["Aggregate"] = None

    # -------- abstract access used by route handlers ----------------------

    def sample_count(self) -> int:
        if self.is_lazy:
            return len(self.rawlog)  # LazyRawLog __len__
        return len(self.rawlog.samples)

    def first_time(self) -> int | None:
        if self.is_lazy:
            return self.index.first_time() if self.index is not None else None
        samples = self.rawlog.samples
        return samples[0].curtime if samples else None

    def last_time(self) -> int | None:
        if self.is_lazy:
            return self.index.last_time() if self.index is not None else None
        samples = self.rawlog.samples
        return samples[-1].curtime if samples else None

    def median_interval_seconds(self) -> int | None:
        if self.is_lazy:
            return self.index.median_interval_seconds() if self.index is not None else None
        from atop_web.llm.context import _median_interval_seconds

        return _median_interval_seconds(self.rawlog.samples)

    def iter_samples(self):
        """Yield every sample in the session.

        For eager sessions this is ``rawlog.samples`` directly.
        For lazy sessions this yields ``SampleView`` objects; the caller
        sees an object whose attribute surface matches eager ``Sample``.
        """
        if self.is_lazy:
            return iter(self.rawlog)  # LazyRawLog __iter__
        return iter(self.rawlog.samples)

    def samples_in_range(self, start: int | None, end: int | None):
        """Return a concrete list of samples (views) in the inclusive window."""
        if self.is_lazy:
            if self.index is None:
                return []
            if start is None and end is None:
                return list(self.rawlog)
            lo = start if start is not None else -(1 << 62)
            hi = end if end is not None else (1 << 62)
            lo_i, hi_i = self.index.slice_by_time(lo, hi)
            return [self.rawlog[i] for i in range(lo_i, hi_i)]
        samples = self.rawlog.samples
        if start is None and end is None:
            return list(samples)
        lo = start if start is not None else -(1 << 62)
        hi = end if end is not None else (1 << 62)
        return [s for s in samples if lo <= s.curtime <= hi]

    def ndeviat_stats(self) -> tuple[float, int]:
        """Return ``(avg_per_sample, max_per_sample)`` for /api/summary."""
        if self.is_lazy:
            idx = self.index
            if idx is None or len(idx) == 0:
                return (0.0, 0)
            nds = idx.ndeviats
            n = len(nds)
            total = 0
            mx = 0
            for v in nds:
                total += v
                if v > mx:
                    mx = v
            return (total / n, mx)
        samples = self.rawlog.samples
        if not samples:
            return (0.0, 0)
        total = sum(s.ndeviat for s in samples)
        mx = max(s.ndeviat for s in samples)
        return (total / len(samples), mx)


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, Session] = {}

    def create(self, filename: str, size_bytes: int, rawlog: RawLog) -> Session:
        """Legacy eager path: ``rawlog`` is a fully-decoded ``RawLog``."""
        session_id = secrets.token_urlsafe(12)
        session = Session(
            session_id=session_id,
            filename=filename,
            size_bytes=size_bytes,
            rawlog=rawlog,
            is_lazy=False,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def create_lazy(
        self,
        filename: str,
        size_bytes: int,
        lazy_rawlog: "LazyRawLog",
        aggregate: Optional["Aggregate"] = None,
    ) -> Session:
        """Lazy path: the session owns a ``LazyRawLog`` and its file handle."""
        session_id = secrets.token_urlsafe(12)
        session = Session(
            session_id=session_id,
            filename=filename,
            size_bytes=size_bytes,
            rawlog=lazy_rawlog,
            is_lazy=True,
            source_path=lazy_rawlog._source_path,
            file_handle=lazy_rawlog._file,
            index=lazy_rawlog.index,
            aggregate=aggregate,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def require(self, session_id: str) -> Session:
        session = self.get(session_id)
        if session is None:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="session not found")
        return session

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())

    def clear(self) -> None:
        # Close every file handle the lazy sessions held before dropping
        # the dict. The LazyRawLog's ``close()`` covers both the handle
        # and its LRU cache; fall back to a bare handle close if the
        # lazy object is somehow gone.
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for sess in sessions:
            if not sess.is_lazy:
                continue
            lazy = sess.rawlog
            close = getattr(lazy, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            fh = sess.file_handle
            if fh is not None and not getattr(fh, "closed", True):
                try:
                    fh.close()
                except Exception:
                    pass


_STORE = SessionStore()


def get_store() -> SessionStore:
    return _STORE
