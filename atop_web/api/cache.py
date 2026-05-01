"""Tiny in-process LRU + TTL cache for the read endpoints.

Phase 23 T-22. The dashboard makes the same expensive read twice in
a row almost every time: once when the page first loads, once when
the user pans or adjusts a time filter without actually changing it.
A 5-minute TTL with 32 slots is enough to collapse those duplicate
requests without going near a real cache like Redis (forbidden by
the Phase 23 constraints).

Keys are tuples (session_id, endpoint, ...) so ``invalidate_session``
can wipe every entry for a given session when the user replaces the
session with a fresh upload. Values are whatever the builder returns;
we store them by reference (the builder is a route handler returning
a plain dict, so sharing a reference is safe as long as the caller
does not mutate it after returning).
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Tuple


# Defaults - the factory function below reads the env once at import
# time. Tests reimport the module with a patched env to flip them.
_DEFAULT_MAX_ENTRIES = 32
_DEFAULT_TTL_SECONDS = 300


class ResponseCache:
    """LRU + TTL cache keyed by tuples.

    Thread-safe via a single ``threading.Lock``. The lock is coarse
    but the critical sections are all constant time, so contention
    on the handful of routes that go through the cache stays cheap
    relative to the work we're skipping.
    """

    def __init__(
        self,
        *,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._store: "OrderedDict[Tuple, Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    # Separate hook so tests can freeze the clock without reaching
    # into ``time`` globally.
    def _now(self) -> float:
        return time.monotonic()

    def get_or_compute(self, key: Tuple, builder: Callable[[], Any]) -> Any:
        """Return the cached value for ``key``, or compute + store it."""
        now = self._now()
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                expiry, value = entry
                if expiry > now:
                    # Touch for LRU ordering.
                    self._store.move_to_end(key)
                    return value
                # Stale; fall through to rebuild.
                del self._store[key]
        # Build outside the lock so a slow builder doesn't block
        # reads for unrelated keys. We may race with another caller
        # computing the same key; that's fine - the second writer
        # will just overwrite.
        value = builder()
        now = self._now()
        with self._lock:
            self._store[key] = (now + self.ttl_seconds, value)
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)
        return value

    def invalidate_session(self, session_id: str) -> int:
        """Drop every key whose first element equals ``session_id``."""
        removed = 0
        with self._lock:
            for key in list(self._store.keys()):
                if key and key[0] == session_id:
                    del self._store[key]
                    removed += 1
        return removed

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


# ---------------------------------------------------------------------------
# Module-level singleton

_CACHE: ResponseCache | None = None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def get_response_cache() -> ResponseCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = ResponseCache(
            max_entries=_env_int("ATOP_RESPONSE_CACHE_MAX", _DEFAULT_MAX_ENTRIES),
            ttl_seconds=_env_int("ATOP_RESPONSE_CACHE_TTL", _DEFAULT_TTL_SECONDS),
        )
    return _CACHE
