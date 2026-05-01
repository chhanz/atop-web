"""T-22 TTL response cache: hit/miss/invalidate semantics.

Phase 23 adds a small in-process LRU with per-entry TTL in front of
the heavy read endpoints (charts, processes, dashboard). The cache
must:

* return the memoized value instead of re-running the builder on a
  second call with the same key, while the TTL has not expired;
* drop a stale entry when the TTL has expired and rebuild;
* evict the oldest entry when the capacity cap is hit;
* drop every entry whose key starts with a given session prefix when
  that session is invalidated (new upload replaces it, or a future
  DELETE endpoint wipes it).

These tests exercise the cache helper in isolation so the route
integration tests can trust it.
"""

from __future__ import annotations

import time

import pytest

from atop_web.api.cache import ResponseCache


def test_hit_avoids_rebuild():
    cache = ResponseCache(max_entries=8, ttl_seconds=60)
    calls = 0

    def builder():
        nonlocal calls
        calls += 1
        return {"v": calls}

    v1 = cache.get_or_compute(("s1", "summary"), builder)
    v2 = cache.get_or_compute(("s1", "summary"), builder)
    assert v1 == {"v": 1}
    assert v2 == {"v": 1}
    assert calls == 1


def test_different_keys_miss_independently():
    cache = ResponseCache(max_entries=8, ttl_seconds=60)
    cache.get_or_compute(("s1", "a"), lambda: "a")
    cache.get_or_compute(("s1", "b"), lambda: "b")
    assert cache.get_or_compute(("s1", "a"), lambda: "X") == "a"
    assert cache.get_or_compute(("s1", "b"), lambda: "X") == "b"


def test_ttl_expiry_triggers_rebuild(monkeypatch):
    cache = ResponseCache(max_entries=8, ttl_seconds=1)
    now = [1000.0]
    monkeypatch.setattr(cache, "_now", lambda: now[0])

    calls = 0

    def builder():
        nonlocal calls
        calls += 1
        return calls

    assert cache.get_or_compute(("s1", "k"), builder) == 1
    now[0] += 0.5
    assert cache.get_or_compute(("s1", "k"), builder) == 1
    now[0] += 1.0  # now past TTL
    assert cache.get_or_compute(("s1", "k"), builder) == 2


def test_lru_eviction_at_capacity():
    cache = ResponseCache(max_entries=3, ttl_seconds=60)
    cache.get_or_compute(("s1", "a"), lambda: "a")
    cache.get_or_compute(("s1", "b"), lambda: "b")
    cache.get_or_compute(("s1", "c"), lambda: "c")
    # Touch 'a' so 'b' becomes the LRU entry.
    cache.get_or_compute(("s1", "a"), lambda: "X")
    # Inserting a new key pushes out the LRU entry.
    cache.get_or_compute(("s1", "d"), lambda: "d")
    # 'b' must have been evicted; the rebuild therefore runs.
    calls = [0]

    def rebuild():
        calls[0] += 1
        return "b2"

    got = cache.get_or_compute(("s1", "b"), rebuild)
    assert got == "b2"
    assert calls[0] == 1


def test_invalidate_session_drops_matching_keys():
    cache = ResponseCache(max_entries=8, ttl_seconds=60)
    cache.get_or_compute(("s1", "a"), lambda: "a")
    cache.get_or_compute(("s1", "b"), lambda: "b")
    cache.get_or_compute(("s2", "a"), lambda: "a2")
    cache.invalidate_session("s1")
    # Rebuild counters.
    built: list[str] = []

    def mk(label):
        def _b():
            built.append(label)
            return label

        return _b

    cache.get_or_compute(("s1", "a"), mk("s1a"))
    cache.get_or_compute(("s1", "b"), mk("s1b"))
    cache.get_or_compute(("s2", "a"), mk("s2a"))
    assert built == ["s1a", "s1b"]  # only s2 survived invalidation


def test_cache_is_thread_safe():
    import threading

    cache = ResponseCache(max_entries=64, ttl_seconds=60)
    running = [True]
    errors: list[str] = []

    def worker(i):
        for j in range(100):
            try:
                cache.get_or_compute((f"s{i}", str(j)), lambda: (i, j))
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(str(exc))
                running[0] = False

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_env_ttl_override(monkeypatch):
    """ATOP_RESPONSE_CACHE_TTL sets the default TTL for the global cache."""
    monkeypatch.setenv("ATOP_RESPONSE_CACHE_TTL", "2")
    import importlib

    import atop_web.api.cache as cache_mod

    importlib.reload(cache_mod)
    assert cache_mod.get_response_cache().ttl_seconds == 2
    # Reset for the rest of the suite.
    monkeypatch.delenv("ATOP_RESPONSE_CACHE_TTL", raising=False)
    importlib.reload(cache_mod)
