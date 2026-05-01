"""Unit tests for the dual eager/lazy ``Session`` store.

Phase 22 T-05. Sessions need to carry either a fully decoded ``RawLog``
or a ``LazyRawLog`` plus its index and aggregate, and ``clear()`` must
close any file handles the lazy path holds. Eager sessions continue to
work unchanged — lazy is additive until T-11 flips the default.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atop_web.api.sessions import Session, SessionStore
from atop_web.parser import parse_file
from atop_web.parser.aggregate import Aggregate, build_aggregate
from atop_web.parser.lazy import LazyRawLog


def test_eager_create_still_works(rawlog_path: Path):
    # Regression: the pre-Phase-22 ``create`` signature must keep working.
    eager = parse_file(rawlog_path, max_samples=3)
    store = SessionStore()
    sess = store.create(filename="f", size_bytes=100, rawlog=eager)
    assert isinstance(sess, Session)
    assert sess.is_lazy is False
    assert sess.rawlog is eager
    # Lazy-only fields are ``None`` on eager sessions.
    assert sess.index is None
    assert sess.aggregate is None
    assert sess.source_path is None
    assert sess.file_handle is None


def test_create_lazy_populates_lazy_fields(rawlog_path: Path):
    lazy = LazyRawLog.open(rawlog_path)
    try:
        store = SessionStore()
        sess = store.create_lazy(
            filename="f",
            size_bytes=rawlog_path.stat().st_size,
            lazy_rawlog=lazy,
        )
        assert sess.is_lazy is True
        # ``sess.rawlog`` still exists for downstream callers that expect
        # the field, but it's the lazy object — routes inspect
        # ``is_lazy`` to decide which code path to take.
        assert sess.rawlog is lazy
        assert sess.index is lazy.index
        assert sess.file_handle is lazy._file
        assert sess.source_path == rawlog_path
        # Aggregate defaults to ``None`` — callers pass it in explicitly
        # when they've built it.
        assert sess.aggregate is None
    finally:
        lazy.close()


def test_create_lazy_accepts_aggregate(rawlog_path: Path):
    lazy = LazyRawLog.open(rawlog_path)
    agg = build_aggregate(lazy)
    try:
        store = SessionStore()
        sess = store.create_lazy(
            filename="f",
            size_bytes=1,
            lazy_rawlog=lazy,
            aggregate=agg,
        )
        assert sess.aggregate is agg
        assert isinstance(sess.aggregate, Aggregate)
    finally:
        lazy.close()


def test_clear_closes_lazy_file_handle(rawlog_path: Path):
    lazy = LazyRawLog.open(rawlog_path)
    store = SessionStore()
    store.create_lazy(filename="f", size_bytes=1, lazy_rawlog=lazy)
    fh = lazy._file
    assert fh.closed is False
    store.clear()
    # After clear, any file handle the session held must be closed so
    # the process does not leak fds when a user re-uploads a rawlog.
    assert fh.closed is True


def test_clear_leaves_eager_sessions_intact(rawlog_path: Path):
    # Regression: clear() on a store that mixes eager + lazy sessions
    # must not raise when the eager side has no file handle.
    eager = parse_file(rawlog_path, max_samples=1)
    store = SessionStore()
    store.create(filename="eager", size_bytes=1, rawlog=eager)
    lazy = LazyRawLog.open(rawlog_path)
    store.create_lazy(filename="lazy", size_bytes=1, lazy_rawlog=lazy)
    store.clear()
    # Both slots cleared.
    assert store.list_ids() == []


def test_require_404_still_works():
    from fastapi import HTTPException

    store = SessionStore()
    with pytest.raises(HTTPException) as ei:
        store.require("missing")
    assert ei.value.status_code == 404
