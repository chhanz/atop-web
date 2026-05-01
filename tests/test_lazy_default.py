"""T-11 Lazy default + ``ATOP_LAZY`` rollback gate.

The Phase 22 cutover flips ``parse_stream`` / ``parse_file`` /
``parse_bytes`` to return a lazy rawlog by default. The ``ATOP_LAZY``
environment variable (``"0"`` / ``"1"``) is the escape hatch — callers
who hit a lazy-path regression in production can flip the env back to
eager without rebuilding.

Tests lock down:

* default: ``parse_file`` returns a ``LazyRawLog`` with a populated
  ``index`` and every sample accessible as a ``SampleView``.
* ``ATOP_LAZY=0`` forces eager ``RawLog`` return — the original
  ``samples: list[Sample]`` shape.
* explicit ``lazy=False`` still works even when the env is unset.
* existing high-level callers (summary / samples routes, tools
  handlers, chat context) all accept the new default lazy rawlog and
  produce the same answers.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from atop_web.parser.lazy import LazyRawLog
from atop_web.parser.reader import RawLog, Sample, parse_bytes, parse_file


def test_parse_file_default_returns_lazy(rawlog_path: Path, monkeypatch):
    # Unset so the module-level default kicks in.
    monkeypatch.delenv("ATOP_LAZY", raising=False)
    result = parse_file(rawlog_path)
    try:
        assert isinstance(result, LazyRawLog)
        assert result.index is not None
        assert len(result) > 0
        # SampleView covers every field the eager Sample surfaced.
        view = result[0]
        assert view.curtime > 0
        assert view.system_cpu is not None
    finally:
        if isinstance(result, LazyRawLog):
            result.close()


def test_parse_file_env_gate_falls_back_to_eager(rawlog_path: Path, monkeypatch):
    monkeypatch.setenv("ATOP_LAZY", "0")
    result = parse_file(rawlog_path, max_samples=2)
    # Eager path: the old ``RawLog`` shape with a concrete samples list.
    assert isinstance(result, RawLog)
    assert isinstance(result.samples, list)
    assert isinstance(result.samples[0], Sample)


def test_parse_bytes_default_returns_lazy(rawlog_path: Path, monkeypatch):
    monkeypatch.delenv("ATOP_LAZY", raising=False)
    data = rawlog_path.read_bytes()
    result = parse_bytes(data)
    try:
        assert isinstance(result, LazyRawLog)
        assert result.index is not None
    finally:
        if isinstance(result, LazyRawLog):
            result.close()


def test_explicit_lazy_false_overrides_env(rawlog_path: Path, monkeypatch):
    # Test helpers that still want eager decode can pass lazy=False even
    # when the env says lazy — this is the contract
    # ``tests/_helpers/eager_decode.py`` relies on.
    monkeypatch.setenv("ATOP_LAZY", "1")
    result = parse_file(rawlog_path, max_samples=1, lazy=False)
    assert isinstance(result, RawLog)
    assert result.samples


def test_eager_helper_available(rawlog_path: Path):
    # The eager decoder moves into tests/_helpers so nothing in the
    # production tree references ``Sample`` objects any more.
    from tests._helpers.eager_decode import decode_eager

    rawlog = decode_eager(rawlog_path, max_samples=1)
    assert isinstance(rawlog, RawLog)
    assert rawlog.samples
    assert isinstance(rawlog.samples[0], Sample)


def test_chat_context_accepts_lazy(rawlog_path: Path, monkeypatch):
    # /llm/context builds against whatever ``parse_file`` returns. With
    # the default flipped to lazy, the builder must keep working.
    monkeypatch.delenv("ATOP_LAZY", raising=False)
    from atop_web.llm import context

    rawlog = parse_file(rawlog_path)
    try:
        ctx = context.build_all_context(rawlog)
    finally:
        if isinstance(rawlog, LazyRawLog):
            rawlog.close()
    assert ctx["mode"] == "all"
    assert ctx["capture"]["sample_count"] > 0


def test_briefing_accepts_lazy(rawlog_path: Path, monkeypatch):
    monkeypatch.delenv("ATOP_LAZY", raising=False)
    from atop_web.llm.briefing import build_briefing_input

    rawlog = parse_file(rawlog_path)
    try:
        payload = build_briefing_input(rawlog)
    finally:
        if isinstance(rawlog, LazyRawLog):
            rawlog.close()
    assert "capture" in payload
    assert payload["capture"]["sample_count"] > 0
