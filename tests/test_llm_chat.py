"""Unit tests for the chat context builder and the range hint parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from atop_web.llm import chat, context
from atop_web.llm.provider import LLMProvider, LLMProviderError
from atop_web.parser.reader import parse_file


@pytest.fixture(scope="module")
def rawlog(rawlog_path: Path):
    return parse_file(rawlog_path, max_samples=30)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def test_build_all_context_has_spike_candidates(rawlog):
    ctx = context.build_all_context(rawlog)
    assert ctx["mode"] == "all"
    assert ctx["capture"]["sample_count"] == len(rawlog.samples)
    assert "aggregate" in ctx
    assert "top_processes_by_cpu" in ctx
    assert isinstance(ctx.get("spike_candidates", []), list)
    # Aggregate cpu_pct should be populated (numeric), not ``None``.
    assert ctx["aggregate"].get("cpu_pct") is not None


def test_build_all_context_empty_samples(rawlog):
    empty = type(rawlog)(header=rawlog.header, samples=[], spec=rawlog.spec)
    ctx = context.build_all_context(empty)
    assert ctx["mode"] == "all"
    assert ctx["capture"]["sample_count"] == 0
    assert ctx["note"] == "no samples"
    assert "aggregate" not in ctx


def test_build_range_context_filters_samples(rawlog):
    from datetime import datetime, timezone

    def to_iso(ts):
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    samples = rawlog.samples
    mid = samples[len(samples) // 2].curtime
    ctx = context.build_range_context(rawlog, samples[0].curtime, mid)
    assert ctx["mode"] == "range"
    assert ctx["range"]["sample_count"] <= len(samples)
    assert ctx["range"]["sample_count"] >= 1
    # Timestamps are ISO8601 strings so the model does not have to do
    # epoch -> ISO arithmetic and hallucinate dates.
    assert ctx["range"]["first"] == to_iso(samples[0].curtime)
    assert ctx["range"]["last"] <= to_iso(mid)


def test_build_range_context_no_samples_in_range(rawlog):
    ctx = context.build_range_context(rawlog, 1, 2)
    assert ctx["range"]["sample_count"] == 0
    assert ctx.get("note") == "no samples in requested range"


def test_build_range_context_handles_none_fields(rawlog_27_path):
    rawlog27 = parse_file(rawlog_27_path, max_samples=10)
    ctx = context.build_range_context(
        rawlog27,
        rawlog27.samples[0].curtime,
        rawlog27.samples[-1].curtime,
    )
    agg = ctx["aggregate"]
    # availablemem is not recorded on atop 2.7 so the aggregate is ``None``.
    assert agg["mem_available_mib"] is None
    # mem_used_mib is derived from (physmem - ...) so it is still computable.
    assert agg["mem_used_mib"] is not None


def test_serialize_context_fits_budget(rawlog):
    ctx = context.build_all_context(rawlog)
    text, truncated = context.serialize_context(ctx)
    assert len(text) <= context.MAX_CONTEXT_CHARS
    assert isinstance(truncated, bool)


def test_serialize_context_shrinks_when_oversized(rawlog, monkeypatch):
    ctx = context.build_all_context(rawlog)
    # Force the budget tiny so the shrink loop runs at least once.
    monkeypatch.setattr(context, "MAX_CONTEXT_CHARS", 400)
    text, truncated = context.serialize_context(ctx)
    assert len(text) <= 400
    assert truncated is True


# ---------------------------------------------------------------------------
# Range hint parser
# ---------------------------------------------------------------------------


def test_extract_range_hint_parses_single_tag():
    buf = 'prefix <range start="2026-04-27T14:00:00Z" end="2026-04-27T14:05:00Z" reason="cpu"/> suffix'
    hints, safe, remaining = chat.extract_range_hints(buf)
    assert len(hints) == 1
    assert hints[0] == {
        "start": "2026-04-27T14:00:00Z",
        "end": "2026-04-27T14:05:00Z",
        "reason": "cpu",
    }
    assert "range" not in safe
    assert "prefix " in safe and " suffix" in safe
    assert remaining == ""


def test_extract_range_hint_multiple_tags():
    buf = '<range start="a" end="b" reason="x"/>mid<range start="c" end="d" reason="y"/>end'
    hints, safe, remaining = chat.extract_range_hints(buf)
    assert len(hints) == 2
    assert [h["reason"] for h in hints] == ["x", "y"]
    assert safe == "mid" + "end"
    assert remaining == ""


def test_extract_range_hint_holds_partial_tag():
    buf = 'text <ran'
    hints, safe, remaining = chat.extract_range_hints(buf)
    assert hints == []
    assert safe == "text "
    assert remaining == "<ran"


def test_extract_range_hint_releases_unrelated_lt():
    buf = "compare a < b so"
    hints, safe, remaining = chat.extract_range_hints(buf)
    assert hints == []
    # ``<`` with no follow on never turns into <range, so it must flow through.
    # (We keep the buffering conservative but require that unrelated text
    # does not get held forever.)
    assert "compare" in safe


def test_extract_range_hint_handles_attr_order():
    buf = '<range reason="mem" end="2026-04-27T14:10:00Z" start="2026-04-27T14:00:00Z"/>'
    hints, safe, _ = chat.extract_range_hints(buf)
    assert hints[0]["start"] == "2026-04-27T14:00:00Z"
    assert hints[0]["reason"] == "mem"
    assert safe == ""


# ---------------------------------------------------------------------------
# stream_chat integration with a stub provider
# ---------------------------------------------------------------------------


class _StubStreamProvider(LLMProvider):
    name = "stub-stream"
    model = "stub"

    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    def health(self):
        return {"ok": True, "provider": self.name, "model": self.model, "detail": "stub"}

    def complete_json(self, system, user, schema):
        raise NotImplementedError

    def stream(self, system, user, history=None):
        for c in self._chunks:
            yield c


class _ErrorStreamProvider(LLMProvider):
    name = "stub-err"
    model = "stub"

    def health(self):
        return {"ok": False, "provider": self.name, "model": self.model, "detail": "down"}

    def complete_json(self, system, user, schema):
        raise NotImplementedError

    def stream(self, system, user, history=None):
        yield "partial"
        raise LLMProviderError("boom")


def test_stream_chat_emits_tokens_and_done(rawlog):
    provider = _StubStreamProvider(["hello ", "world"])
    req = chat.ChatRequest(message="whats up")
    events = list(chat.stream_chat(provider, rawlog, req))
    types = [e.type for e in events]
    assert "token" in types
    assert types[-1] == "done"
    combined = "".join(e.payload["text"] for e in events if e.type == "token")
    assert combined == "hello world"


def test_stream_chat_extracts_range_hints(rawlog):
    from datetime import datetime, timezone

    def iso(ts):
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    # Pick timestamps inside the fixture capture so the Phase 19 server
    # side validator does not drop the hint as a hallucination.
    samples = rawlog.samples
    inside_start = samples[0].curtime + 60
    inside_end = inside_start + 1800
    chunks = [
        "There is a ",
        f'<range start="{iso(inside_start)}" end="{iso(inside_end)}" reason="cpu"/>',
        " spike.",
    ]
    provider = _StubStreamProvider(chunks)
    req = chat.ChatRequest(message="find problems")
    events = list(chat.stream_chat(provider, rawlog, req))
    hints = [e.payload for e in events if e.type == "range_hint"]
    assert len(hints) == 1
    assert hints[0]["reason"] == "cpu"
    tokens = "".join(e.payload["text"] for e in events if e.type == "token")
    # The raw tag must never leak into the token stream.
    assert "<range" not in tokens
    assert "There is a " in tokens
    assert " spike." in tokens


def test_stream_chat_emits_error_on_provider_failure(rawlog):
    provider = _ErrorStreamProvider()
    req = chat.ChatRequest(message="anything")
    events = list(chat.stream_chat(provider, rawlog, req))
    types = [e.type for e in events]
    # Token("partial") first, then error, no done.
    assert "error" in types
    assert "done" not in types
    err = next(e for e in events if e.type == "error")
    assert "boom" in err.payload["message"]


def test_stream_chat_uses_range_mode_when_bounds_given(rawlog):
    provider = _StubStreamProvider(["ok"])
    mid = rawlog.samples[len(rawlog.samples) // 2].curtime
    req = chat.ChatRequest(
        message="what",
        time_range_start=rawlog.samples[0].curtime,
        time_range_end=mid,
    )
    events = list(chat.stream_chat(provider, rawlog, req))
    done = next(e for e in events if e.type == "done")
    assert done.payload["mode"] == "range"


def test_stream_chat_prompt_mentions_null_handling():
    # The shared system prompt tells the model to treat null as
    # ``not measured`` so Phase 16 behavior is preserved.
    from atop_web.llm import prompts

    assert "not measured" in prompts.SYSTEM_CHAT.lower()
    assert "<range" in prompts.SYSTEM_CHAT


def test_parse_iso_epoch_accepts_z_suffix():
    assert chat.parse_iso_epoch("2026-04-27T14:00:00Z") == 1777298400


def test_parse_iso_epoch_returns_none_for_empty():
    assert chat.parse_iso_epoch(None) is None
    assert chat.parse_iso_epoch("") is None
    assert chat.parse_iso_epoch("   ") is None


def test_parse_iso_epoch_rejects_invalid():
    with pytest.raises(ValueError):
        chat.parse_iso_epoch("not-a-date")
