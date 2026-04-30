"""Phase 19 regression guards for range_hint validation + ISO8601 context.

These tests focus on the hallucination failure mode David reported: the
model emitted a ``<range/>`` tag with a date two days earlier than the
capture actually covered. The fixes span three places:

* context timestamps are pre formatted ISO8601 UTC strings so the model
  does not have to do epoch arithmetic;
* ``_validate_and_widen_hint`` drops any hint whose start/end falls
  outside the capture window;
* narrow hints (width < interval*2) are auto widened so the filtered
  query actually returns samples.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atop_web.llm import chat, context
from atop_web.parser.reader import Sample, parse_file


@pytest.fixture(scope="module")
def rawlog(rawlog_path: Path):
    return parse_file(rawlog_path, max_samples=30)


# ---------------------------------------------------------------------------
# Context builder ships ISO8601 strings, never raw epochs
# ---------------------------------------------------------------------------


def _is_iso_utc(value: str) -> bool:
    return (
        isinstance(value, str)
        and value.endswith("Z")
        and len(value) == len("2026-04-27T14:00:00Z")
    )


def test_context_timestamps_are_iso8601(rawlog):
    # ``mode: all`` - capture block plus every spike candidate window.
    ctx = context.build_all_context(rawlog)
    cap = ctx["capture"]
    assert _is_iso_utc(cap["start"]), cap["start"]
    assert _is_iso_utc(cap["end"]), cap["end"]
    # ``*_epoch`` aliases are still integers for internal use; the ISO
    # strings are what the model sees in the JSON payload.
    assert isinstance(cap["start_epoch"], int)
    assert isinstance(cap["end_epoch"], int)
    for spike in ctx.get("spike_candidates", []):
        assert _is_iso_utc(spike["start"]), spike
        assert _is_iso_utc(spike["end"]), spike
        assert _is_iso_utc(spike["center"]), spike

    # ``mode: range`` - both capture and range blocks.
    samples = rawlog.samples
    rctx = context.build_range_context(
        rawlog, samples[0].curtime, samples[-1].curtime
    )
    assert _is_iso_utc(rctx["capture"]["start"])
    assert _is_iso_utc(rctx["capture"]["end"])
    assert _is_iso_utc(rctx["range"]["start"])
    assert _is_iso_utc(rctx["range"]["end"])
    assert _is_iso_utc(rctx["range"]["first"])
    assert _is_iso_utc(rctx["range"]["last"])


# ---------------------------------------------------------------------------
# Hint validation
# ---------------------------------------------------------------------------


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def test_range_hint_dropped_when_outside_capture():
    # Capture: 2026-04-27 07:22 .. 07:30 (8 minutes of fake samples).
    cap_start = 1777864920
    cap_end = cap_start + 600
    # Hallucinated: two days earlier - well outside the window.
    hint = {
        "start": _iso(cap_start - 2 * 86400),
        "end": _iso(cap_start - 2 * 86400 + 300),
        "reason": "hallucinated",
    }
    out = chat._validate_and_widen_hint(hint, cap_start, cap_end, 60)
    assert out is None


def test_range_hint_kept_when_inside_capture():
    cap_start = 1777864920
    cap_end = cap_start + 3600
    hint = {
        "start": _iso(cap_start + 300),
        "end": _iso(cap_start + 1500),
        "reason": "cpu spike",
    }
    out = chat._validate_and_widen_hint(hint, cap_start, cap_end, 60)
    assert out is not None
    assert out["reason"] == "cpu spike"
    assert _is_iso_utc(out["start"])
    assert _is_iso_utc(out["end"])


def test_range_hint_auto_expand_when_too_narrow():
    # Interval 600s -> min width is 1200s. A 60s hint must be widened.
    cap_start = 1777864920
    cap_end = cap_start + 86400
    center = cap_start + 3600
    hint = {
        "start": _iso(center),
        "end": _iso(center + 60),
        "reason": "cpu moment",
    }
    out = chat._validate_and_widen_hint(hint, cap_start, cap_end, 600)
    assert out is not None
    width_seconds = chat.parse_iso_epoch(out["end"]) - chat.parse_iso_epoch(
        out["start"]
    )
    assert width_seconds >= 1200
    assert out.get("widened") is True


# ---------------------------------------------------------------------------
# Tag parsing with and without label attribute
# ---------------------------------------------------------------------------


def test_extract_range_hints_parses_valid_tag():
    buf = (
        'intro '
        '<range start="2026-04-27T14:00:00Z" end="2026-04-27T14:10:00Z" '
        'label="CPU spike"/>'
        ' trailer'
    )
    hints, safe, remaining = chat.extract_range_hints(buf)
    assert len(hints) == 1
    h = hints[0]
    assert h["start"] == "2026-04-27T14:00:00Z"
    assert h["end"] == "2026-04-27T14:10:00Z"
    assert h["reason"] == "CPU spike"
    assert "<range" not in safe
    assert remaining == ""


def test_extract_range_hints_falls_back_to_default_label_when_missing():
    # Neither ``label`` nor ``reason`` is present - parser must supply a
    # default so the UI badge still has something to render.
    buf = '<range start="2026-04-27T14:00:00Z" end="2026-04-27T14:10:00Z"/>'
    hints, _, _ = chat.extract_range_hints(buf)
    assert len(hints) == 1
    assert hints[0]["reason"] == chat.DEFAULT_RANGE_LABEL


def test_extract_range_hints_accepts_legacy_reason_attribute():
    # Back compat: pre Phase 19 prompt still uses ``reason``. The parser
    # should prefer ``label`` but fall back to ``reason`` when only the
    # legacy name is present.
    buf = '<range start="a" end="b" reason="legacy"/>'
    hints, _, _ = chat.extract_range_hints(buf)
    assert hints[0]["reason"] == "legacy"


# ---------------------------------------------------------------------------
# End to end: stream_chat drops a hallucinated hint, keeps a good one
# ---------------------------------------------------------------------------


class _ScriptedProvider:
    """Minimal stand in so we can exercise stream_chat without Bedrock."""

    name = "scripted"
    model = "stub"

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def health(self):
        return {"ok": True, "provider": self.name, "model": self.model, "detail": ""}

    def complete_json(self, system, user, schema):
        raise NotImplementedError

    def supports_tools(self) -> bool:
        # Phase 20 added a tool use branch in ``stream_chat``; this
        # legacy test exercises the plain streaming path, so the stub
        # must explicitly opt out.
        return False

    def stream(self, system, user, history=None):
        for c in self._chunks:
            yield c


def test_stream_chat_drops_hallucinated_range_hint(rawlog):
    cap_start = rawlog.samples[0].curtime
    inside = cap_start + 300
    inside_end = inside + 1800
    # Fabricated date: 30 days before the capture.
    bad_start = cap_start - 30 * 86400
    bad_end = bad_start + 600
    chunks = [
        "Here is one good and one bad range: ",
        f'<range start="{_iso(inside)}" end="{_iso(inside_end)}" label="good"/>',
        " and ",
        f'<range start="{_iso(bad_start)}" end="{_iso(bad_end)}" label="bad"/>',
        " done.",
    ]
    provider = _ScriptedProvider(chunks)
    req = chat.ChatRequest(message="show spikes")
    events = list(chat.stream_chat(provider, rawlog, req))
    emitted_labels = [
        e.payload.get("reason") for e in events if e.type == "range_hint"
    ]
    assert "good" in emitted_labels
    assert "bad" not in emitted_labels
