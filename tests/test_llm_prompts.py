"""Prompt contract tests.

These are cheap string checks that guard the user facing behavior: if
someone softens the range pinpointing instructions, the chat endpoint
silently regresses into the "zoom in yourself" failure mode from Phase 17.
Catching the regression at the prompt string level saves a full LLM round
trip.
"""

from __future__ import annotations

from atop_web.llm import prompts


def test_system_chat_mentions_range_tag_shape():
    text = prompts.SYSTEM_CHAT
    assert "<range" in text
    # Required attributes must be named explicitly so the model matches
    # the exact tag our parser recognizes.
    assert 'start="' in text
    assert 'end="' in text
    # ``label="..."`` is the new preferred attribute name since Phase 19;
    # the parser still accepts ``reason=`` for back compat but the prompt
    # now teaches the new form exclusively.
    assert 'label="' in text
    assert "ISO8601" in text


def test_system_chat_instructs_model_to_emit_range_tag():
    text = prompts.SYSTEM_CHAT.lower()
    # The prompt must use strong wording so the model actually emits the
    # tag instead of saying "you could zoom in from X to Y" in prose.
    assert "must" in text
    # Phase 19 explicitly names the "zoom in" prose mistake so the model
    # recognizes the pattern even when the phrasing is different.
    assert "zoom in" in text


def test_system_chat_forbids_fabricated_timestamps():
    text = prompts.SYSTEM_CHAT.lower()
    # Phase 19 replaced "never fabricate timestamps" with a more specific
    # rule set: do not do arithmetic, copy verbatim, stay inside the
    # capture window. Check for the capture window clause since that is
    # the anti hallucination rule that actually drops bad tags server
    # side.
    assert "capture.start" in text
    assert "capture.end" in text
    assert "must fall inside" in text or "inside the capture" in text


def test_system_chat_allows_markdown():
    # Phase 18-c rendered assistant output as sanitized markdown; the
    # prompt must explicitly allow it so the model stops apologizing for
    # using bullet points.
    text = prompts.SYSTEM_CHAT.lower()
    assert "markdown" in text


def test_system_chat_still_treats_null_as_not_measured():
    # Phase 16 regression guard: 2.7 captures ship null availablemem /
    # inflight, and the prompt must keep the "not measured" semantic.
    text = prompts.SYSTEM_CHAT.lower()
    assert "not measured" in text


def test_system_briefing_unchanged_style_directives():
    # The Level 1 briefing remains JSON only; this catches accidental
    # edits that would break the schema validator.
    text = prompts.SYSTEM_BRIEFING
    assert "single JSON object" in text
    assert "metric_hint" in text


def test_system_chat_mentions_sample_interval_rule():
    # Phase 18.5: the prompt must tell the model to respect the
    # ``recommended_min_range_seconds`` field so tags are wide enough to
    # contain samples. Without this line the model happily emits 10
    # minute ranges on a capture with 559s intervals.
    text = prompts.SYSTEM_CHAT
    assert "interval_seconds" in text
    assert "recommended_min_range_seconds" in text
    # An explicit "no samples in range" warning catches the exact user
    # facing symptom so future prompt edits cannot silently drop it.
    assert "no samples in range" in text.lower()
