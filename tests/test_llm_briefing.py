"""Tests for the briefing input builder and schema validation."""

from __future__ import annotations

import json

import pytest

from atop_web.llm import briefing as briefing_mod
from atop_web.llm.briefing import (
    MAX_INPUT_CHARS,
    _fit_to_budget,
    build_briefing_input,
    generate_briefing,
)
from atop_web.llm.provider import LLMProvider, LLMProviderError
from atop_web.llm.schema import validate_briefing


def test_validate_briefing_handles_garbage():
    assert validate_briefing("not-a-dict") == {"issues": []}
    assert validate_briefing({"issues": "nope"}) == {"issues": []}
    assert validate_briefing({}) == {"issues": []}


def test_validate_briefing_normalizes_entries():
    out = validate_briefing(
        {
            "issues": [
                {"title": " t ", "severity": "CRITICAL", "detail": "d"},
                {"title": "n", "severity": "bogus", "detail": "x"},
                {"detail": "no-title"},
                "string",
            ]
        }
    )
    # Only the first three valid entries are kept and normalized.
    assert len(out["issues"]) == 3
    assert out["issues"][0]["severity"] == "critical"
    assert out["issues"][0]["title"] == "t"
    assert out["issues"][1]["severity"] == "info"  # bogus -> info


def test_validate_briefing_caps_at_three():
    raw = {"issues": [
        {"title": str(i), "severity": "info", "detail": "d"} for i in range(10)
    ]}
    out = validate_briefing(raw)
    assert len(out["issues"]) == 3


def test_build_briefing_input_from_real_rawlog(rawlog_path):
    from atop_web.parser import parse_file

    rawlog = parse_file(rawlog_path, max_samples=4)
    payload = build_briefing_input(rawlog)

    assert payload["capture"]["hostname"]
    assert payload["capture"]["sample_count"] == len(rawlog.samples)
    assert "cpu" in payload and "memory" in payload
    assert payload["disk"]["devices"]
    assert payload["network"]["interfaces"]
    assert payload["processes_last"]["by_cpu"]

    # No command lines must leak; the row only has a name.
    first_proc = payload["processes_last"]["by_cpu"][0]
    assert "cmdline" not in first_proc


def test_fit_to_budget_drops_processes_when_over_limit():
    # Build a synthetic oversized payload.
    payload = {
        "capture": {"hostname": "x"},
        "processes_first": {
            "by_cpu": [
                {"pid": i, "name": "proc", "cpu_ticks": 1, "rmem_kb": 1,
                 "vmem_kb": 1, "dsk_read_sectors": 0, "dsk_write_sectors": 0}
                for i in range(5000)
            ],
            "by_rss": [],
        },
        "processes_last": {"by_cpu": [], "by_rss": []},
    }
    text, truncated = _fit_to_budget(payload, budget=4000)
    assert len(text) <= 4000
    assert truncated is True


def test_fit_to_budget_leaves_small_payload_alone():
    payload = {"hello": "world"}
    text, truncated = _fit_to_budget(payload, budget=MAX_INPUT_CHARS)
    assert truncated is False
    assert len(text) < 100


class _StubProvider(LLMProvider):
    name = "stub"

    def __init__(self, response):
        self._response = response

    def health(self) -> dict:
        return {"ok": True, "provider": "stub", "model": None, "detail": ""}

    def complete_json(self, system, user, schema):
        self.last_call = (system, user, schema)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response

    def stream(self, system, user, history=None):
        yield ""


def test_generate_briefing_runs_provider_and_validates(rawlog_path):
    from atop_web.parser import parse_file

    rawlog = parse_file(rawlog_path, max_samples=3)
    stub = _StubProvider(
        {
            "issues": [
                {"title": "High CPU", "severity": "warning", "detail": "d"},
                {"title": "extra", "severity": "info", "detail": "d"},
                {"title": "extra2", "severity": "info", "detail": "d"},
                {"title": "drop me", "severity": "info", "detail": "d"},
            ]
        }
    )
    out = generate_briefing(stub, rawlog)
    assert len(out["issues"]) == 3
    assert out["issues"][0]["title"] == "High CPU"
    # system/user were actually passed.
    assert "atop" in stub.last_call[0]
    assert len(stub.last_call[1]) <= MAX_INPUT_CHARS + 200  # truncation note slack


def test_generate_briefing_propagates_provider_error(rawlog_path):
    from atop_web.parser import parse_file

    rawlog = parse_file(rawlog_path, max_samples=2)
    stub = _StubProvider(LLMProviderError("boom"))
    with pytest.raises(LLMProviderError):
        generate_briefing(stub, rawlog)


def test_build_briefing_input_from_2_7_rawlog_handles_missing_fields(rawlog_27_path):
    """atop 2.7 rawlogs have no availablemem and no per disk inflight.

    The briefing must still build without crashing and must propagate the
    missing signal (``None`` / ``null``) instead of substituting a zero.
    """
    from atop_web.parser import parse_file

    rawlog = parse_file(rawlog_27_path, max_samples=3)
    payload = build_briefing_input(rawlog)

    # Memory bucket has all the usual counters but ``available_mib`` must be
    # null so the model treats it as "not measured".
    for bucket_name in ("first", "last"):
        mem_bucket = payload["memory"][bucket_name]
        assert mem_bucket is not None
        assert mem_bucket["available_mib"] is None
        assert mem_bucket["physmem_mib"] > 0

    # Disk devices carry ``inflight=None`` on 2.7 rawlogs.
    for dev in payload["disk"]["devices"]:
        assert dev["inflight"] is None

    # The payload must serialise to JSON cleanly (json.dumps handles None).
    json.dumps(payload)
