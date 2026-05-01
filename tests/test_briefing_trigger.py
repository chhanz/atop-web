"""Phase 24 bug fix: briefing must be requested even if loadSamples fails.

The symptom on the 266 MB capture was that the AI briefing card never
populated. Root cause: ``onParseComplete`` ran
``await loadSamples()`` before ``requestBriefing`` was called, and
loadSamples issued an ALL-range ``/api/dashboard`` request that timed
out (Phase 24 ALL-range fix takes care of the timeout itself). A
timed-out or thrown loadSamples left ``requestBriefing`` unreached
and the briefing store was never asked to generate anything.

Fix: request the briefing *before* the samples fetch, and wrap the
samples call in try/catch so a slow or failing chart load never
prevents the briefing card from populating.

These tests scan ``app.js`` for the surviving control flow so a
future refactor that reintroduces the ordering bug fails CI.
"""

from __future__ import annotations

import re
from pathlib import Path

APP_JS = Path(__file__).resolve().parent.parent / "atop_web" / "static" / "app.js"


def _on_parse_complete_body() -> str:
    src = APP_JS.read_text()
    match = re.search(
        r"async function onParseComplete\([^)]*\)\s*\{(?P<body>.*?)\n  \}",
        src,
        re.DOTALL,
    )
    assert match, "onParseComplete function not found"
    return match.group("body")


def test_briefing_requested_before_load_samples():
    body = _on_parse_complete_body()
    brief_idx = body.find("requestBriefing")
    samples_idx = body.find("loadSamples")
    assert brief_idx != -1, "requestBriefing call missing"
    assert samples_idx != -1, "loadSamples call missing"
    assert brief_idx < samples_idx, (
        "requestBriefing must run before loadSamples so a slow or failing "
        "chart load does not block the briefing"
    )


def test_load_samples_call_is_guarded():
    """loadSamples has to live inside try/catch or be fire-and-forget."""
    body = _on_parse_complete_body()
    # Any of: try { ... loadSamples(...) ... } catch, or a .catch on
    # the returned promise. We enforce the try-wrap style for clarity.
    assert "try {" in body, "onParseComplete body should wrap loadSamples in try/catch"
    # Find the loadSamples call and verify it's inside a try block.
    try_idx = body.find("try {")
    samples_idx = body.find("loadSamples")
    catch_idx = body.find("} catch", try_idx)
    assert try_idx < samples_idx < catch_idx, (
        "loadSamples call must sit inside the try/catch block so a "
        "thrown error does not abort the rest of onParseComplete"
    )
