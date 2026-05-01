"""T-23 frontend dashboard uses a single fetch.

The page used to fan out to five parallel fetches on first paint
(samples + four system_* charts) plus a sixth for processes. Phase
23 collapses those into one ``/api/dashboard`` call. This test greps
``app.js`` for the surviving call-sites so a refactor that drops back
into the old pattern fails CI instead of silently re-creating the
slow page load.
"""

from __future__ import annotations

import re
from pathlib import Path

APP_JS = Path(__file__).resolve().parent.parent / "atop_web" / "static" / "app.js"


def test_load_samples_uses_dashboard_endpoint():
    src = APP_JS.read_text()
    assert re.search(r"fetch\(`api/dashboard\?", src), (
        "loadSamples should call /api/dashboard for the combined payload"
    )


def test_load_samples_drops_parallel_system_fetches():
    src = APP_JS.read_text()
    # The old implementation issued four system_* fetches in parallel
    # from loadSamples. The new implementation must not reintroduce
    # them (server-pick and ad-hoc tools can still call them; scope
    # this check to the ``loadSamples`` function body).
    match = re.search(
        r"async function loadSamples\(\)[^}]*\{(?P<body>.*?)\n  \}",
        src,
        re.DOTALL,
    )
    assert match, "loadSamples function not found"
    body = match.group("body")
    for forbidden in (
        "api/samples/system_memory",
        "api/samples/system_cpu",
        "api/samples/system_disk",
        "api/samples/system_network",
    ):
        assert forbidden not in body, (
            f"loadSamples must not call {forbidden!r} directly any more"
        )


def test_processes_table_renders_from_dashboard_payload():
    src = APP_JS.read_text()
    # The combined payload already carries the last-sample processes
    # table; the frontend must render it without a follow-up fetch.
    match = re.search(
        r"async function loadSamples\(\)[^}]*\{(?P<body>.*?)\n  \}",
        src,
        re.DOTALL,
    )
    body = match.group("body")
    assert "body.processes" in body, (
        "loadSamples should reuse body.processes from /api/dashboard"
    )
