"""T-21 /api/dashboard fan-out endpoint.

The dashboard view today makes six independent fetches (summary, four
system_* charts, processes). Each of them pays its own session lookup
plus a chunk of sstat inflates, and three-hour windows end up paying
the same inflate work four-over for no good reason. ``/api/dashboard``
collapses the six fetches into one so the server can share work and
the browser only waits on a single round trip.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from atop_web.api.sessions import get_store
from atop_web.main import app
from atop_web.parser import parse_file


@pytest.fixture
def client():
    get_store().clear()
    with TestClient(app) as c:
        yield c
    get_store().clear()


@pytest.fixture
def session(rawlog_path: Path):
    eager = parse_file(rawlog_path, lazy=False)
    store = get_store()
    sess = store.create(
        filename="f", size_bytes=rawlog_path.stat().st_size, rawlog=eager
    )
    return sess


def _iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_dashboard_returns_all_sections(client, session):
    r = client.get("/api/dashboard", params={"session": session.session_id})
    assert r.status_code == 200, r.text
    body = r.json()
    for key in ("summary", "samples", "charts", "processes"):
        assert key in body, f"missing section: {key}"
    assert "cpu" in body["charts"]
    assert "memory" in body["charts"]
    assert "disk" in body["charts"]
    assert "network" in body["charts"]


def test_dashboard_summary_matches_summary_endpoint(client, session):
    r = client.get("/api/dashboard", params={"session": session.session_id})
    combined = r.json()
    r2 = client.get("/api/summary", params={"session": session.session_id})
    expected = r2.json()
    assert combined["summary"] == expected


def test_dashboard_samples_matches_samples_endpoint(client, session):
    r = client.get("/api/dashboard", params={"session": session.session_id})
    combined = r.json()
    r2 = client.get("/api/samples", params={"session": session.session_id})
    expected = r2.json()
    assert combined["samples"] == expected


def test_dashboard_charts_match_system_endpoints(client, session):
    r = client.get("/api/dashboard", params={"session": session.session_id})
    combined = r.json()
    for metric, path in (
        ("cpu", "/api/samples/system_cpu"),
        ("memory", "/api/samples/system_memory"),
        ("disk", "/api/samples/system_disk"),
        ("network", "/api/samples/system_network"),
    ):
        expected = client.get(path, params={"session": session.session_id}).json()
        assert combined["charts"][metric] == expected, metric


def test_dashboard_processes_match_processes_endpoint(client, session):
    r = client.get(
        "/api/dashboard",
        params={"session": session.session_id, "process_limit": 5},
    )
    combined = r.json()
    r2 = client.get(
        "/api/processes",
        params={"session": session.session_id, "limit": 5},
    )
    expected = r2.json()
    assert combined["processes"] == expected


def test_dashboard_respects_time_range(client, session, rawlog_path: Path):
    samples = session.rawlog.samples
    if len(samples) < 4:
        pytest.skip("fixture too small")
    start = samples[1].curtime
    end = samples[-2].curtime
    params = {
        "session": session.session_id,
        "from": _iso(start),
        "to": _iso(end),
    }
    r = client.get("/api/dashboard", params=params)
    body = r.json()
    assert body["samples"]["range"]["from"] == start
    assert body["samples"]["range"]["to"] == end
    for metric in ("cpu", "memory", "disk", "network"):
        assert body["charts"][metric]["range"]["from"] == start
        assert body["charts"][metric]["range"]["to"] == end


def test_dashboard_fan_out_calls_each_handler_once(client, session, monkeypatch):
    """The fan-out must delegate to the underlying handlers exactly once each."""
    calls = {
        "summary": 0,
        "samples": 0,
        "system_cpu": 0,
        "system_memory": 0,
        "system_disk": 0,
        "system_network": 0,
        "processes": 0,
    }
    import atop_web.api.routes.dashboard as dashboard_mod

    orig = dashboard_mod._gather_sections

    def counting(sess, from_epoch, to_epoch, process_limit, process_index):
        # Probe each helper from inside the route by spying on their
        # references, not by mock replacement (that would defeat the
        # point of running the real code).
        result = orig(sess, from_epoch, to_epoch, process_limit, process_index)
        for key in calls:
            if key in result["_call_trace"]:
                calls[key] += 1
        result.pop("_call_trace", None)
        return result

    monkeypatch.setattr(dashboard_mod, "_gather_sections", counting)
    r = client.get("/api/dashboard", params={"session": session.session_id})
    assert r.status_code == 200
    for key, count in calls.items():
        assert count == 1, f"{key} called {count} times"
