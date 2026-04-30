"""API level tests for the LLM endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from atop_web.api.briefings import get_briefing_store
from atop_web.api.jobs import get_job_store
from atop_web.api.sessions import get_store
from atop_web.llm import provider as provider_mod
from atop_web.main import app


@pytest.fixture()
def client(monkeypatch):
    # Each API test starts with a fresh provider cache so env patching is
    # reflected on first access.
    provider_mod.reset_provider_cache()
    monkeypatch.setenv("LLM_PROVIDER", "none")
    get_store().clear()
    get_job_store().clear()
    get_briefing_store().clear()
    with TestClient(app) as c:
        yield c
    get_store().clear()
    get_job_store().clear()
    get_briefing_store().clear()
    provider_mod.reset_provider_cache()


def _poll_until_done(client, job_id, timeout=600.0, interval=0.1):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        res = client.get(f"/api/jobs/{job_id}")
        assert res.status_code == 200, res.text
        data = res.json()
        if data["status"] in ("done", "error"):
            return data
        time.sleep(interval)
    pytest.fail(f"job {job_id} did not finish within {timeout}s")


def _upload_and_parse(client, rawlog_bytes):
    res = client.post(
        "/api/upload",
        params={"sync": 1},
        files={
            "file": ("atop_20260427", rawlog_bytes, "application/octet-stream"),
        },
    )
    assert res.status_code == 200, res.text
    return res.json()["session"]


def _create_job_from_session(session_id):
    # Mint a fake ``done`` job pointing at the given session so we can
    # exercise the briefing endpoints without running the async path.
    job = get_job_store().create(source="upload", filename="fixture")
    get_job_store().mark_done(
        job.job_id,
        {
            "session": session_id,
            "filename": "fixture",
            "size_bytes": 0,
            "sample_count": 1,
        },
    )
    return job.job_id


def test_llm_health_none(client):
    res = client.get("/api/llm/health")
    assert res.status_code == 200
    data = res.json()
    assert data["provider"] == "none"
    assert data["ok"] is True


def test_llm_health_bedrock_without_boto3(client, monkeypatch):
    import sys

    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setitem(sys.modules, "boto3", None)
    provider_mod.reset_provider_cache()
    res = client.get("/api/llm/health")
    assert res.status_code == 200
    data = res.json()
    assert data["provider"] == "bedrock"
    assert data["ok"] is False
    assert "boto3" in data["detail"]


def test_briefing_none_provider_returns_502(client, rawlog_bytes):
    # ``none`` provider has no completion path, so posting a briefing must
    # surface a 502 with a descriptive detail.
    session_id = _upload_and_parse(client, rawlog_bytes)
    job_id = _create_job_from_session(session_id)
    res = client.post(f"/api/jobs/{job_id}/briefing")
    assert res.status_code == 502
    assert "LLM is disabled" in res.json()["detail"]


def test_briefing_get_returns_404_before_generation(client, rawlog_bytes):
    session_id = _upload_and_parse(client, rawlog_bytes)
    job_id = _create_job_from_session(session_id)
    res = client.get(f"/api/jobs/{job_id}/briefing")
    assert res.status_code == 404


def test_briefing_flow_with_stubbed_provider(
    client, monkeypatch, rawlog_bytes
):
    # Install a fake provider that returns a canned issues list.
    from atop_web.llm.provider import LLMProvider

    class _StubProvider(LLMProvider):
        name = "stub"
        model = "stub-model"

        def health(self):
            return {
                "ok": True,
                "provider": self.name,
                "model": self.model,
                "detail": "stub",
            }

        def complete_json(self, system, user, schema):
            return {
                "issues": [
                    {"title": "t1", "severity": "warning", "detail": "d1"},
                    {"title": "t2", "severity": "info", "detail": "d2"},
                ]
            }

        def stream(self, system, user, history=None):
            yield ""

    stub_instance = _StubProvider()
    monkeypatch.setattr(provider_mod, "get_provider", lambda: stub_instance)
    # The route module imports ``get_provider`` at module load; patch both.
    from atop_web.api.routes import llm as llm_route

    monkeypatch.setattr(llm_route, "get_provider", lambda: stub_instance)

    session_id = _upload_and_parse(client, rawlog_bytes)
    job_id = _create_job_from_session(session_id)

    res = client.post(f"/api/jobs/{job_id}/briefing")
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["status"] == "ok"
    assert data["provider"] == "stub"
    assert len(data["issues"]) == 2
    assert data["issues"][0]["title"] == "t1"

    # GET returns the cached briefing.
    res = client.get(f"/api/jobs/{job_id}/briefing")
    assert res.status_code == 200
    assert res.json()["issues"][0]["title"] == "t1"


def test_briefing_404_when_job_missing(client):
    res = client.post("/api/jobs/does-not-exist/briefing")
    assert res.status_code == 404


def test_briefing_409_when_job_not_done(client):
    job = get_job_store().create(source="upload", filename="x")
    res = client.post(f"/api/jobs/{job.job_id}/briefing")
    assert res.status_code == 409


# ---------------------------------------------------------------------------
# Chat stream endpoint (Phase 17)
# ---------------------------------------------------------------------------


def test_chat_stream_none_provider_returns_503(client, rawlog_bytes):
    session_id = _upload_and_parse(client, rawlog_bytes)
    job_id = _create_job_from_session(session_id)
    res = client.post(
        f"/api/jobs/{job_id}/chat/stream",
        json={"message": "hi"},
    )
    assert res.status_code == 503
    assert "disabled" in res.json()["detail"].lower()


def test_chat_stream_job_not_found(client, monkeypatch):
    from atop_web.llm.provider import LLMProvider

    class _StubStream(LLMProvider):
        name = "stubs"
        model = "s"

        def health(self):
            return {"ok": True, "provider": self.name, "model": self.model, "detail": ""}

        def complete_json(self, system, user, schema):
            raise NotImplementedError

        def stream(self, system, user, history=None):
            yield "x"

    stub = _StubStream()
    from atop_web.api.routes import llm as llm_route

    monkeypatch.setattr(llm_route, "get_provider", lambda: stub)
    res = client.post(
        "/api/jobs/does-not-exist/chat/stream",
        json={"message": "hi"},
    )
    assert res.status_code == 404


def _parse_sse(text: str) -> list[tuple[str, str]]:
    """Split SSE text into ``(event, data)`` tuples."""
    frames: list[tuple[str, str]] = []
    current_event: str | None = None
    current_data: list[str] = []
    for line in text.splitlines():
        if line == "":
            if current_event is not None:
                frames.append((current_event, "\n".join(current_data)))
            current_event = None
            current_data = []
            continue
        if line.startswith("event: "):
            current_event = line[len("event: ") :]
        elif line.startswith("data: "):
            current_data.append(line[len("data: ") :])
    if current_event is not None:
        frames.append((current_event, "\n".join(current_data)))
    return frames


def test_chat_stream_emits_sse_events(client, monkeypatch, rawlog_bytes):
    from atop_web.llm.provider import LLMProvider

    class _Provider(LLMProvider):
        name = "stub"
        model = "m"

        def health(self):
            return {"ok": True, "provider": self.name, "model": self.model, "detail": ""}

        def complete_json(self, system, user, schema):
            raise NotImplementedError

        def stream(self, system, user, history=None):
            yield "Hello"
            yield " world"

    from atop_web.api.routes import llm as llm_route
    monkeypatch.setattr(llm_route, "get_provider", lambda: _Provider())

    session_id = _upload_and_parse(client, rawlog_bytes)
    job_id = _create_job_from_session(session_id)
    res = client.post(
        f"/api/jobs/{job_id}/chat/stream",
        json={"message": "analyze"},
    )
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    frames = _parse_sse(res.text)
    events = [f[0] for f in frames]
    assert "token" in events
    assert events[-1] == "done"
    joined = "".join(
        __import__("json").loads(data)["text"]
        for evt, data in frames
        if evt == "token"
    )
    assert joined == "Hello world"


def test_chat_stream_range_hint_event(client, monkeypatch, rawlog_bytes):
    from atop_web.llm.provider import LLMProvider

    class _Provider(LLMProvider):
        name = "stub"
        model = "m"

        def health(self):
            return {"ok": True, "provider": self.name, "model": self.model, "detail": ""}

        def complete_json(self, system, user, schema):
            raise NotImplementedError

        def stream(self, system, user, history=None):
            yield "issue "
            yield '<range start="2026-04-27T14:00:00Z" end="2026-04-27T14:02:00Z" reason="cpu"/>'
            yield " spotted"

    from atop_web.api.routes import llm as llm_route
    monkeypatch.setattr(llm_route, "get_provider", lambda: _Provider())

    session_id = _upload_and_parse(client, rawlog_bytes)
    job_id = _create_job_from_session(session_id)
    res = client.post(
        f"/api/jobs/{job_id}/chat/stream",
        json={"message": "find spikes"},
    )
    assert res.status_code == 200
    frames = _parse_sse(res.text)
    import json as _json
    range_frames = [
        _json.loads(data) for evt, data in frames if evt == "range_hint"
    ]
    assert len(range_frames) == 1
    assert range_frames[0]["reason"] == "cpu"
    # The stub emits a 2 minute window; Phase 19 auto widens anything
    # narrower than ``interval_seconds * 2`` so we only check the reason
    # and the widened flag survived the sanitizer.
    assert range_frames[0].get("widened") is True
    # Tag must not leak into tokens.
    tokens = "".join(
        _json.loads(data)["text"] for evt, data in frames if evt == "token"
    )
    assert "<range" not in tokens
    assert "issue" in tokens and "spotted" in tokens


def test_chat_stream_rejects_missing_message(client, monkeypatch, rawlog_bytes):
    from atop_web.llm.provider import LLMProvider

    class _Provider(LLMProvider):
        name = "stub"
        model = "m"

        def health(self):
            return {"ok": True, "provider": self.name, "model": self.model, "detail": ""}

        def complete_json(self, system, user, schema):
            raise NotImplementedError

        def stream(self, system, user, history=None):
            yield "x"

    from atop_web.api.routes import llm as llm_route
    monkeypatch.setattr(llm_route, "get_provider", lambda: _Provider())

    session_id = _upload_and_parse(client, rawlog_bytes)
    job_id = _create_job_from_session(session_id)
    res = client.post(
        f"/api/jobs/{job_id}/chat/stream",
        json={"message": "   "},
    )
    assert res.status_code == 400
