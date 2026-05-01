"""T-10 Upload path spools large payloads to disk and parses lazily.

Before Phase 22 the upload path read the whole rawlog into a single
``bytes`` value, which meant a 266 MB rawlog ate 266 MB of Python heap
before the parser even started. This test locks down the new behaviour:

* ``run_parse_job`` accepts a filesystem path, not just ``bytes``. The
  upload route now hands it a ``Path`` (from the SpooledTemporaryFile)
  and parses straight off disk.
* The resulting session is lazy (``is_lazy=True``), so the session store
  holds the file handle + index, not a materialized ``list[Sample]``.
* The upload endpoint still returns the same 202 Accepted + job id shape
  the frontend polls on.

The 266 MB capture lives on the dev host under ``Downloads/``; these
tests skip cleanly on a clean checkout.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from atop_web.api.jobs import get_job_store
from atop_web.api.parsing import run_parse_job
from atop_web.api.sessions import get_store
from atop_web.main import app


@pytest.fixture
def client():
    get_store().clear()
    get_job_store().clear()
    with TestClient(app) as c:
        yield c
    get_store().clear()
    get_job_store().clear()


def test_run_parse_job_accepts_path(rawlog_path: Path):
    # The post-Phase-22 signature lets the caller hand over a Path. The
    # job completes and creates a lazy session without ever materializing
    # the rawlog into a Python ``bytes``.
    get_store().clear()
    get_job_store().clear()
    job = get_job_store().create(source="upload", filename="x")
    run_parse_job(
        job.job_id,
        rawlog_path,
        filename=rawlog_path.name,
        source="upload",
    )
    final = get_job_store().get(job.job_id)
    assert final is not None
    assert final.status == "done", final.error
    sess_id = final.result["session"]
    sess = get_store().require(sess_id)
    assert sess.is_lazy is True
    assert sess.index is not None


def test_upload_route_uses_spooled_file(client: TestClient, rawlog_path: Path):
    # The upload route's sync path must still go through parse_bytes-ish
    # behaviour without regressing the public job/session shape. After
    # T-10 the bytes round-trip is a SpooledTemporaryFile(max_size=0)
    # instead of a single bytes object — detectable via the session
    # being lazy.
    data = rawlog_path.read_bytes()
    res = client.post(
        "/api/upload",
        params={"sync": 1},
        files={"file": (rawlog_path.name, io.BytesIO(data), "application/octet-stream")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    sess = get_store().require(body["session"])
    assert sess.is_lazy is True


def test_upload_spools_straight_to_disk(monkeypatch):
    # ``SpooledTemporaryFile(max_size=0)`` forces an immediate rollover
    # so the payload never spends time living in Python bytes. The T-10
    # helper exposes this as ``spool_upload`` — writing through it and
    # closing it should leave the payload on disk, accessible via the
    # file's ``.name`` attribute.
    from atop_web.api.parsing import spool_upload

    def chunks():
        yield b"a" * (1 << 20)
        yield b"b" * (1 << 20)
        yield b"c" * 100

    path = spool_upload(chunks())
    try:
        assert path.exists()
        size = path.stat().st_size
        assert size == (1 << 20) * 2 + 100
        # The file content round-trips so parsers that seek + read small
        # windows (the Phase 22 lazy decoder) see the original bytes.
        with path.open("rb") as fh:
            assert fh.read(4) == b"aaaa"
    finally:
        path.unlink(missing_ok=True)


def test_tmpdir_honors_env(monkeypatch, tmp_path: Path):
    # The upload path must honor ``TMPDIR`` so deployments that put
    # /tmp on tmpfs (Docker default) can redirect the spool to the
    # persistent volume.
    from atop_web.api import parsing

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    path = parsing.spool_upload([b"hello, world"])
    try:
        assert str(path).startswith(str(tmp_path))
    finally:
        path.unlink(missing_ok=True)


def test_upload_route_preserves_job_contract(client: TestClient, rawlog_path: Path):
    # Async path: the 202 body shape is unchanged (job_id + source +
    # filename + size_bytes). Frontends depend on this.
    res = client.post(
        "/api/upload",
        files={
            "file": (
                rawlog_path.name,
                io.BytesIO(rawlog_path.read_bytes()),
                "application/octet-stream",
            )
        },
    )
    assert res.status_code == 202
    body = res.json()
    assert "job_id" in body
    assert body["source"] == "upload"
    assert body["filename"] == rawlog_path.name
    assert body["size_bytes"] > 0


def test_server_pick_parses_from_path(client: TestClient, rawlog_path: Path):
    # /api/files/parse picks a file off the server mount. The post-T-10
    # handler passes the Path straight to run_parse_job; the client-
    # observable shape is unchanged.
    import atop_web.api.routes.files as files_mod

    # Point the server picker at the directory that holds the fixture
    # file so the request resolves.
    files_mod.ATOP_LOG_DIR = str(rawlog_path.parent)
    try:
        res = client.post(
            "/api/files/parse",
            params={"sync": 1},
            json={"name": rawlog_path.name},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        sess = get_store().require(body["session"])
        assert sess.is_lazy is True
    finally:
        files_mod.ATOP_LOG_DIR = os.environ.get("ATOP_LOG_DIR", "/var/log/atop")
