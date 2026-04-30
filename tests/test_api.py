"""API tests via FastAPI TestClient."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from atop_web.api.jobs import get_job_store
from atop_web.api.sessions import get_store
from atop_web.main import app, create_app


@pytest.fixture()
def client():
    get_store().clear()
    get_job_store().clear()
    with TestClient(app) as c:
        yield c
    get_store().clear()
    get_job_store().clear()


def _poll_until_done(
    client: TestClient, job_id: str, timeout: float = 600.0, interval: float = 0.1
) -> dict:
    """Poll ``/api/jobs/{job_id}`` until the job finishes or times out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = client.get(f"/api/jobs/{job_id}")
        assert res.status_code == 200, res.text
        data = res.json()
        if data["status"] in ("done", "error"):
            return data
        time.sleep(interval)
    pytest.fail(f"job {job_id} did not finish within {timeout}s")


def test_healthz(client: TestClient):
    res = client.get("/healthz")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "log_dir" in data


def test_index_served(client: TestClient):
    res = client.get("/")
    assert res.status_code == 200
    assert b"atop-web" in res.content


def test_upload_requires_file(client: TestClient):
    res = client.post("/api/upload")
    assert res.status_code == 422


def test_upload_rejects_empty(client: TestClient):
    res = client.post("/api/upload", files={"file": ("empty", b"", "application/octet-stream")})
    assert res.status_code == 400


def test_upload_rejects_bad_magic(client: TestClient):
    # In async mode a structurally invalid file finishes as a failed job.
    res = client.post(
        "/api/upload",
        files={"file": ("bad", b"\x00" * 1024, "application/octet-stream")},
    )
    assert res.status_code == 202
    job = _poll_until_done(client, res.json()["job_id"])
    assert job["status"] == "error"
    assert job["error"]


def test_full_flow(client: TestClient, rawlog_bytes: bytes):
    res = client.post(
        "/api/upload",
        params={"sync": 1},
        files={
            "file": ("atop_20260427", rawlog_bytes, "application/octet-stream"),
        },
    )
    assert res.status_code == 200, res.text
    upload_data = res.json()
    session = upload_data["session"]
    assert upload_data["sample_count"] > 0
    assert upload_data["hostname"]
    assert upload_data["kernel"]

    res = client.get("/api/summary", params={"session": session})
    assert res.status_code == 200
    summary = res.json()
    assert summary["sample_count"] == upload_data["sample_count"]
    assert summary["system"]["hostname"] == upload_data["hostname"]
    assert summary["rawlog"]["rawheadlen"] == 480
    assert summary["rawlog"]["rawreclen"] == 96
    assert summary["rawlog"]["tstatlen"] == 968
    assert summary["time_range"]["duration_seconds"] >= 0

    res = client.get("/api/samples", params={"session": session})
    assert res.status_code == 200
    samples = res.json()
    assert samples["count"] == upload_data["sample_count"]
    assert len(samples["timeline"]) == samples["count"]
    assert len(samples["cpu"]["ticks"]) == samples["count"]
    assert len(samples["mem"]["rss_kb"]) == samples["count"]
    assert len(samples["dsk"]["read_sectors"]) == samples["count"]
    assert len(samples["net"]["tcp_sent"]) == samples["count"]

    res = client.get(
        "/api/processes",
        params={"session": session, "index": 0, "sort_by": "cpu", "limit": 50},
    )
    assert res.status_code == 200
    proc = res.json()
    assert proc["count"] > 0
    assert proc["sort_by"] == "cpu"
    ticks = [p["cpu_ticks"] for p in proc["processes"]]
    assert ticks == sorted(ticks, reverse=True)

    res = client.get(
        "/api/processes",
        params={"session": session, "sort_by": "rmem", "limit": 20},
    )
    assert res.status_code == 200
    rmems = [p["rmem_kb"] for p in res.json()["processes"]]
    assert rmems == sorted(rmems, reverse=True)


def test_session_not_found(client: TestClient):
    res = client.get("/api/summary", params={"session": "nope"})
    assert res.status_code == 404


# Reverse proxy regression tests.
#
# The application is expected to work in two deployment modes:
#
# 1. Direct root deployment, ``ATOP_ROOT_PATH`` unset. The HTML base is "./"
#    so relative asset and API URLs resolve under whatever origin serves the
#    page.
#
# 2. Behind a reverse proxy that strips an external prefix (for example
#    Traefik stripprefix, nginx proxy_pass with trailing slash, Caddy
#    handle_path). The prefix is communicated to the app via
#    ``ATOP_ROOT_PATH`` so that the injected ``<base href>`` points at the
#    external prefix; internal static and API routes still live at ``/...``
#    and must keep working.


@pytest.fixture()
def _clean_store():
    get_store().clear()
    yield
    get_store().clear()


def test_root_deployment_base_href_and_static(_clean_store):
    app_no_prefix = create_app(root_path="")
    with TestClient(app_no_prefix) as c:
        res = c.get("/")
        assert res.status_code == 200
        assert '<base href="./" />' in res.text
        assert 'href="static/style.css"' in res.text
        assert 'src="static/app.js"' in res.text

        css = c.get("/static/style.css")
        assert css.status_code == 200


def test_subpath_deployment_base_href_and_static(_clean_store):
    app_with_prefix = create_app(root_path="/atop")
    with TestClient(app_with_prefix) as c:
        res = c.get("/")
        assert res.status_code == 200
        assert '<base href="/atop/" />' in res.text

        # After prefix strip by a reverse proxy, internal routes still start
        # at "/", so the static mount must answer at /static/* regardless of
        # the configured external prefix.
        css = c.get("/static/style.css")
        assert css.status_code == 200
        js = c.get("/static/app.js")
        assert js.status_code == 200


def test_subpath_deployment_api_still_at_slash_api(_clean_store):
    app_with_prefix = create_app(root_path="/atop")
    with TestClient(app_with_prefix) as c:
        res = c.get("/api/files")
        assert res.status_code == 200
        data = res.json()
        assert "log_dir" in data
        assert "files" in data


# Server side file browser tests. The ATOP_LOG_DIR that the files router
# reads is captured at import time, so these tests monkeypatch the module
# attribute directly instead of the environment variable.


@pytest.fixture()
def tmp_log_dir(tmp_path, monkeypatch, _clean_store):
    from atop_web.api.routes import files as files_module

    monkeypatch.setattr(files_module, "ATOP_LOG_DIR", str(tmp_path))
    return tmp_path


def test_api_files_lists_mounted_files(tmp_log_dir, client: TestClient):
    (tmp_log_dir / "atop_20260427").write_bytes(b"\x00" * 10)
    (tmp_log_dir / "atop_20260428").write_bytes(b"\x00" * 20)
    # Files that do not match the atop rawlog naming should be ignored.
    (tmp_log_dir / "random.log").write_text("nope")
    (tmp_log_dir / ".hidden").write_text("nope")

    res = client.get("/api/files")
    assert res.status_code == 200
    data = res.json()
    assert data["enabled"] is True
    assert data["log_dir"] == str(tmp_log_dir)

    names = [f["name"] for f in data["files"]]
    assert set(names) == {"atop_20260427", "atop_20260428"}

    # Each entry carries the required metadata.
    by_name = {f["name"]: f for f in data["files"]}
    assert by_name["atop_20260427"]["date_guess"] == "2026-04-27"
    assert by_name["atop_20260428"]["date_guess"] == "2026-04-28"
    for f in data["files"]:
        assert isinstance(f["size"], int)
        assert f["size"] > 0
        assert isinstance(f["mtime"], str) and "T" in f["mtime"]


def test_api_files_empty_dir(tmp_log_dir, client: TestClient):
    res = client.get("/api/files")
    assert res.status_code == 200
    data = res.json()
    assert data["enabled"] is True
    assert data["files"] == []
    assert data["log_dir"] == str(tmp_log_dir)


def test_api_files_parse_traversal_blocked(tmp_log_dir, client: TestClient):
    (tmp_log_dir / "atop_20260427").write_bytes(b"\x00" * 10)

    for bad in ["../etc/passwd", "sub/atop_20260427", "..", "./atop_20260427"]:
        res = client.post("/api/files/parse", json={"name": bad})
        assert res.status_code in (400, 422), (bad, res.status_code, res.text)


def test_api_files_parse_unknown_name(tmp_log_dir, client: TestClient):
    res = client.post("/api/files/parse", json={"name": "atop_20991231"})
    assert res.status_code == 404


def test_api_files_parse_real_rawlog(tmp_log_dir, client: TestClient, rawlog_bytes: bytes):
    target = tmp_log_dir / "atop_20260427"
    target.write_bytes(rawlog_bytes)

    res = client.post(
        "/api/files/parse", params={"sync": 1}, json={"name": "atop_20260427"}
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["sample_count"] > 0
    assert data["hostname"]
    assert data["source"] == "server"

    # Follow up session calls must work as if the file had been uploaded.
    session = data["session"]
    res = client.get("/api/summary", params={"session": session})
    assert res.status_code == 200
    assert res.json()["sample_count"] == data["sample_count"]


def test_api_files_disabled_when_dir_missing(monkeypatch, _clean_store, client: TestClient):
    from atop_web.api.routes import files as files_module

    monkeypatch.setattr(files_module, "ATOP_LOG_DIR", "/definitely/not/a/real/path")

    res = client.get("/api/files")
    assert res.status_code == 200
    data = res.json()
    assert data["enabled"] is False
    assert data["files"] == []

    res = client.post("/api/files/parse", json={"name": "atop_20260427"})
    assert res.status_code == 404


# Phase 2-A: background job flow.


def test_upload_returns_job_and_completes(client: TestClient, rawlog_bytes: bytes):
    res = client.post(
        "/api/upload",
        files={"file": ("atop_20260427", rawlog_bytes, "application/octet-stream")},
    )
    assert res.status_code == 202, res.text
    body = res.json()
    assert body["source"] == "upload"
    job_id = body["job_id"]
    assert job_id

    job = _poll_until_done(client, job_id)
    assert job["status"] == "done", job
    assert job["stage"] == "done"
    assert job["progress"] == 100

    result = job["result"]
    assert result["session"]
    assert result["sample_count"] > 0
    assert result["source"] == "upload"

    res = client.get("/api/summary", params={"session": result["session"]})
    assert res.status_code == 200
    assert res.json()["sample_count"] == result["sample_count"]


def test_files_parse_returns_job_and_completes(
    tmp_log_dir, client: TestClient, rawlog_bytes: bytes
):
    (tmp_log_dir / "atop_20260427").write_bytes(rawlog_bytes)

    res = client.post("/api/files/parse", json={"name": "atop_20260427"})
    assert res.status_code == 202, res.text
    body = res.json()
    assert body["source"] == "server"
    assert body["filename"] == "atop_20260427"
    job_id = body["job_id"]
    assert job_id

    job = _poll_until_done(client, job_id)
    assert job["status"] == "done", job
    assert job["result"]["source"] == "server"
    assert job["result"]["session"]


def test_job_not_found(client: TestClient):
    res = client.get("/api/jobs/does-not-exist")
    assert res.status_code == 404


def test_job_response_includes_stage_label_and_detail_keys(
    client: TestClient, rawlog_bytes: bytes
):
    res = client.post(
        "/api/upload",
        files={"file": ("atop_20260427", rawlog_bytes, "application/octet-stream")},
    )
    assert res.status_code == 202, res.text
    job_id = res.json()["job_id"]

    # Poll once quickly; the exact stage is racy but the schema must carry
    # both the ``stage_label`` (English) and the optional ``detail`` field
    # even before the job is done.
    res = client.get(f"/api/jobs/{job_id}")
    assert res.status_code == 200
    snap = res.json()
    for key in ("stage", "stage_label", "progress", "detail"):
        assert key in snap, key

    # Drain the job so the background thread does not leak between tests.
    job = _poll_until_done(client, job_id)
    assert job["status"] == "done"
    assert job["stage_label"]  # final label should be set.
    assert job["detail"] is None  # detail cleared on completion.


def test_job_error_reported(client: TestClient):
    res = client.post(
        "/api/upload",
        files={"file": ("bogus", b"not a rawlog" + b"\x00" * 512, "application/octet-stream")},
    )
    assert res.status_code == 202
    job = _poll_until_done(client, res.json()["job_id"])
    assert job["status"] == "error"
    assert job["error"]
    assert job["result"] is None


# Phase 2-B: time range filtering on samples and processes.


@pytest.fixture()
def loaded_session(client: TestClient, rawlog_bytes: bytes) -> dict:
    res = client.post(
        "/api/upload",
        params={"sync": 1},
        files={"file": ("atop_20260427", rawlog_bytes, "application/octet-stream")},
    )
    assert res.status_code == 200, res.text
    return res.json()


def test_samples_filtered_by_from_to(client: TestClient, loaded_session: dict):
    session = loaded_session["session"]

    res = client.get("/api/samples", params={"session": session})
    assert res.status_code == 200
    full = res.json()
    assert full["count"] == full["total"] == loaded_session["sample_count"]
    assert full["count"] > 2

    # Pick an inner window using the timeline returned by the unfiltered
    # response and check that the filtered response only includes those
    # samples.
    timeline = full["timeline"]
    lo_epoch = timeline[1]
    hi_epoch = timeline[-2]
    from datetime import datetime, timezone

    lo_iso = datetime.fromtimestamp(lo_epoch, tz=timezone.utc).isoformat()
    hi_iso = datetime.fromtimestamp(hi_epoch, tz=timezone.utc).isoformat()

    res = client.get(
        "/api/samples",
        params={"session": session, "from": lo_iso, "to": hi_iso},
    )
    assert res.status_code == 200
    filtered = res.json()
    assert filtered["count"] == timeline.index(hi_epoch) - timeline.index(lo_epoch) + 1
    assert filtered["count"] < full["count"]
    assert filtered["total"] == full["total"]
    assert filtered["timeline"][0] >= lo_epoch
    assert filtered["timeline"][-1] <= hi_epoch
    assert filtered["range"]["from"] == lo_epoch
    assert filtered["range"]["to"] == hi_epoch

    # Each parallel array must share the filtered length.
    for key in ("cpu", "mem", "dsk", "net"):
        inner = filtered[key]
        for series in inner.values():
            assert len(series) == filtered["count"]


def test_samples_invalid_from(client: TestClient, loaded_session: dict):
    session = loaded_session["session"]
    res = client.get(
        "/api/samples",
        params={"session": session, "from": "not-a-date"},
    )
    assert res.status_code == 400
    assert "from" in res.json()["detail"]


def test_processes_filtered_by_from_to(client: TestClient, loaded_session: dict):
    session = loaded_session["session"]

    # Build a range that excludes the very first sample and keeps the last.
    full = client.get("/api/samples", params={"session": session}).json()
    timeline = full["timeline"]
    from datetime import datetime, timezone

    lo_iso = datetime.fromtimestamp(timeline[1], tz=timezone.utc).isoformat()
    hi_iso = datetime.fromtimestamp(timeline[-1], tz=timezone.utc).isoformat()

    # Without range, the default sample should be somewhere valid.
    res = client.get("/api/processes", params={"session": session, "sort_by": "cpu"})
    assert res.status_code == 200
    default_all = res.json()
    assert default_all["curtime"] in timeline

    # With range and index=0, the selected sample is the first sample of the
    # filtered subset, i.e. timeline[1], not timeline[0].
    res = client.get(
        "/api/processes",
        params={"session": session, "from": lo_iso, "to": hi_iso, "index": 0, "sort_by": "cpu"},
    )
    assert res.status_code == 200
    ranged = res.json()
    assert ranged["curtime"] == timeline[1]

    # Index beyond the filtered subset should 404, proving the filter bit.
    filtered_count = len([t for t in timeline if timeline[1] <= t <= timeline[-1]])
    res = client.get(
        "/api/processes",
        params={
            "session": session,
            "from": lo_iso,
            "to": hi_iso,
            "index": filtered_count,
            "sort_by": "cpu",
        },
    )
    assert res.status_code == 404


def test_processes_invalid_to(client: TestClient, loaded_session: dict):
    session = loaded_session["session"]
    res = client.get(
        "/api/processes",
        params={"session": session, "to": "garbage"},
    )
    assert res.status_code == 400


# Phase 4-A: extended sort.


def test_processes_sort_by_pid_asc(client: TestClient, loaded_session: dict):
    session = loaded_session["session"]
    res = client.get(
        "/api/processes",
        params={"session": session, "sort_by": "pid", "order": "asc", "limit": 50},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["sort_by"] == "pid"
    assert data["order"] == "asc"
    pids = [p["pid"] for p in data["processes"]]
    assert pids == sorted(pids)
    assert len(pids) > 1


def test_processes_sort_by_name_asc_desc(client: TestClient, loaded_session: dict):
    session = loaded_session["session"]

    res = client.get(
        "/api/processes",
        params={"session": session, "sort_by": "name", "order": "asc", "limit": 100},
    )
    assert res.status_code == 200
    names_asc = [p["name"].lower() for p in res.json()["processes"]]
    assert names_asc == sorted(names_asc)

    res = client.get(
        "/api/processes",
        params={"session": session, "sort_by": "name", "order": "desc", "limit": 100},
    )
    assert res.status_code == 200
    names_desc = [p["name"].lower() for p in res.json()["processes"]]
    assert names_desc == sorted(names_desc, reverse=True)


def test_processes_sort_order_invalid(client: TestClient, loaded_session: dict):
    session = loaded_session["session"]
    res = client.get(
        "/api/processes",
        params={"session": session, "sort_by": "pid", "order": "sideways"},
    )
    assert res.status_code == 422


def test_processes_sort_by_invalid_column(client: TestClient, loaded_session: dict):
    session = loaded_session["session"]
    res = client.get(
        "/api/processes",
        params={"session": session, "sort_by": "bogus"},
    )
    assert res.status_code == 400


# Phase 4-B: derived units + meta.


def test_processes_includes_cpu_pct_and_mb_fields(
    client: TestClient, loaded_session: dict
):
    session = loaded_session["session"]
    res = client.get(
        "/api/processes",
        params={"session": session, "sort_by": "cpu", "limit": 5},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["processes"]
    for p in data["processes"]:
        assert "cpu_pct" in p
        assert "rmem_mb" in p and isinstance(p["rmem_mb"], (int, float))
        assert "vmem_mb" in p and isinstance(p["vmem_mb"], (int, float))
        assert "dsk_read_mb" in p
        assert "dsk_write_mb" in p
        # raw values must still be present
        assert "cpu_ticks" in p
        assert "rmem_kb" in p
        assert "vmem_kb" in p
        assert "dsk_read_sectors" in p
        # Sanity check on the conversion itself.
        assert abs(p["rmem_mb"] - p["rmem_kb"] / 1024.0) < 1e-3


def test_processes_response_has_meta(client: TestClient, loaded_session: dict):
    session = loaded_session["session"]
    res = client.get(
        "/api/processes",
        params={"session": session, "sort_by": "cpu", "limit": 1},
    )
    assert res.status_code == 200
    data = res.json()
    assert "meta" in data
    meta = data["meta"]
    assert meta["hertz"] and meta["hertz"] > 0
    assert meta["interval_sec"] == data["interval"]
    # ncpu may be None if the sstat first bytes are out of range, but on a
    # well formed rawlog this must resolve to a positive integer.
    assert meta["ncpu"] is None or meta["ncpu"] > 0


def test_system_memory_endpoint_basic(client: TestClient, loaded_session: dict):
    session = loaded_session["session"]
    res = client.get("/api/samples/system_memory", params={"session": session})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["pagesize"] > 0
    assert data["count"] > 0
    assert isinstance(data["swap_configured"], bool)
    first = data["samples"][0]
    for key in (
        "curtime",
        "physmem",
        "freemem",
        "buffermem",
        "slabmem",
        "cachemem",
        "availablemem",
        "totswap",
        "freeswap",
        "swapcached",
    ):
        assert key in first, key
    assert first["physmem"] > 0


def test_system_memory_endpoint_filters_by_time(client: TestClient, loaded_session: dict):
    session = loaded_session["session"]
    unfiltered = client.get("/api/samples/system_memory", params={"session": session}).json()
    if unfiltered["count"] < 3:
        return
    from datetime import datetime, timezone

    first_ts = unfiltered["samples"][1]["curtime"]
    last_ts = unfiltered["samples"][-2]["curtime"]
    lo = datetime.fromtimestamp(first_ts, tz=timezone.utc).isoformat()
    hi = datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat()

    filtered = client.get(
        "/api/samples/system_memory",
        params={"session": session, "from": lo, "to": hi},
    ).json()
    assert filtered["count"] < unfiltered["count"]
    assert filtered["samples"][0]["curtime"] >= first_ts
    assert filtered["samples"][-1]["curtime"] <= last_ts


def test_system_cpu_endpoint_basic(client: TestClient, loaded_session: dict):
    session = loaded_session["session"]
    res = client.get("/api/samples/system_cpu", params={"session": session})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["hertz"] > 0
    assert data["ncpu"] > 0
    assert data["count"] > 0
    first = data["samples"][0]
    for key in ("curtime", "interval", "nrcpu", "all", "cpus"):
        assert key in first, key
    assert set(first["all"].keys()) >= {
        "cpunr",
        "stime",
        "utime",
        "ntime",
        "itime",
        "wtime",
        "Itime",
        "Stime",
        "steal",
        "guest",
    }
    # Per CPU rows: len must match nrcpu and utime sum equals all.utime.
    assert len(first["cpus"]) == first["nrcpu"]
    sum_u = sum(c["utime"] for c in first["cpus"])
    assert abs(sum_u - first["all"]["utime"]) <= 1


def test_system_cpu_endpoint_filters_by_time(client: TestClient, loaded_session: dict):
    session = loaded_session["session"]
    unfiltered = client.get("/api/samples/system_cpu", params={"session": session}).json()
    if unfiltered["count"] < 3:
        return
    from datetime import datetime, timezone

    lo_ts = unfiltered["samples"][1]["curtime"]
    hi_ts = unfiltered["samples"][-2]["curtime"]
    lo = datetime.fromtimestamp(lo_ts, tz=timezone.utc).isoformat()
    hi = datetime.fromtimestamp(hi_ts, tz=timezone.utc).isoformat()
    filtered = client.get(
        "/api/samples/system_cpu",
        params={"session": session, "from": lo, "to": hi},
    ).json()
    assert filtered["count"] < unfiltered["count"]
    assert filtered["samples"][0]["curtime"] >= lo_ts
    assert filtered["samples"][-1]["curtime"] <= hi_ts


def test_system_disk_endpoint_basic(client: TestClient, loaded_session: dict):
    session = loaded_session["session"]
    res = client.get("/api/samples/system_disk", params={"session": session})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["count"] > 0
    assert "devices" in data
    assert data["devices"]["disks"], "expected at least one physical disk name"
    first = data["samples"][0]
    for key in ("curtime", "interval", "disks", "mdds", "lvms"):
        assert key in first, key
    first_dev = first["disks"][0]
    for key in (
        "name",
        "nread",
        "nrsect",
        "nwrite",
        "nwsect",
        "io_ms",
        "avque",
        "ndisc",
        "ndsect",
        "inflight",
        "kind",
    ):
        assert key in first_dev, key
    assert first_dev["kind"] == "dsk"
    assert first_dev["nread"] > 0


def test_system_network_endpoint_basic(client: TestClient, loaded_session: dict):
    session = loaded_session["session"]
    res = client.get("/api/samples/system_network", params={"session": session})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["count"] > 0
    assert "interfaces" in data
    assert "lo" in data["interfaces"]
    first = data["samples"][0]
    for key in ("curtime", "interval", "nrintf", "interfaces"):
        assert key in first, key
    assert first["nrintf"] >= 1
    assert any(i["name"] == "lo" for i in first["interfaces"])
    lo = next(i for i in first["interfaces"] if i["name"] == "lo")
    for key in (
        "name",
        "type",
        "speed_mbps",
        "duplex",
        "rbyte",
        "rpack",
        "rerrs",
        "rdrop",
        "sbyte",
        "spack",
        "serrs",
        "sdrop",
    ):
        assert key in lo, key
    assert lo["type"] == "v"


def test_system_network_endpoint_filters_by_time(
    client: TestClient, loaded_session: dict
):
    session = loaded_session["session"]
    unfiltered = client.get(
        "/api/samples/system_network", params={"session": session}
    ).json()
    if unfiltered["count"] < 3:
        return
    from datetime import datetime, timezone

    lo_ts = unfiltered["samples"][1]["curtime"]
    hi_ts = unfiltered["samples"][-2]["curtime"]
    lo = datetime.fromtimestamp(lo_ts, tz=timezone.utc).isoformat()
    hi = datetime.fromtimestamp(hi_ts, tz=timezone.utc).isoformat()
    filtered = client.get(
        "/api/samples/system_network",
        params={"session": session, "from": lo, "to": hi},
    ).json()
    assert filtered["count"] < unfiltered["count"]
    assert filtered["samples"][0]["curtime"] >= lo_ts
    assert filtered["samples"][-1]["curtime"] <= hi_ts


def test_system_disk_endpoint_filters_by_time(
    client: TestClient, loaded_session: dict
):
    session = loaded_session["session"]
    unfiltered = client.get(
        "/api/samples/system_disk", params={"session": session}
    ).json()
    if unfiltered["count"] < 3:
        return
    from datetime import datetime, timezone

    lo_ts = unfiltered["samples"][1]["curtime"]
    hi_ts = unfiltered["samples"][-2]["curtime"]
    lo = datetime.fromtimestamp(lo_ts, tz=timezone.utc).isoformat()
    hi = datetime.fromtimestamp(hi_ts, tz=timezone.utc).isoformat()
    filtered = client.get(
        "/api/samples/system_disk",
        params={"session": session, "from": lo, "to": hi},
    ).json()
    assert filtered["count"] < unfiltered["count"]
    assert filtered["samples"][0]["curtime"] >= lo_ts
    assert filtered["samples"][-1]["curtime"] <= hi_ts


def test_system_memory_swap_not_configured_when_all_zero(
    client: TestClient, loaded_session: dict
):
    session = loaded_session["session"]
    data = client.get("/api/samples/system_memory", params={"session": session}).json()
    # The sample rawlog has no swap configured; all totswap should be 0.
    swap_totals = [s["totswap"] for s in data["samples"]]
    if all(v == 0 for v in swap_totals):
        assert data["swap_configured"] is False


def test_processes_cpu_pct_zero_when_no_interval(monkeypatch, client: TestClient):
    # Build a synthetic rawlog structure where the sample interval is 0 so
    # the route hits the "cannot compute %" branch without depending on the
    # real rawlog file.
    from atop_web.parser.reader import Process, RawLog, Header, Sample
    from atop_web.api.sessions import get_store as _session_store

    header = Header(
        magic=0xFEEDBEEF,
        aversion_raw=0x820C,
        aversion="2.12",
        rawheadlen=480,
        rawreclen=96,
        hertz=100,
        pidwidth=7,
        sstatlen=0,
        tstatlen=968,
        pagesize=4096,
        supportflags=0,
        osrel=0,
        osvers=0,
        ossub=0,
        cstatlen=0,
        sysname="Linux",
        nodename="synthetic",
        release="synthetic",
        version="synthetic",
        machine="x86_64",
        domainname="",
    )
    proc = Process(
        pid=42, tgid=42, ppid=1, name="fake", state="S", cmdline="fake",
        nthr=1, isproc=True, utime=1000, stime=500, rmem_kb=2048, vmem_kb=4096,
        rio=10, wio=5, rsz=20, wsz=10, tcpsnd=1, tcprcv=2, udpsnd=3, udprcv=4,
    )
    sample = Sample(
        curtime=1777200000, interval=0, ndeviat=1, nactproc=1, ntask=1,
        totproc=1, totrun=0, totslpi=1, totslpu=0, totzomb=0, nrcpu=4,
        processes=[proc],
    )
    rawlog = RawLog(header=header, samples=[sample])

    sess = _session_store().create(filename="synthetic", size_bytes=0, rawlog=rawlog)

    res = client.get(
        "/api/processes",
        params={"session": sess.session_id, "sort_by": "cpu", "limit": 1},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["processes"][0]["cpu_pct"] is None
    assert data["meta"]["interval_sec"] == 0
