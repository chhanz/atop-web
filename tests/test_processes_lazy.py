"""T-08 /api/processes lazy parity.

For a given (session, time, sort_by, order) request, eager and lazy
sessions must yield byte-identical JSON. The process list depends on
per-process tstat decode, which is the most expensive piece of the
lazy path — we want this test to catch any regression where the LRU
or decode plumbing drops a column or re-orders rows.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from atop_web.api.sessions import get_store
from atop_web.main import app
from atop_web.parser import parse_file
from atop_web.parser.lazy import LazyRawLog


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _cleanup_store():
    yield
    get_store().clear()


def _make_pair(path: Path):
    eager = parse_file(path)
    store = get_store()
    eager_sess = store.create(
        filename="f", size_bytes=path.stat().st_size, rawlog=eager
    )
    lazy = LazyRawLog.open(path)
    lazy_sess = store.create_lazy(
        filename="f", size_bytes=path.stat().st_size, lazy_rawlog=lazy
    )
    return eager_sess.session_id, lazy_sess.session_id, eager


def _strip(d: dict) -> dict:
    d = dict(d)
    d.pop("session", None)
    return d


def test_processes_default_parity(rawlog_path: Path, client: TestClient):
    eager_id, lazy_id, _ = _make_pair(rawlog_path)
    r_e = client.get("/api/processes", params={"session": eager_id}).json()
    r_l = client.get("/api/processes", params={"session": lazy_id}).json()
    assert _strip(r_e) == _strip(r_l)


@pytest.mark.parametrize(
    "sort_by,order",
    [
        ("cpu", "desc"),
        ("rmem", "desc"),
        ("vmem", "asc"),
        ("pid", "asc"),
        ("name", "asc"),
        ("dsk", "desc"),
        ("net", "desc"),
        ("nthr", "desc"),
        ("state", "asc"),
    ],
)
def test_processes_sort_variants(
    rawlog_path: Path, client: TestClient, sort_by, order
):
    eager_id, lazy_id, _ = _make_pair(rawlog_path)
    r_e = client.get(
        "/api/processes",
        params={"session": eager_id, "sort_by": sort_by, "order": order},
    ).json()
    r_l = client.get(
        "/api/processes",
        params={"session": lazy_id, "sort_by": sort_by, "order": order},
    ).json()
    assert _strip(r_e) == _strip(r_l)


def test_processes_index_param_parity(rawlog_path: Path, client: TestClient):
    eager_id, lazy_id, eager = _make_pair(rawlog_path)
    if len(eager.samples) < 2:
        pytest.skip("need multiple samples")
    params = {"index": 1, "limit": 50}
    r_e = client.get("/api/processes", params={"session": eager_id, **params}).json()
    r_l = client.get("/api/processes", params={"session": lazy_id, **params}).json()
    assert _strip(r_e) == _strip(r_l)


def test_processes_time_param_parity(rawlog_path: Path, client: TestClient):
    eager_id, lazy_id, eager = _make_pair(rawlog_path)
    if len(eager.samples) < 2:
        pytest.skip("need multiple samples")
    target = eager.samples[len(eager.samples) // 2].curtime
    params = {"time": target, "limit": 10}
    r_e = client.get("/api/processes", params={"session": eager_id, **params}).json()
    r_l = client.get("/api/processes", params={"session": lazy_id, **params}).json()
    assert _strip(r_e) == _strip(r_l)


def test_processes_range_parity(rawlog_path: Path, client: TestClient):
    eager_id, lazy_id, eager = _make_pair(rawlog_path)
    if len(eager.samples) < 4:
        pytest.skip("need multiple samples")
    from datetime import datetime, timezone

    fmt = lambda t: datetime.fromtimestamp(t, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    lo = fmt(eager.samples[1].curtime)
    hi = fmt(eager.samples[-2].curtime)
    params = {"from": lo, "to": hi}
    r_e = client.get("/api/processes", params={"session": eager_id, **params}).json()
    r_l = client.get("/api/processes", params={"session": lazy_id, **params}).json()
    assert _strip(r_e) == _strip(r_l)
