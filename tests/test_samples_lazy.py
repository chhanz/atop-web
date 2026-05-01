"""T-07 /api/samples lazy parity.

Hard constraint: the response schema is **unchanged**. Eager and lazy
sessions must return byte-identical JSON on the main ``/api/samples``
endpoint and on every ``/api/samples/system_*`` variant. ``cpu_ticks``
in the response is the sum of ``utime + stime`` across processes — a
tstat-derived value — so we always drive the lazy path through the full
decode for the samples in the window. The pre-aggregate cache is a
separate read-only index that tool handlers (T-09) and future endpoints
can consult; it does not replace the row shape of this endpoint.

Tests:
* ``/api/samples`` eager vs lazy deep-equal, with and without a range.
* ``/api/samples/system_cpu`` eager vs lazy deep-equal.
* ``/api/samples/system_memory`` eager vs lazy deep-equal.
* ``/api/samples/system_disk`` eager vs lazy deep-equal.
* ``/api/samples/system_network`` eager vs lazy deep-equal.
* Session exposes an aggregate when the caller passes one at create
  time; the aggregate's ``lookup`` is callable on a session.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from atop_web.api.sessions import get_store
from atop_web.main import app
from atop_web.parser import parse_file
from atop_web.parser.aggregate import build_aggregate
from atop_web.parser.lazy import LazyRawLog


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _cleanup_store():
    yield
    get_store().clear()


def _make_pair(path: Path):
    """Create an eager and a lazy session on the same rawlog."""
    eager = parse_file(path)
    store = get_store()
    eager_sess = store.create(
        filename="f", size_bytes=path.stat().st_size, rawlog=eager
    )
    lazy = LazyRawLog.open(path)
    lazy_sess = store.create_lazy(
        filename="f", size_bytes=path.stat().st_size, lazy_rawlog=lazy
    )
    return eager_sess.session_id, lazy_sess.session_id


def _strip_session(d: dict) -> dict:
    d = dict(d)
    d.pop("session", None)
    return d


def test_samples_root_eager_lazy_parity(rawlog_path: Path, client: TestClient):
    eager_id, lazy_id = _make_pair(rawlog_path)
    r_e = client.get("/api/samples", params={"session": eager_id}).json()
    r_l = client.get("/api/samples", params={"session": lazy_id}).json()
    assert _strip_session(r_e) == _strip_session(r_l)


def test_samples_root_with_range(rawlog_path: Path, client: TestClient):
    eager_id, lazy_id = _make_pair(rawlog_path)
    eager = parse_file(rawlog_path)
    if len(eager.samples) < 4:
        pytest.skip("need at least 4 samples")
    start = eager.samples[1].curtime
    end = eager.samples[-2].curtime
    from datetime import datetime, timezone

    fmt = lambda t: datetime.fromtimestamp(t, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    params_common = {"from": fmt(start), "to": fmt(end)}
    r_e = client.get(
        "/api/samples", params={"session": eager_id, **params_common}
    ).json()
    r_l = client.get(
        "/api/samples", params={"session": lazy_id, **params_common}
    ).json()
    assert _strip_session(r_e) == _strip_session(r_l)


def test_samples_system_cpu_parity(rawlog_path: Path, client: TestClient):
    eager_id, lazy_id = _make_pair(rawlog_path)
    r_e = client.get("/api/samples/system_cpu", params={"session": eager_id}).json()
    r_l = client.get("/api/samples/system_cpu", params={"session": lazy_id}).json()
    assert _strip_session(r_e) == _strip_session(r_l)


def test_samples_system_memory_parity(rawlog_path: Path, client: TestClient):
    eager_id, lazy_id = _make_pair(rawlog_path)
    r_e = client.get(
        "/api/samples/system_memory", params={"session": eager_id}
    ).json()
    r_l = client.get(
        "/api/samples/system_memory", params={"session": lazy_id}
    ).json()
    assert _strip_session(r_e) == _strip_session(r_l)


def test_samples_system_disk_parity(rawlog_path: Path, client: TestClient):
    eager_id, lazy_id = _make_pair(rawlog_path)
    r_e = client.get(
        "/api/samples/system_disk", params={"session": eager_id}
    ).json()
    r_l = client.get(
        "/api/samples/system_disk", params={"session": lazy_id}
    ).json()
    assert _strip_session(r_e) == _strip_session(r_l)


def test_samples_system_network_parity(rawlog_path: Path, client: TestClient):
    eager_id, lazy_id = _make_pair(rawlog_path)
    r_e = client.get(
        "/api/samples/system_network", params={"session": eager_id}
    ).json()
    r_l = client.get(
        "/api/samples/system_network", params={"session": lazy_id}
    ).json()
    assert _strip_session(r_e) == _strip_session(r_l)


def test_session_exposes_aggregate_when_provided(rawlog_path: Path):
    # The aggregate is a separate read-only index we carry alongside lazy
    # sessions so later endpoints / tool handlers can answer wide windows
    # without re-walking the rawlog. The samples endpoint does not use
    # it — tstat-dependent fields cannot come from a downsampled grid —
    # but T-07 locks down that sessions plumb it end-to-end.
    lazy = LazyRawLog.open(rawlog_path)
    agg = build_aggregate(lazy)
    sess = get_store().create_lazy(
        filename="f", size_bytes=1, lazy_rawlog=lazy, aggregate=agg
    )
    assert sess.aggregate is agg
    if lazy.index.first_time() is None:
        pytest.skip("empty fixture")
    start = lazy.index.first_time()
    end = lazy.index.last_time()
    result = agg.lookup("cpu_busy", "1m", start, end)
    assert result.hit is True
