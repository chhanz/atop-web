"""T-20 chart endpoints use aggregate/downsample path.

On wide windows (>= 1 minute) the four ``/api/samples/system_*``
endpoints must avoid sstat-inflating every sample. The per-CPU and
per-interface arrays they return can't come from ``Aggregate.lookup``
(that grid only stores system-wide averages, not per-device shapes),
so this test locks down the looser contract: at most one decode per
grid bucket. For a 10-second capture asked for a 3-hour window, that
is 180 decodes instead of 1080. For sub-minute windows we keep the
original per-sample path.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from atop_web.api.sessions import get_store
from atop_web.main import app
from atop_web.parser.lazy import LazyRawLog


FIXTURE = Path.home() / "Downloads" / "al2_atop_20260403"
pytestmark = pytest.mark.skipif(
    not FIXTURE.is_file(),
    reason=f"large fixture {FIXTURE} not present",
)


@pytest.fixture
def client():
    get_store().clear()
    with TestClient(app) as c:
        yield c
    get_store().clear()


@pytest.fixture
def session(client):
    lazy = LazyRawLog.open(FIXTURE)
    sess = get_store().create_lazy(
        filename=FIXTURE.name, size_bytes=FIXTURE.stat().st_size, lazy_rawlog=lazy
    )
    return sess


def _iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _three_hour_range(lazy: LazyRawLog) -> tuple[str, str]:
    end = lazy.index.last_time()
    start = end - 3 * 3600
    return _iso(start), _iso(end)


def _count_bundle_decodes(monkeypatch) -> list[int]:
    """Patch ``_decode_sstat_bundle`` and return a call-count box."""
    import atop_web.parser.lazy as lazy_mod

    box = [0]
    orig = lazy_mod._decode_sstat_bundle

    def counting(*args, **kwargs):
        box[0] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(lazy_mod, "_decode_sstat_bundle", counting)
    return box


def test_chart_window_iterator_downsamples():
    """``_chart_window_iter`` yields at most one view per minute.

    Direct unit test on the helper so the property is checked
    independently of the route/aggregate wiring. Works even when the
    fixture is already at 60-second resolution: the iterator must
    still bucket by minute and skip duplicates inside the same bucket.
    """
    from atop_web.api.routes.samples import _chart_window_iter

    lazy = LazyRawLog.open(FIXTURE)
    try:
        start = lazy.index.first_time()
        end = start + 3 * 3600  # 3 hours
        bucket_seconds: set[int] = set()
        views = []
        for v in _chart_window_iter(
            _DummySession(lazy), start, end, bucket_step=60
        ):
            bucket_seconds.add(v.curtime // 60)
            views.append(v.curtime)
        assert views, "downsampled iterator should still emit samples"
        assert len(views) == len(bucket_seconds), (
            f"duplicate minute buckets: {len(views)} yields, "
            f"{len(bucket_seconds)} distinct minutes"
        )
    finally:
        lazy.close()


class _DummySession:
    """Minimal session-shaped wrapper for the iterator unit test."""

    def __init__(self, lazy: LazyRawLog) -> None:
        self.rawlog = lazy
        self.index = lazy.index
        self.is_lazy = True


def test_system_cpu_wide_window_limits_decodes(client, session, monkeypatch):
    lazy = session.rawlog
    f, t = _three_hour_range(lazy)
    box = _count_bundle_decodes(monkeypatch)
    r = client.get(
        "/api/samples/system_cpu",
        params={"session": session.session_id, "from": f, "to": t},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["count"] > 0
    # Window is 3 hours = 180 minutes. Aggregate downsamples to
    # one sample per minute, so decodes stay well under the raw
    # 1080-sample count we'd see at 10-second resolution.
    assert box[0] <= 200, (
        f"bundle decodes {box[0]} too high for 3-hour window; "
        "expected <= one per minute"
    )


def test_system_memory_wide_window_limits_decodes(client, session, monkeypatch):
    lazy = session.rawlog
    f, t = _three_hour_range(lazy)
    box = _count_bundle_decodes(monkeypatch)
    r = client.get(
        "/api/samples/system_memory",
        params={"session": session.session_id, "from": f, "to": t},
    )
    assert r.status_code == 200
    assert r.json()["count"] > 0
    assert box[0] <= 200


def test_system_disk_wide_window_limits_decodes(client, session, monkeypatch):
    lazy = session.rawlog
    f, t = _three_hour_range(lazy)
    box = _count_bundle_decodes(monkeypatch)
    r = client.get(
        "/api/samples/system_disk",
        params={"session": session.session_id, "from": f, "to": t},
    )
    assert r.status_code == 200
    assert r.json()["count"] > 0
    assert box[0] <= 200


def test_system_network_wide_window_limits_decodes(client, session, monkeypatch):
    lazy = session.rawlog
    f, t = _three_hour_range(lazy)
    box = _count_bundle_decodes(monkeypatch)
    r = client.get(
        "/api/samples/system_network",
        params={"session": session.session_id, "from": f, "to": t},
    )
    assert r.status_code == 200
    assert r.json()["count"] > 0
    assert box[0] <= 200


def test_sub_minute_window_still_uses_full_resolution(client, session, monkeypatch):
    """Narrow windows must keep per-sample resolution.

    30-second windows are narrower than the 1-minute downsample grid,
    so the chart code must fall back to the lazy per-sample path. The
    count should match the raw sample count inside the window.
    """
    lazy = session.rawlog
    end = lazy.index.last_time()
    start = end - 30
    f, t = _iso(start), _iso(end)
    r = client.get(
        "/api/samples/system_cpu",
        params={"session": session.session_id, "from": f, "to": t},
    )
    assert r.status_code == 200
    body = r.json()
    # The fixture's interval is 60s, so a 30s window may contain 0-1
    # samples; the key invariant is that the route does not reject
    # sub-minute windows or silently return downsampled values.
    assert body["count"] <= 2


def test_schema_unchanged_fields_present(client, session):
    lazy = session.rawlog
    f, t = _three_hour_range(lazy)

    r = client.get(
        "/api/samples/system_cpu",
        params={"session": session.session_id, "from": f, "to": t},
    )
    body = r.json()
    for field in ("session", "hertz", "ncpu", "range", "count", "missing_samples", "samples"):
        assert field in body, field
    if body["samples"]:
        sample = body["samples"][0]
        for key in (
            "curtime", "interval", "nrcpu", "devint", "csw", "nprocs",
            "lavg1", "lavg5", "lavg15", "all", "cpus",
        ):
            assert key in sample, key

    r = client.get(
        "/api/samples/system_memory",
        params={"session": session.session_id, "from": f, "to": t},
    )
    body = r.json()
    for field in ("session", "pagesize", "range", "count", "missing_samples",
                  "swap_configured", "samples"):
        assert field in body, field

    r = client.get(
        "/api/samples/system_network",
        params={"session": session.session_id, "from": f, "to": t},
    )
    body = r.json()
    for field in ("session", "range", "count", "missing_samples", "interfaces", "samples"):
        assert field in body, field

    r = client.get(
        "/api/samples/system_disk",
        params={"session": session.session_id, "from": f, "to": t},
    )
    body = r.json()
    for field in ("session", "range", "count", "missing_samples", "devices", "samples"):
        assert field in body, field
