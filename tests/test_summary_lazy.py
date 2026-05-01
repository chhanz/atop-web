"""T-06 /api/summary lazy parity.

Both session paths must return byte-identical JSON — no field added,
removed, or renamed. The test fabricates an eager and a lazy session
on the same rawlog and asserts the two summary responses are deep-equal.
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


def test_summary_lazy_matches_eager(rawlog_path: Path, client: TestClient):
    # Eager session.
    eager = parse_file(rawlog_path)
    store = get_store()
    eager_sess = store.create(
        filename="f", size_bytes=rawlog_path.stat().st_size, rawlog=eager
    )
    r1 = client.get("/api/summary", params={"session": eager_sess.session_id})
    assert r1.status_code == 200
    eager_body = r1.json()

    # Lazy session on the same file.
    lazy = LazyRawLog.open(rawlog_path)
    lazy_sess = store.create_lazy(
        filename="f", size_bytes=rawlog_path.stat().st_size, lazy_rawlog=lazy
    )
    r2 = client.get("/api/summary", params={"session": lazy_sess.session_id})
    assert r2.status_code == 200
    lazy_body = r2.json()

    # The session id differs; everything else must match bit-for-bit.
    eager_body.pop("session")
    lazy_body.pop("session")
    assert lazy_body == eager_body


def test_summary_lazy_smoke_fields(rawlog_path: Path, client: TestClient):
    # Smoke test: lazy sessions populate every documented summary field.
    lazy = LazyRawLog.open(rawlog_path)
    sess = get_store().create_lazy(filename="f", size_bytes=1, lazy_rawlog=lazy)
    body = client.get("/api/summary", params={"session": sess.session_id}).json()
    assert body["sample_count"] > 0
    tr = body["time_range"]
    assert tr["start"] > 0
    assert tr["end"] >= tr["start"]
    assert tr["duration_seconds"] == tr["end"] - tr["start"]
    assert tr["interval_seconds"] is not None
    assert tr["recommended_min_range_seconds"] > 0
    assert body["system"]["hostname"]
    assert body["rawlog"]["hertz"] > 0
    assert body["tasks"]["avg_per_sample"] > 0
    assert body["tasks"]["max_per_sample"] > 0


def test_summary_sample_index_helpers(rawlog_path: Path):
    """SampleIndex grows new helpers the summary route relies on."""
    from atop_web.parser.lazy import LazyRawLog as LR
    from atop_web.parser.reader import parse_stream

    with rawlog_path.open("rb") as fh:
        rlog = parse_stream(fh, lazy=True)
    idx = rlog.index
    # Boundary helpers.
    assert idx.first_time() == idx.timestamps[0]
    assert idx.last_time() == idx.timestamps[-1]
    # Median interval: must match the eager computation in
    # llm.context._median_interval_seconds.
    eager = parse_file(rawlog_path)
    from atop_web.llm.context import _median_interval_seconds

    assert idx.median_interval_seconds() == _median_interval_seconds(eager.samples)


def test_summary_lazy_handles_empty_index(rawlog_path: Path):
    # Edge case: summary must not crash when the lazy index is empty.
    # We simulate by building an index with zero samples.
    from atop_web.parser.index import SampleIndex
    from atop_web.parser.reader import parse_stream

    with rawlog_path.open("rb") as fh:
        rlog = parse_stream(fh, lazy=True)
    spec = rlog.index.spec
    import array

    empty = SampleIndex(
        offsets=array.array("q", []),
        timestamps=array.array("q", []),
        scomplens=array.array("I", []),
        pcomplens=array.array("I", []),
        ndeviats=array.array("I", []),
        spec=spec,
    )
    assert empty.first_time() is None
    assert empty.last_time() is None
    assert empty.median_interval_seconds() is None
