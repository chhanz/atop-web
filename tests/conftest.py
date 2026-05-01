"""Shared pytest fixtures for atop-web tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


SAMPLE_FILENAME = "atop_20260427"
# atop 2.7.1 AL2 rawlog used for version dispatch tests (Phase 16).
SAMPLE_27_FILENAME = "al2_atop_20260403"


def _candidate_dirs() -> list[Path]:
    env = os.environ.get("ATOP_LOG_DIR")
    dirs: list[Path] = []
    if env:
        dirs.append(Path(env))
    dirs.extend(
        [
            Path("/app/logs"),
            Path("/data"),
            Path("/var/log/atop"),
            # Dev convenience: AL2 fixtures are typically placed under the
            # operator's Downloads during evaluation. Adding the path lets
            # the 2.7 tests run on the host without a symlink requiring
            # root into /var/log/atop.
            Path.home() / "Downloads",
        ]
    )
    return dirs


@pytest.fixture(scope="session")
def rawlog_path() -> Path:
    for candidate in _candidate_dirs():
        fp = candidate / SAMPLE_FILENAME
        if fp.is_file():
            return fp
    pytest.skip(f"sample rawlog {SAMPLE_FILENAME} not found in any known directory")


@pytest.fixture(scope="session")
def rawlog_bytes(rawlog_path: Path) -> bytes:
    return rawlog_path.read_bytes()


@pytest.fixture(autouse=True)
def _clear_response_cache():
    """Drop the in-process response cache between every test.

    The Phase 23 TTL cache is a module-level singleton keyed by
    ``session_id``. Test session ids collide across tests when the
    ``SessionStore`` is reused, so leaving cache entries around would
    return a previous test's body on the next lookup. Clearing in
    teardown (and once before the first test via the import) keeps
    behaviour identical to a fresh process.
    """
    from atop_web.api.cache import get_response_cache

    get_response_cache().clear()
    yield
    get_response_cache().clear()


@pytest.fixture(scope="session")
def rawlog_27_path() -> Path:
    """Path to an atop 2.7.1 rawlog written on AL2.

    The file is 278 MiB, so tests that use it should pass ``max_samples`` to
    avoid decoding the whole capture on every run.
    """
    for candidate in _candidate_dirs():
        fp = candidate / SAMPLE_27_FILENAME
        if fp.is_file():
            return fp
    pytest.skip(
        f"sample rawlog {SAMPLE_27_FILENAME} not found in any known directory"
    )
