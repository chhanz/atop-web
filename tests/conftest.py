"""Shared pytest fixtures for atop-web tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


SAMPLE_FILENAME = "atop_20260427"
# atop 2.7.1 AL2 rawlog used for version dispatch tests (Phase 16).
SAMPLE_27_FILENAME = "al2_atop_20260403"
# atop 2.11.1 RHEL 10.1 rawlog used for Stage A version dispatch tests.
# RHEL 10 / SLES 15-SP7 / SLES 16.0 all ship atop 2.11.1 which shares the
# 2.12 struct layout; this capture exercises the SPEC_2_11 -> SPEC_2_12
# CDEF aliasing path.
SAMPLE_211_FILENAME = "atop_20260506_rl10"
# atop 2.10.0 Ubuntu 24.04 rawlog used for Stage B version dispatch tests.
# 2.10 has distinct struct sizes (tstat=992, sstat=1_030_216) and a
# pre-cgroup rawrecord layout, so this fixture is the only way to
# exercise SPEC_2_10 end to end.
SAMPLE_210_FILENAME = "atop_20260506_u24"


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


@pytest.fixture(scope="session")
def rawlog_211_path() -> Path:
    """Path to an atop 2.11.1 rawlog (RHEL 10.1).

    Tests that use this fixture should still pass ``max_samples`` to avoid
    decoding the whole capture on every run.
    """
    for candidate in _candidate_dirs():
        fp = candidate / SAMPLE_211_FILENAME
        if fp.is_file():
            return fp
    pytest.skip(
        f"sample rawlog {SAMPLE_211_FILENAME} not found in any known directory"
    )


@pytest.fixture(scope="session")
def rawlog_210_path() -> Path:
    """Path to an atop 2.10.0 rawlog (Ubuntu 24.04).

    Exercises SPEC_2_10 end to end (distinct tstat=992 / sstat=1_030_216
    sizes, pre-cgroup rawrecord). Pass ``max_samples`` in tests to keep
    parse cost low.
    """
    for candidate in _candidate_dirs():
        fp = candidate / SAMPLE_210_FILENAME
        if fp.is_file():
            return fp
    pytest.skip(
        f"sample rawlog {SAMPLE_210_FILENAME} not found in any known directory"
    )
