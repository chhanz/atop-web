"""Eager-decode convenience wrapper, kept for tests only.

Phase 22 flipped the production default to lazy. A handful of tests
still want to compare lazy output against a fully materialized
``RawLog`` (dataclass with ``samples: list[Sample]``); this module is
where they go for it. Do not import from here in production code.
"""

from __future__ import annotations

from pathlib import Path

from atop_web.parser.reader import RawLog, parse_file


def decode_eager(path: str | Path, *, max_samples: int | None = None) -> RawLog:
    """Force the legacy eager decode regardless of the ``ATOP_LAZY`` env."""
    return parse_file(path, max_samples=max_samples, lazy=False)
