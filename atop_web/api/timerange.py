"""Helpers for parsing optional ISO8601 time range filters."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException


def parse_iso_epoch(value: str | None, *, field: str) -> int | None:
    """Parse an ISO8601 timestamp into an epoch second integer.

    ``None`` and empty strings are returned as ``None`` so the caller can
    easily treat them as "no bound". Naive timestamps are interpreted as
    UTC, matching the semantics used by the rawlog header.
    """
    if value is None or value == "":
        return None
    text = value.strip()
    # ``datetime.fromisoformat`` in Python 3.11+ handles trailing ``Z`` only
    # on 3.11+, but we normalize here to stay forward compatible.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"invalid ISO8601 in {field!r}: {value!r}"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def filter_samples(samples, from_epoch: int | None, to_epoch: int | None):
    """Return the subset of ``samples`` whose ``curtime`` is in range."""
    if from_epoch is None and to_epoch is None:
        return list(samples)
    lo = from_epoch if from_epoch is not None else -(1 << 62)
    hi = to_epoch if to_epoch is not None else (1 << 62)
    return [s for s in samples if lo <= s.curtime <= hi]
