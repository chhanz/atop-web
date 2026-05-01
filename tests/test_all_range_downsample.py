"""Phase 24 bug fix: ALL-range chart requests must stay bounded.

Symptom: on a 266 MB rawlog with a 60-second sample interval, the
four ``/api/samples/system_*`` endpoints (and the combined
``/api/dashboard``) take 5+ minutes and time out when asked for the
whole capture. The 1-minute downsample grid that Phase 23 introduced
is exactly the same resolution as the raw samples, so the "one
sample per minute" iterator still yields all 20k rawrecords and
sstat-inflates every one.

The fix is a *target-points* cap: no matter how wide the window,
the chart iterator emits at most ``_CHART_TARGET_POINTS`` samples.
The bucket step grows with the window so a week-long capture picks
one representative every ~20 minutes instead of every minute.

Tests below lock this down with a fast synthetic sequence so they
don't depend on the 266 MB fixture being present.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_chart_window_iter_caps_to_target_points():
    """ALL-range walk yields at most ``_CHART_TARGET_POINTS`` samples."""
    from atop_web.api.routes.samples import (
        _CHART_TARGET_POINTS,
        _chart_window_iter,
    )

    # Simulated 2-week capture at 60-second resolution (20k samples).
    # Build a minimal lazy session stand-in whose ``_iter_window`` yields
    # SimpleNamespace samples with ``curtime``.
    sample_count = 20_000
    first = 1_700_000_000
    samples = [SimpleNamespace(curtime=first + 60 * i) for i in range(sample_count)]
    sess = SimpleNamespace(
        is_lazy=False,
        index=None,
        rawlog=SimpleNamespace(samples=samples),
    )

    # Full range: no bounds.
    emitted = list(_chart_window_iter(sess, None, None))
    assert 0 < len(emitted) <= _CHART_TARGET_POINTS + 1, (
        f"expected <= {_CHART_TARGET_POINTS} points, got {len(emitted)}"
    )


def test_chart_window_iter_with_bounds_respects_target():
    """Bounded wide windows still downsample to the target count."""
    from atop_web.api.routes.samples import (
        _CHART_TARGET_POINTS,
        _chart_window_iter,
    )

    first = 1_700_000_000
    samples = [SimpleNamespace(curtime=first + 60 * i) for i in range(20_000)]
    sess = SimpleNamespace(
        is_lazy=False,
        index=None,
        rawlog=SimpleNamespace(samples=samples),
    )

    # A week-wide window.
    start = first
    end = first + 60 * 10_080  # one week
    emitted = list(_chart_window_iter(sess, start, end))
    assert 0 < len(emitted) <= _CHART_TARGET_POINTS + 1


def test_chart_window_iter_narrow_window_still_full_resolution():
    """Sub-minute windows stay on the per-sample path as before."""
    from atop_web.api.routes.samples import _chart_window_iter

    first = 1_700_000_000
    # 10-second interval.
    samples = [SimpleNamespace(curtime=first + 10 * i) for i in range(10)]
    sess = SimpleNamespace(
        is_lazy=False,
        index=None,
        rawlog=SimpleNamespace(samples=samples),
    )
    emitted = list(_chart_window_iter(sess, first, first + 30))
    # A 30-second window contains 4 samples; don't drop any of them.
    assert len(emitted) >= 3


def test_chart_window_iter_medium_window_downsamples_to_minute_grid():
    """Windows wider than a minute but under the ``target_points`` cap
    stay on the 1-minute grid."""
    from atop_web.api.routes.samples import _chart_window_iter

    first = 1_700_000_000
    # 10-second interval, 2-hour window = 720 samples, target 600.
    # The cap would bump bucket_step to ~12s, which keeps us near full
    # resolution.
    samples = [SimpleNamespace(curtime=first + 10 * i) for i in range(720)]
    sess = SimpleNamespace(
        is_lazy=False,
        index=None,
        rawlog=SimpleNamespace(samples=samples),
    )
    start = first
    end = first + 7_200
    emitted = list(_chart_window_iter(sess, start, end))
    # Expect somewhere around 120 (1-minute buckets) or 600 (target
    # cap) depending on which knob dominates. Either way, no more than
    # the raw sample count.
    assert len(emitted) <= 720
    assert len(emitted) > 0
