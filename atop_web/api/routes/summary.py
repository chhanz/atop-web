"""GET /api/summary: high level statistics across the whole session."""

from __future__ import annotations

from fastapi import APIRouter, Query

from atop_web.api.sessions import get_store
from atop_web.llm.context import (
    _median_interval_seconds,
    _recommended_min_range,
)

router = APIRouter()


@router.get("/summary")
def summary(session: str = Query(...)) -> dict:
    sess = get_store().require(session)
    rawlog = sess.rawlog
    header = rawlog.header

    if rawlog.samples:
        start = rawlog.samples[0].curtime
        end = rawlog.samples[-1].curtime
        total_ndeviat = sum(s.ndeviat for s in rawlog.samples)
        avg_ndeviat = total_ndeviat / len(rawlog.samples)
        max_ndeviat = max(s.ndeviat for s in rawlog.samples)
    else:
        start = 0
        end = 0
        avg_ndeviat = 0.0
        max_ndeviat = 0

    interval_seconds = _median_interval_seconds(rawlog.samples)
    recommended_min_range = _recommended_min_range(interval_seconds)

    return {
        "session": session,
        "filename": sess.filename,
        "size_bytes": sess.size_bytes,
        "sample_count": len(rawlog.samples),
        "time_range": {
            "start": start,
            "end": end,
            "duration_seconds": end - start if rawlog.samples else 0,
            # Exposed so the chat client can enforce a minimum range width
            # when the user clicks a ``<range/>`` hint badge. Without this
            # a 10 minute tag on a capture with 600s sample intervals
            # would land between samples and show "no samples in range".
            "interval_seconds": interval_seconds,
            "recommended_min_range_seconds": recommended_min_range,
        },
        "system": {
            "hostname": header.nodename,
            "sysname": header.sysname,
            "release": header.release,
            "version": header.version,
            "machine": header.machine,
            "domainname": header.domainname,
            "pagesize": header.pagesize,
        },
        "rawlog": {
            "aversion": header.aversion,
            "rawheadlen": header.rawheadlen,
            "rawreclen": header.rawreclen,
            "sstatlen": header.sstatlen,
            "tstatlen": header.tstatlen,
            "hertz": header.hertz,
        },
        "tasks": {
            "avg_per_sample": avg_ndeviat,
            "max_per_sample": max_ndeviat,
        },
    }
