"""GET /api/dashboard: fan-out wrapper for the six page-load fetches.

Phase 23 T-21. The dashboard HTML used to load via six independent
calls: summary, /api/samples, the four /api/samples/system_* charts
and /api/processes. Each of those round-trips paid for its own
session lookup plus - for the wide windows the UI defaults to - its
own sstat inflates over the same samples. Collapsing the six into
one server-side fan-out gives a single 202 + single JSON back to the
browser, and lets us add a response cache (T-22) without chasing
every entry point.

The route deliberately reuses the existing handler functions rather
than re-implementing the work here: each underlying route is the
single source of truth for its section, and this wrapper just stitches
the results into one envelope. That keeps the JSON schema of every
individual section byte-identical to what the old endpoint returned.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from atop_web.api.cache import get_response_cache
from atop_web.api.routes import processes as processes_route
from atop_web.api.routes import samples as samples_route
from atop_web.api.routes import summary as summary_route
from atop_web.api.sessions import get_store
from atop_web.api.timerange import parse_iso_epoch

router = APIRouter()


def _gather_sections(
    sess,
    from_epoch: int | None,
    to_epoch: int | None,
    process_limit: int,
    process_index: int | None,
) -> dict:
    """Call each section handler in turn and collect the results.

    Returns a dict with a ``_call_trace`` key listing which handlers
    actually ran, so the dashboard route's own tests can assert the
    fan-out shape without having to monkeypatch each handler. The
    trace is stripped before the route returns to the client.
    """
    session_id = sess.session_id
    trace: list[str] = []

    summary_body = summary_route.summary(session=session_id)
    trace.append("summary")

    # Convert epoch back to ISO for the handler; ``samples`` takes
    # ISO strings via ``parse_iso_epoch`` internally.
    def _iso(epoch: int | None) -> str | None:
        if epoch is None:
            return None
        from datetime import datetime, timezone

        return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    from_iso = _iso(from_epoch)
    to_iso = _iso(to_epoch)

    samples_body = samples_route.samples(
        session=session_id, from_=from_iso, to=to_iso
    )
    trace.append("samples")

    cpu_body = samples_route.system_cpu(
        session=session_id, from_=from_iso, to=to_iso
    )
    trace.append("system_cpu")

    mem_body = samples_route.system_memory(
        session=session_id, from_=from_iso, to=to_iso
    )
    trace.append("system_memory")

    disk_body = samples_route.system_disk(
        session=session_id, from_=from_iso, to=to_iso
    )
    trace.append("system_disk")

    net_body = samples_route.system_network(
        session=session_id, from_=from_iso, to=to_iso
    )
    trace.append("system_network")

    try:
        processes_body = processes_route.processes(
            session=session_id,
            time=None,
            index=process_index,
            limit=process_limit,
            sort_by="cpu",
            order="desc",
            from_=from_iso,
            to=to_iso,
        )
    except HTTPException as exc:
        # The processes handler is strict about "no samples" errors;
        # surface an empty payload so the rest of the dashboard still
        # renders in that edge case.
        if exc.status_code == 404:
            processes_body = {
                "session": session_id,
                "curtime": 0,
                "interval": 0,
                "ndeviat": 0,
                "nactproc": 0,
                "sort_by": "cpu",
                "order": "desc",
                "count": 0,
                "meta": {"hertz": None, "ncpu": None, "interval_sec": 0},
                "processes": [],
            }
        else:
            raise
    trace.append("processes")

    return {
        "summary": summary_body,
        "samples": samples_body,
        "charts": {
            "cpu": cpu_body,
            "memory": mem_body,
            "disk": disk_body,
            "network": net_body,
        },
        "processes": processes_body,
        "_call_trace": trace,
    }


@router.get("/dashboard")
def dashboard(
    session: str = Query(...),
    from_: str | None = Query(None, alias="from", description="ISO8601 lower bound"),
    to: str | None = Query(None, description="ISO8601 upper bound"),
    process_limit: int = Query(
        200, ge=1, le=10000,
        description="Cap on the top-N process rows returned.",
    ),
    process_index: int | None = Query(
        None, ge=0,
        description="Sample index inside the window for the processes section.",
    ),
) -> dict:
    sess = get_store().require(session)
    from_epoch = parse_iso_epoch(from_, field="from")
    to_epoch = parse_iso_epoch(to, field="to")

    cache = get_response_cache()
    key = (
        session,
        "dashboard",
        from_epoch,
        to_epoch,
        process_limit,
        process_index,
    )

    def builder():
        body = _gather_sections(
            sess=sess,
            from_epoch=from_epoch,
            to_epoch=to_epoch,
            process_limit=process_limit,
            process_index=process_index,
        )
        body.pop("_call_trace", None)
        return body

    return cache.get_or_compute(key, builder)
