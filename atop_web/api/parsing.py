"""Shared parsing pipeline used by the upload and server file routes.

Both entry points hand off to the same background function: they give a
``Job`` id, a reference to the in memory bytes to parse, and a label for
the resulting session. The background worker drives the parser, updates
the job store with per stage progress, and finally creates a session that
the client can query with the usual ``/api/samples`` / ``/api/processes``
endpoints.
"""

from __future__ import annotations

import threading

from atop_web.api.jobs import (
    STAGE_BUILDING_SAMPLES,
    STAGE_DECODING_SSTAT,
    STAGE_DECODING_TSTAT,
    STAGE_HEADER,
    STAGE_SCANNING,
    get_job_store,
)
from atop_web.api.sessions import get_store
from atop_web.parser import RawLogError, parse_bytes


# Map parser stage keys to the canonical job stage constants so the parser
# stays decoupled from the job store module.
_STAGE_MAP = {
    "header": STAGE_HEADER,
    "scanning": STAGE_SCANNING,
    "decoding_sstat": STAGE_DECODING_SSTAT,
    "decoding_tstat": STAGE_DECODING_TSTAT,
    "building_samples": STAGE_BUILDING_SAMPLES,
}


def _make_progress_cb(job_id: str):
    """Return a parser friendly progress callback bound to ``job_id``.

    The callback throttles updates to at most one per percent change so the
    job store is not hammered in the tight sample loop.
    """
    job_store = get_job_store()
    last = {"stage": None, "progress": -1}

    def cb(stage_key: str, current: int, total: int | None, progress: int) -> None:
        stage = _STAGE_MAP.get(stage_key, stage_key)
        detail: str | None
        if total is not None and total > 0 and stage_key in (
            "decoding_sstat",
            "decoding_tstat",
        ):
            detail = f"{current} / {total} samples"
        else:
            detail = None

        # Skip redundant updates to keep the /api/jobs poll responses cheap.
        if stage == last["stage"] and progress == last["progress"]:
            return
        last["stage"] = stage
        last["progress"] = progress

        job_store.update(
            job_id,
            stage=stage,
            progress=progress,
            status="running",
            detail=detail,
        )

    return cb


def run_parse_job(
    job_id: str,
    data: bytes,
    *,
    filename: str,
    source: str,
) -> None:
    """Synchronous parsing pipeline.

    This runs on a daemon thread so the event loop is never blocked by
    ``parse_bytes``; the real work is CPU bound python, not I/O.
    """
    job_store = get_job_store()
    job_store.update(job_id, status="running", stage=STAGE_HEADER)

    progress_cb = _make_progress_cb(job_id)

    try:
        rawlog = parse_bytes(data, progress_cb=progress_cb)
    except RawLogError as exc:
        job_store.mark_error(job_id, str(exc))
        return
    except Exception as exc:  # defensive; surface as error rather than crash task
        job_store.mark_error(job_id, f"parse failed: {exc}")
        return

    job_store.update(
        job_id,
        stage=STAGE_BUILDING_SAMPLES,
        progress=90,
        detail=f"{len(rawlog.samples)} samples",
    )

    session = get_store().create(
        filename=filename,
        size_bytes=len(data),
        rawlog=rawlog,
    )

    header = rawlog.header
    result = {
        "session": session.session_id,
        "filename": session.filename,
        "size_bytes": session.size_bytes,
        "sample_count": len(rawlog.samples),
        "hostname": header.nodename,
        "kernel": header.release,
        "aversion": header.aversion,
        "source": source,
    }
    job_store.mark_done(job_id, result)


def schedule_parse_job(
    job_id: str,
    data: bytes,
    *,
    filename: str,
    source: str,
) -> None:
    """Fire off ``run_parse_job`` on a daemon thread and return immediately.

    A dedicated thread keeps the work independent of the request event loop;
    this matters under TestClient where the loop is torn down at the end of
    the request that started the job.
    """
    thread = threading.Thread(
        target=run_parse_job,
        args=(job_id, data),
        kwargs={"filename": filename, "source": source},
        daemon=True,
        name=f"parse-job-{job_id}",
    )
    thread.start()
