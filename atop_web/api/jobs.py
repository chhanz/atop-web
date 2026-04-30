"""In process job store for background rawlog parsing.

Parsing a large rawlog can take tens of seconds. The HTTP handlers that
trigger a parse therefore return immediately with a ``job_id``; the work
runs in a background task and the client polls ``GET /api/jobs/{id}`` until
the status flips to ``done`` or ``error``.

Jobs only live in memory. A simple TTL sweeps old entries every time a new
job is created or a lookup happens; there is no separate timer thread.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TTL_SECONDS = 3600

STAGE_PENDING = "pending"
STAGE_UPLOAD_SAVED = "upload_saved"
STAGE_HEADER = "header"
STAGE_SCANNING = "scanning"
STAGE_DECODING_SSTAT = "decoding_sstat"
STAGE_DECODING_TSTAT = "decoding_tstat"
# ``parsing`` is kept for backward compatibility; the parser pipeline emits
# the more specific stages above, but legacy callers (sync tests, retries)
# may still advance a job to this generic stage.
STAGE_PARSING = "parsing"
STAGE_BUILDING_SAMPLES = "building_samples"
STAGE_DONE = "done"
STAGE_ERROR = "error"

# Progress fraction that marks the start of each stage. The ranges are:
#   pending        0
#   upload_saved   5
#   header         10
#   scanning       15          (sample count confirmed)
#   decoding_sstat 15 .. 70    (continuous within the stage via progress_cb)
#   decoding_tstat 70 .. 85
#   building       85
#   done           100
_STAGE_PROGRESS = {
    STAGE_PENDING: 0,
    STAGE_UPLOAD_SAVED: 5,
    STAGE_HEADER: 10,
    STAGE_SCANNING: 15,
    STAGE_DECODING_SSTAT: 15,
    STAGE_PARSING: 50,
    STAGE_DECODING_TSTAT: 70,
    STAGE_BUILDING_SAMPLES: 85,
    STAGE_DONE: 100,
    STAGE_ERROR: 100,
}

# English user facing labels. We keep the machine key in ``stage`` and the
# human label in ``stage_label`` so the UI can pick either.
_STAGE_LABELS = {
    STAGE_PENDING: "Pending",
    STAGE_UPLOAD_SAVED: "Upload saved",
    STAGE_HEADER: "Validating header (rawheader)",
    STAGE_SCANNING: "Scanning records (rawrecord index)",
    STAGE_DECODING_SSTAT: "Decoding system stats",
    STAGE_PARSING: "Parsing",
    STAGE_DECODING_TSTAT: "Decoding process stats",
    STAGE_BUILDING_SAMPLES: "Building samples",
    STAGE_DONE: "Done",
    STAGE_ERROR: "Error",
}


def stage_label(stage: str) -> str:
    """Return the English user facing label for a stage key."""
    return _STAGE_LABELS.get(stage, stage)


@dataclass
class Job:
    job_id: str
    source: str
    status: str = "pending"
    stage: str = STAGE_PENDING
    progress: int = 0
    filename: str | None = None
    detail: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "source": self.source,
            "status": self.status,
            "stage": self.stage,
            "stage_label": stage_label(self.stage),
            "progress": self.progress,
            "detail": self.detail,
            "filename": self.filename,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error,
        }


# Sentinel used to allow ``update(detail=None)`` to clear the detail field
# while still defaulting to "do not change" when the caller omits the kwarg.
_UNSET = object()


class JobStore:
    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._ttl = ttl_seconds

    def create(self, source: str, filename: str | None = None) -> Job:
        job_id = secrets.token_urlsafe(12)
        job = Job(job_id=job_id, source=source, filename=filename)
        with self._lock:
            self._gc_locked()
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._gc_locked()
            return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        *,
        stage: str | None = None,
        progress: int | None = None,
        status: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        detail: Any = _UNSET,
    ) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if stage is not None:
                job.stage = stage
                if progress is None:
                    job.progress = _STAGE_PROGRESS.get(stage, job.progress)
            if progress is not None:
                # Never let the progress go backwards; the UI relies on this.
                new_value = max(0, min(100, int(progress)))
                if new_value > job.progress:
                    job.progress = new_value
            if status is not None:
                job.status = status
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            if detail is not _UNSET:
                job.detail = detail  # type: ignore[assignment]
            job.updated_at = time.time()
            return job

    def mark_running(self, job_id: str) -> None:
        self.update(job_id, status="running", stage=STAGE_PARSING)

    def mark_done(self, job_id: str, result: dict[str, Any]) -> None:
        self.update(
            job_id,
            status="done",
            stage=STAGE_DONE,
            progress=100,
            result=result,
            detail=None,
        )

    def mark_error(self, job_id: str, message: str) -> None:
        self.update(
            job_id,
            status="error",
            stage=STAGE_ERROR,
            progress=100,
            error=message,
        )

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()

    def _gc_locked(self) -> None:
        deadline = time.time() - self._ttl
        stale = [jid for jid, j in self._jobs.items() if j.updated_at < deadline]
        for jid in stale:
            self._jobs.pop(jid, None)


_STORE = JobStore()


def get_job_store() -> JobStore:
    return _STORE
