"""Shared parsing pipeline used by the upload and server file routes.

Phase 22 T-10: the pipeline now takes a filesystem path instead of a
``bytes`` payload. Uploads spool straight to disk via
``SpooledTemporaryFile(max_size=0)`` so the rawlog bytes never live as
a single Python ``bytes`` (which would cost memory equal to the file
size), and the parser reads them through the lazy path that keeps
only an offset index in memory.

For server-side picks the path is already on disk, so the flow simply
reuses the same ``run_parse_job(path, ...)`` entry point.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import BinaryIO, Iterable, Union

from atop_web.api.jobs import (
    STAGE_BUILDING_SAMPLES,
    STAGE_DECODING_SSTAT,
    STAGE_DECODING_TSTAT,
    STAGE_HEADER,
    STAGE_SCANNING,
    get_job_store,
)
from atop_web.api.sessions import get_store
from atop_web.parser import RawLogError
from atop_web.parser.aggregate import build_aggregate
from atop_web.parser.lazy import LazyRawLog


# Map parser stage keys to the canonical job stage constants so the parser
# stays decoupled from the job store module.
_STAGE_MAP = {
    "header": STAGE_HEADER,
    "scanning": STAGE_SCANNING,
    "decoding_sstat": STAGE_DECODING_SSTAT,
    "decoding_tstat": STAGE_DECODING_TSTAT,
    "building_samples": STAGE_BUILDING_SAMPLES,
    # Phase 22 lazy mode emits this instead of building_samples.
    "index_built": STAGE_BUILDING_SAMPLES,
}


# ---------------------------------------------------------------------------
# Upload spooling


# Chunk size for the async upload read loop. Picked to match typical
# network MTU multiples without being so large that a slow TLS uploader
# blocks the loop for more than a few ms per chunk.
_UPLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB


def _open_spool() -> "tuple[BinaryIO, Path]":
    """Create a disk-backed temp file and return ``(handle, path)``.

    ``NamedTemporaryFile(delete=False)`` gives us the same "payload lives
    on disk, not in a Python bytes" property as a rolled-over
    ``SpooledTemporaryFile`` without depending on the 3.12+ ``delete=``
    kwarg on the spooled variant. The caller is responsible for closing
    the handle and for eventually unlinking the path.
    """
    tmpdir = os.environ.get("TMPDIR") or tempfile.gettempdir()
    spool = tempfile.NamedTemporaryFile(
        mode="w+b", dir=tmpdir, prefix="atop_upload_", delete=False
    )
    return spool, Path(spool.name)


def spool_upload(chunks: Iterable[bytes]) -> Path:
    """Write ``chunks`` to a temp file on disk and return its path.

    Honors ``TMPDIR`` so deployments on Docker with tmpfs at /tmp can
    redirect the spool to a persistent volume via env. The file never
    accumulates a full-payload Python buffer: chunks are written
    through the OS file handle as they arrive and released.
    """
    spool, path = _open_spool()
    try:
        for chunk in chunks:
            if chunk:
                spool.write(chunk)
        spool.flush()
    finally:
        spool.close()
    return path


async def spool_upload_async(upload_file) -> Path:
    """Async wrapper that chunks an ``UploadFile`` directly to disk."""
    spool, path = _open_spool()
    try:
        while True:
            chunk = await upload_file.read(_UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            spool.write(chunk)
        spool.flush()
    finally:
        spool.close()
    return path


def _make_progress_cb(job_id: str):
    """Return a parser friendly progress callback bound to ``job_id``."""
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


# ---------------------------------------------------------------------------
# Job runner


def run_parse_job(
    job_id: str,
    source_path: Union[str, Path, bytes],
    *,
    filename: str,
    source: str,
    owns_path: bool = False,
) -> None:
    """Parse a rawlog off disk and create a lazy session.

    ``source_path`` is a filesystem path the LazyRawLog will keep open
    for the session's lifetime. The legacy ``bytes`` signature is still
    accepted for back-compat with existing tests — we spool it to a
    temp file first so the lazy decoder has something to seek in.

    ``owns_path`` marks the file as owned by this pipeline (typically a
    SpooledTemporaryFile). It is **not** deleted here because the
    session keeps a handle open; cleanup is the responsibility of
    ``SessionStore.clear()`` and the "delete session" path we leave for
    a later phase. This deliberately trades a small disk-tmp leak for
    simplicity while the lazy session is alive.
    """
    job_store = get_job_store()
    job_store.update(job_id, status="running", stage=STAGE_HEADER)

    progress_cb = _make_progress_cb(job_id)

    # Back-compat: some callers still hand us raw bytes. Spool it so the
    # lazy parser has a file to seek in.
    if isinstance(source_path, (bytes, bytearray, memoryview)):
        source_path = spool_upload([bytes(source_path)])
        owns_path = True

    path = Path(source_path)

    try:
        lazy = LazyRawLog.open(path)
        # Walk the index once for progress reporting — the actual scan
        # happened inside ``parse_stream(lazy=True)``. We synthesize a
        # final 85% event so the frontend's job poll shows something
        # past "scanning".
        progress_cb("index_built", len(lazy), len(lazy), 85)
    except RawLogError as exc:
        job_store.mark_error(job_id, str(exc))
        return
    except Exception as exc:  # defensive
        job_store.mark_error(job_id, f"parse failed: {exc}")
        return

    try:
        aggregate = build_aggregate(lazy)
    except Exception:  # aggregate failure must not bring the whole parse down
        aggregate = None

    size_bytes = path.stat().st_size
    session = get_store().create_lazy(
        filename=filename,
        size_bytes=size_bytes,
        lazy_rawlog=lazy,
        aggregate=aggregate,
    )

    job_store.update(
        job_id,
        stage=STAGE_BUILDING_SAMPLES,
        progress=95,
        detail=f"{len(lazy)} samples",
    )

    header = lazy.header
    result = {
        "session": session.session_id,
        "filename": session.filename,
        "size_bytes": session.size_bytes,
        "sample_count": len(lazy),
        "hostname": header.nodename,
        "kernel": header.release,
        "aversion": header.aversion,
        "source": source,
    }
    job_store.mark_done(job_id, result)


def schedule_parse_job(
    job_id: str,
    source_path: Union[str, Path, bytes],
    *,
    filename: str,
    source: str,
    owns_path: bool = False,
) -> None:
    """Fire off ``run_parse_job`` on a daemon thread and return immediately."""
    thread = threading.Thread(
        target=run_parse_job,
        args=(job_id, source_path),
        kwargs={"filename": filename, "source": source, "owns_path": owns_path},
        daemon=True,
        name=f"parse-job-{job_id}",
    )
    thread.start()
