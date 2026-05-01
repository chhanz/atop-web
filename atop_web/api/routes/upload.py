"""POST /api/upload: accept a rawlog upload and parse it asynchronously.

Phase 22 T-10: the upload path streams the request body in 1 MiB
chunks straight into a ``SpooledTemporaryFile`` (disk-backed from the
first byte), then hands the resulting file path to ``run_parse_job``.
The lazy parser opens the file and keeps only an offset index in
memory — the full payload never materialises as Python ``bytes``.

The endpoint's external contract (202 Accepted + job id for async
mode, 200 + session id for sync mode) is unchanged.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, status
from fastapi.responses import JSONResponse

from atop_web.api.jobs import STAGE_UPLOAD_SAVED, get_job_store
from atop_web.api.parsing import (
    run_parse_job,
    schedule_parse_job,
    spool_upload_async,
)

router = APIRouter()

MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def _validate_size(size_bytes: int) -> None:
    if size_bytes <= 0:
        raise HTTPException(status_code=400, detail="empty upload")
    if size_bytes > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="upload too large")


@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
    sync: int = Query(0, ge=0, le=1, description="1 returns after parsing completes"),
) -> JSONResponse:
    filename = file.filename or "uploaded.rawlog"
    path = await spool_upload_async(file)
    size_bytes = path.stat().st_size
    try:
        _validate_size(size_bytes)
    except HTTPException:
        path.unlink(missing_ok=True)
        raise

    if sync:
        job = get_job_store().create(source="upload", filename=filename)
        run_parse_job(
            job.job_id, path, filename=filename, source="upload", owns_path=True
        )
        final = get_job_store().get(job.job_id)
        if final is None or final.status != "done":
            raise HTTPException(
                status_code=400,
                detail=(final.error if final else "parse failed"),
            )
        return JSONResponse(final.result)

    job = get_job_store().create(source="upload", filename=filename)
    get_job_store().update(job.job_id, stage=STAGE_UPLOAD_SAVED)
    schedule_parse_job(
        job.job_id, path, filename=filename, source="upload", owns_path=True
    )

    return JSONResponse(
        {
            "job_id": job.job_id,
            "source": "upload",
            "filename": filename,
            "size_bytes": size_bytes,
        },
        status_code=status.HTTP_202_ACCEPTED,
    )
