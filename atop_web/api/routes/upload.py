"""POST /api/upload: accept a rawlog upload and parse it asynchronously.

Parsing is handed off to a background task and the client polls
``/api/jobs/{id}`` for completion. The legacy synchronous mode is still
available by passing ``?sync=1``; it is handy for callers that want a
single round trip (for example the TestClient) but blocks for the full
duration of the parse.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, status
from fastapi.responses import JSONResponse

from atop_web.api.jobs import STAGE_UPLOAD_SAVED, get_job_store
from atop_web.api.parsing import run_parse_job, schedule_parse_job

router = APIRouter()

MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def _read_upload(data: bytes) -> None:
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="upload too large")


@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
    sync: int = Query(0, ge=0, le=1, description="1 returns after parsing completes"),
) -> JSONResponse:
    data = await file.read()
    _read_upload(data)

    filename = file.filename or "uploaded.rawlog"

    if sync:
        job = get_job_store().create(source="upload", filename=filename)
        run_parse_job(job.job_id, data, filename=filename, source="upload")
        final = get_job_store().get(job.job_id)
        if final is None or final.status != "done":
            raise HTTPException(
                status_code=400,
                detail=(final.error if final else "parse failed"),
            )
        return JSONResponse(final.result)

    job = get_job_store().create(source="upload", filename=filename)
    get_job_store().update(job.job_id, stage=STAGE_UPLOAD_SAVED)
    schedule_parse_job(job.job_id, data, filename=filename, source="upload")

    return JSONResponse(
        {
            "job_id": job.job_id,
            "source": "upload",
            "filename": filename,
            "size_bytes": len(data),
        },
        status_code=status.HTTP_202_ACCEPTED,
    )
