"""GET /api/jobs/{job_id}: poll the state of a background parse."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from atop_web.api.jobs import get_job_store

router = APIRouter()


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = get_job_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_dict()
