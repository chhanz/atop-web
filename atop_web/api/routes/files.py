"""Server side file browser for a mounted atop log directory.

The typical deployment pattern is to bind mount the directory that atop
writes its rawlog files into (on the host, usually ``/var/log/atop``) into
the container read only. Operators can then pick a file directly from the
web UI without going through the upload flow.

Two endpoints live here:

- ``GET  /api/files``       -> list candidate rawlog files in the mount
- ``POST /api/files/parse`` -> parse a file by name and open a session

Any operation that resolves a file name must defend against path traversal.
Names are constrained to a safe character set, and the resolved real path is
checked to stay inside the configured directory after symlink resolution.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from atop_web.api.jobs import STAGE_UPLOAD_SAVED, get_job_store
from atop_web.api.parsing import run_parse_job, schedule_parse_job

router = APIRouter()

ATOP_LOG_DIR = os.environ.get("ATOP_LOG_DIR", "/var/log/atop")

# Accept the bare atop rawlog file name shape (letters, digits, dot, dash,
# underscore). This is intentionally strict; rawlog files produced by atop
# do not need anything outside this set.
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Prefer files that look like an atop rawlog. atop writes names such as
# ``atop_20260427`` by default, but operators sometimes rotate or rename
# them, so the filter is a superset: we also accept files without an
# extension. Directories, dotfiles and extensions like .gz / .log are left
# out to keep the picker focused.
_ATOP_NAME_RE = re.compile(r"^atop_\d{8}(?:[A-Za-z0-9._-]*)?$")

_DATE_RE = re.compile(r"atop_(\d{4})(\d{2})(\d{2})")


class ParseRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


def _log_dir() -> Path | None:
    if not ATOP_LOG_DIR:
        return None
    path = Path(ATOP_LOG_DIR)
    if not path.is_dir():
        return None
    return path


def _is_candidate(entry: Path) -> bool:
    if not entry.is_file():
        return False
    if entry.name.startswith("."):
        return False
    return bool(_ATOP_NAME_RE.match(entry.name))


def _date_guess(name: str) -> str | None:
    match = _DATE_RE.search(name)
    if not match:
        return None
    yyyy, mm, dd = match.groups()
    try:
        return f"{yyyy}-{mm}-{dd}"
    except Exception:
        return None


def _safe_resolve(base: Path, name: str) -> Path:
    if not _NAME_RE.match(name) or "/" in name or ".." in name:
        raise HTTPException(status_code=400, detail="invalid file name")

    base_real = Path(os.path.realpath(base))
    target = base / name
    target_real = Path(os.path.realpath(target))
    try:
        target_real.relative_to(base_real)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="path traversal blocked") from exc
    if not target_real.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return target_real


@router.get("/files")
def list_files() -> dict:
    base = _log_dir()
    if base is None:
        return {
            "enabled": False,
            "log_dir": ATOP_LOG_DIR or "",
            "files": [],
        }

    files = []
    for entry in base.iterdir():
        if not _is_candidate(entry):
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        files.append(
            {
                "name": entry.name,
                "size": stat.st_size,
                "mtime": mtime.isoformat(),
                "date_guess": _date_guess(entry.name),
            }
        )

    files.sort(key=lambda f: f["mtime"], reverse=True)

    return {
        "enabled": True,
        "log_dir": str(base),
        "files": files,
    }


@router.post("/files/parse")
async def parse_file(
    request: ParseRequest,
    sync: int = Query(0, ge=0, le=1, description="1 returns after parsing completes"),
) -> JSONResponse:
    base = _log_dir()
    if base is None:
        raise HTTPException(status_code=404, detail="log directory not configured")

    target = _safe_resolve(base, request.name)

    data = target.read_bytes()
    if not data:
        raise HTTPException(status_code=400, detail="file is empty")

    filename = target.name

    if sync:
        job = get_job_store().create(source="server", filename=filename)
        run_parse_job(job.job_id, data, filename=filename, source="server")
        final = get_job_store().get(job.job_id)
        if final is None or final.status != "done":
            raise HTTPException(
                status_code=400,
                detail=(final.error if final else "parse failed"),
            )
        return JSONResponse(final.result)

    job = get_job_store().create(source="server", filename=filename)
    get_job_store().update(job.job_id, stage=STAGE_UPLOAD_SAVED)
    schedule_parse_job(job.job_id, data, filename=filename, source="server")

    return JSONResponse(
        {
            "job_id": job.job_id,
            "source": "server",
            "filename": filename,
            "size_bytes": len(data),
        },
        status_code=status.HTTP_202_ACCEPTED,
    )
