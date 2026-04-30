"""LLM endpoints: provider health, job briefings, and chat streaming.

Layout decisions:

* ``GET /api/llm/health`` is stateless; it asks ``get_provider()`` for a
  snapshot and returns it. The UI uses it to decide whether to render the
  AI briefing card at all.
* ``POST /api/jobs/{job_id}/briefing`` generates a fresh briefing for the
  session that the job produced. Providers raise ``LLMProviderError`` when
  they cannot honor the request; we surface that as a 502 so the UI can
  show an inline error banner.
* ``GET /api/jobs/{job_id}/briefing`` returns the cached briefing or 404.
* ``POST /api/jobs/{job_id}/chat/stream`` streams a chat reply as SSE. The
  ``none`` provider returns 503 so the UI can hide the panel; other
  providers surface errors inline as ``error`` SSE frames.
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from atop_web.api.briefings import BriefingEntry, get_briefing_store
from atop_web.api.jobs import get_job_store
from atop_web.api.sessions import get_store as get_session_store
from atop_web.llm import get_provider
from atop_web.llm.briefing import generate_briefing
from atop_web.llm.chat import ChatRequest, parse_iso_epoch, stream_chat
from atop_web.llm.provider import LLMProviderError, PROVIDER_NONE

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/llm/health")
def llm_health() -> dict:
    provider = get_provider()
    snap = provider.health()
    snap.setdefault("provider", provider.name)
    return snap


def _rawlog_for_job(job_id: str):
    job = get_job_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != "done" or not job.result:
        raise HTTPException(
            status_code=409,
            detail=f"job is not ready (status={job.status})",
        )
    session_id = job.result.get("session")
    if not session_id:
        raise HTTPException(status_code=409, detail="job has no session id")
    session = get_session_store().get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"session {session_id} not found"
        )
    return session.rawlog


@router.post("/jobs/{job_id}/briefing")
def create_briefing(job_id: str) -> dict:
    rawlog = _rawlog_for_job(job_id)
    provider = get_provider()
    try:
        briefing = generate_briefing(provider, rawlog)
    except LLMProviderError as exc:
        entry = BriefingEntry(
            job_id=job_id,
            status="error",
            provider=provider.name,
            model=getattr(provider, "model", None),
            error=str(exc),
        )
        get_briefing_store().put(entry)
        # 502 keeps this path distinct from 4xx so the UI can show the
        # returned detail without mistaking it for a client bug.
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    entry = BriefingEntry(
        job_id=job_id,
        status="ok",
        provider=provider.name,
        model=getattr(provider, "model", None),
        issues=briefing.get("issues", []),
    )
    get_briefing_store().put(entry)
    return entry.to_dict()


@router.get("/jobs/{job_id}/briefing")
def read_briefing(job_id: str) -> dict:
    entry = get_briefing_store().get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="briefing not generated yet")
    return entry.to_dict()


def _sse_frame(event: str, data: dict | str) -> str:
    if isinstance(data, (dict, list)):
        payload = json.dumps(data, ensure_ascii=False)
    else:
        payload = str(data)
    # Escape any embedded newlines so they stay inside a single SSE frame.
    lines = payload.split("\n")
    data_lines = "\n".join(f"data: {line}" for line in lines)
    return f"event: {event}\n{data_lines}\n\n"


async def _chat_event_stream(
    provider,
    rawlog,
    chat_req: ChatRequest,
) -> Iterator[bytes]:
    try:
        for ev in stream_chat(provider, rawlog, chat_req):
            yield _sse_frame(ev.type, ev.payload).encode("utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("chat stream crashed")
        yield _sse_frame("error", {"message": f"internal error: {exc}"}).encode(
            "utf-8"
        )


@router.post("/jobs/{job_id}/chat/stream")
async def chat_stream(job_id: str, request: Request) -> StreamingResponse:
    provider = get_provider()
    if provider.name == PROVIDER_NONE:
        # 503 lets the UI differentiate "LLM disabled" from other failures
        # so it can hide the chat panel gracefully.
        raise HTTPException(
            status_code=503,
            detail="LLM is disabled. Set LLM_PROVIDER to enable chat.",
        )
    rawlog = _rawlog_for_job(job_id)

    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"invalid JSON body: {exc}"
        ) from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        raise HTTPException(
            status_code=400, detail="'message' is required and must be a non empty string"
        )

    tr = body.get("time_range") or {}
    if not isinstance(tr, dict):
        raise HTTPException(
            status_code=400, detail="'time_range' must be an object"
        )
    try:
        start_epoch = parse_iso_epoch(tr.get("start"))
        end_epoch = parse_iso_epoch(tr.get("end"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    raw_history = body.get("history") or []
    if not isinstance(raw_history, list):
        raise HTTPException(status_code=400, detail="'history' must be a list")
    history: list[dict] = []
    for entry in raw_history:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            history.append({"role": role, "content": content})

    chat_req = ChatRequest(
        message=message,
        time_range_start=start_epoch,
        time_range_end=end_epoch,
        history=history,
    )

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # disable nginx / traefik response buffering
    }
    return StreamingResponse(
        _chat_event_stream(provider, rawlog, chat_req),
        media_type="text/event-stream",
        headers=headers,
    )
