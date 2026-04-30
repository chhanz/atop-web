"""In process session store for parsed rawlog files."""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from typing import Dict

from atop_web.parser import RawLog


@dataclass
class Session:
    session_id: str
    filename: str
    size_bytes: int
    rawlog: RawLog


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, Session] = {}

    def create(self, filename: str, size_bytes: int, rawlog: RawLog) -> Session:
        session_id = secrets.token_urlsafe(12)
        session = Session(
            session_id=session_id,
            filename=filename,
            size_bytes=size_bytes,
            rawlog=rawlog,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def require(self, session_id: str) -> Session:
        session = self.get(session_id)
        if session is None:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="session not found")
        return session

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


_STORE = SessionStore()


def get_store() -> SessionStore:
    return _STORE
