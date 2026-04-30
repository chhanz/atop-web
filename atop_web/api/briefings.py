"""In process store for Level 1 briefing results.

Briefings are tied to a parsing job id: once the job finishes the UI fires
one ``POST /api/jobs/{id}/briefing`` request and the result is cached here
for subsequent ``GET`` polls or re-renders. Storage is deliberately in
memory; the briefing is cheap to regenerate from the session if needed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BriefingEntry:
    job_id: str
    status: str  # "ok" | "error"
    provider: str
    model: str | None
    issues: list[dict] = field(default_factory=list)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "provider": self.provider,
            "model": self.model,
            "issues": self.issues,
            "error": self.error,
            "created_at": self.created_at,
            "truncated": self.truncated,
        }


class BriefingStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, BriefingEntry] = {}

    def put(self, entry: BriefingEntry) -> BriefingEntry:
        with self._lock:
            self._items[entry.job_id] = entry
        return entry

    def get(self, job_id: str) -> BriefingEntry | None:
        with self._lock:
            return self._items.get(job_id)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


_STORE = BriefingStore()


def get_briefing_store() -> BriefingStore:
    return _STORE
