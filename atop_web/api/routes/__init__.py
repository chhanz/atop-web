"""FastAPI routers grouped by concern."""

from atop_web.api.routes import upload, samples, processes, summary, files, jobs

__all__ = ["upload", "samples", "processes", "summary", "files", "jobs"]
