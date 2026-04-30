"""FastAPI application entry point for atop-web."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from atop_web.api.routes import (
    files,
    jobs,
    llm,
    processes,
    samples,
    summary,
    upload,
)

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_LOG_DIR = os.environ.get("ATOP_LOG_DIR", "/var/log/atop")


def _compute_base_href(root_path: str) -> str:
    """Return the value to inject into the HTML <base href>.

    When served directly on the host root, a relative base (``./``) keeps
    asset and API URLs working regardless of how the page is fetched. When
    running behind a reverse proxy that mounts the app at a sub path, the
    operator sets ``ATOP_ROOT_PATH`` to that external prefix so the browser
    resolves assets and XHR calls back through the proxy.
    """
    if not root_path:
        return "./"
    return root_path.rstrip("/") + "/"


def create_app(root_path: str | None = None) -> FastAPI:
    """Build the FastAPI application.

    ``root_path`` is only used to render the ``<base href>`` tag in the served
    HTML; it deliberately does not set FastAPI's ``root_path`` argument.

    The expected deployment pattern is that a reverse proxy (nginx, Caddy,
    Traefik, ...) strips the external prefix before forwarding the request,
    so the application always sees paths starting at ``/``. Setting FastAPI's
    ``root_path`` on top of a stripping proxy would double apply the prefix to
    static mounts and route matching.
    """
    if root_path is None:
        root_path = os.environ.get("ATOP_ROOT_PATH", "")
    base_href = _compute_base_href(root_path)

    app = FastAPI(
        title="atop-web",
        version="0.1.0",
        description="Web based visualization for atop rawlog files.",
    )

    app.include_router(upload.router, prefix="/api", tags=["upload"])
    app.include_router(samples.router, prefix="/api", tags=["samples"])
    app.include_router(processes.router, prefix="/api", tags=["processes"])
    app.include_router(summary.router, prefix="/api", tags=["summary"])
    app.include_router(files.router, prefix="/api", tags=["files"])
    app.include_router(jobs.router, prefix="/api", tags=["jobs"])
    app.include_router(llm.router, prefix="/api", tags=["llm"])

    @app.get("/healthz", tags=["meta"])
    def healthz() -> dict:
        return {"status": "ok", "log_dir": DEFAULT_LOG_DIR}

    if STATIC_DIR.is_dir():
        app.mount(
            "/static", StaticFiles(directory=STATIC_DIR), name="static"
        )

        index_html = (STATIC_DIR / "index.html").read_text()
        rendered_index = index_html.replace(
            "<!--BASE_HREF-->",
            f'<base href="{base_href}" />',
        )

        @app.get("/", include_in_schema=False)
        def index() -> HTMLResponse:
            return HTMLResponse(rendered_index)

    return app


app = create_app()
