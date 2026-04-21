"""
FastAPI application factory.

Usage:
    uvicorn web.app:app --reload --port 8000

Or from the project root:
    SECRET_KEY=<key> uvicorn web.app:app --reload
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from arxiv_lib.appdb import init_app_db
from arxiv_lib.config import SECRET_KEY
from web.limiter import limiter
from web.routers import admin, auth, groups, papers, recommendations, search, users


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.DEBUG)
    if not SECRET_KEY or len(SECRET_KEY) < 32:
        raise RuntimeError(
            "SECRET_KEY must be set to a random string of at least 32 characters. "
            "Generate one with: python3 -c \"import secrets; print(secrets.token_hex(32))\""
        )
    init_app_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="arXiv Recommender API",
        version="0.1.0",
        lifespan=lifespan,
        # Disable built-in API docs and schema endpoints. Leaving them enabled
        # gives attackers a free machine-readable description of every endpoint,
        # parameter, and auth requirement.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    _cors_origins_env = os.environ.get("CORS_ALLOW_ORIGINS", "")
    _cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    if not _cors_origins:
        # Development fallback — set CORS_ALLOW_ORIGINS in production
        _cors_origins = ["http://localhost:5173", "http://localhost:3000"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )

    app.include_router(auth.router,            prefix="/api")
    app.include_router(papers.router,          prefix="/api")
    app.include_router(users.router,           prefix="/api")
    app.include_router(recommendations.router, prefix="/api")
    app.include_router(search.router,          prefix="/api")
    app.include_router(admin.router,           prefix="/api")
    app.include_router(groups.router,          prefix="/api")

    # Serve the React SPA for all non-API routes. Only mounted when the dist/
    # directory exists, so local development without a built frontend still works.
    _dist = Path(__file__).parent / "frontend" / "dist"
    if _dist.exists():
        # Serve static assets directly; fall back to index.html for SPA routing.
        app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

        # Resolve once at startup so every request can use is_relative_to()
        # without an additional syscall.
        _dist_resolved = _dist.resolve()

        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            try:
                # Collapse any .. segments to a canonical absolute path.
                # resolve() calls stat() internally, which raises OSError if any
                # path component is too long (ENAMETOOLONG), inaccessible, etc.
                file_path = (_dist / full_path).resolve()
            except OSError:
                # Treat any filesystem error (e.g. overlong filename from a
                # scanner probe) as a non-existent path and serve the SPA shell.
                return FileResponse(_dist / "index.html")

            # Guard against path-traversal: only serve files that resolve to a
            # path still inside the dist directory.  is_file() confirms the path
            # is a regular file (not a directory or symlink to outside dist).
            if file_path.is_relative_to(_dist_resolved) and file_path.is_file():
                return FileResponse(file_path)

            # All other paths (unknown routes, SPA client-side routes) get the
            # React entry point so the frontend router can handle them.
            return FileResponse(_dist / "index.html")

    return app


app = create_app()
