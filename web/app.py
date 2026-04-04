"""
FastAPI application factory.

Usage:
    uvicorn web.app:app --reload --port 8000

Or from the project root:
    SECRET_KEY=<key> uvicorn web.app:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from arxiv_lib.appdb import init_app_db
from arxiv_lib.config import SECRET_KEY
from web.routers import admin, auth, papers, recommendations, users

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.DEBUG)
    if not SECRET_KEY:
        import warnings
        warnings.warn(
            "SECRET_KEY is not set. JWTs will be signed with an empty key — "
            "do not run this in production without setting SECRET_KEY.",
            stacklevel=2,
        )
    init_app_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="arXiv Recommender API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",  # Vite dev server
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router,            prefix="/api")
    app.include_router(papers.router,          prefix="/api")
    app.include_router(users.router,           prefix="/api")
    app.include_router(recommendations.router, prefix="/api")
    app.include_router(admin.router,           prefix="/api")

    return app


app = create_app()
