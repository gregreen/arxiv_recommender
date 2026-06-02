"""
Analytics endpoints.

POST /api/analytics/event  — record a page visit (auth required)

Page paths are normalised before storage: numeric and UUID path segments are
replaced with [id] so /groups/42/manage and /groups/99/manage both map to the
canonical /groups/[id]/manage.  This prevents the pages table from exploding
with per-entity rows while still capturing which feature was used.
"""

import re
import sqlite3

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, field_validator

from web.dependencies import get_current_user, get_db
from web.limiter import limiter

router = APIRouter(prefix="/analytics", tags=["analytics"])

# Matches a pure integer or a UUID-shaped segment (hex groups separated by hyphens).
_ID_SEGMENT = re.compile(
    r"^(?:\d+|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)


def _normalise_page(path: str) -> str:
    """Replace numeric / UUID path segments with [id]."""
    parts = path.split("/")
    return "/".join("[id]" if _ID_SEGMENT.match(p) else p for p in parts)


class EventRequest(BaseModel):
    page: str

    @field_validator("page")
    @classmethod
    def validate_page(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("page must not be empty")
        if len(v) > 200:
            raise ValueError("page must be 200 characters or fewer")
        return v


@router.post("/event", status_code=204)
@limiter.limit("60/minute")
def record_event(
    request: Request,
    body: EventRequest,
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    page = _normalise_page(body.page)
    db.execute(
        "INSERT INTO page_events (user_id, page) VALUES (?, ?)",
        (user["id"], page),
    )
    db.commit()
