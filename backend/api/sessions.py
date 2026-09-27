"""
REST API endpoints for retrieving persisted confidence session summaries.

Endpoints
---------
GET /api/sessions/                  List recent sessions (paginated)
GET /api/sessions/{session_id}      Retrieve a single session summary
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select, desc

import db
from models.session import ConfidenceSession

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("/")
async def list_sessions(
    limit: int = Query(20, ge=1, le=100, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
):
    """List recent confidence-scoring sessions, newest first."""
    async with db.AsyncSessionLocal() as session:
        stmt = (
            select(ConfidenceSession)
            .order_by(desc(ConfidenceSession.created_at))
            .offset(offset)
            .limit(limit)
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()
        return {
            "sessions": [row.to_dict() for row in rows],
            "limit": limit,
            "offset": offset,
            "count": len(rows),
        }


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Retrieve a persisted session summary by its session_id."""
    async with db.AsyncSessionLocal() as session:
        stmt = select(ConfidenceSession).where(
            ConfidenceSession.session_id == session_id
        )
        result = await session.execute(stmt)
        row: Optional[ConfidenceSession] = result.scalar_one_or_none()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found",
            )
        return row.to_dict()
