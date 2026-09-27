"""
SQLAlchemy ORM model for the ``confidence_sessions`` table.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, Float, Integer, JSON, String, Uuid, func

from db import Base


class ConfidenceSession(Base):
    """Persisted summary of a single confidence-scoring session."""

    __tablename__ = "confidence_sessions"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="Opaque ID supplied by the caller (e.g. the interviewer app).",
    )
    average_score = Column(Float, nullable=False)
    min_score = Column(Float)
    max_score = Column(Float)
    score_timeline = Column(JSON, default=list)
    low_confidence = Column(JSON, default=list)
    total_frames = Column(Integer)
    duration_seconds = Column(Float)
    signals_summary = Column(JSON, default=dict)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<ConfidenceSession(session_id={self.session_id!r}, "
            f"avg={self.average_score:.1f})>"
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "id": str(self.id),
            "session_id": self.session_id,
            "average_score": self.average_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "score_timeline": self.score_timeline,
            "low_confidence": self.low_confidence,
            "total_frames": self.total_frames,
            "duration_seconds": self.duration_seconds,
            "signals_summary": self.signals_summary,
            "created_at": (
                self.created_at.isoformat() if self.created_at else None
            ),
        }
