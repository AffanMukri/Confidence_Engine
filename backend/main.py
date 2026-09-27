"""
Confidence Scoring Engine — FastAPI Application Entry Point.

Start with::

    uvicorn main:app --reload --host 127.0.0.1 --port 8765

Or programmatically::

    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8765, reload=True)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import config
from db import init_db, close_db, use_sqlite_fallback, get_active_database_url

# Ensure the ORM model is imported so Base.metadata knows about it
from models.session import ConfidenceSession  # noqa: F401

logger = logging.getLogger("confidence_engine")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables on startup, close pool on shutdown."""
    logger.info("Starting Confidence Scoring Engine …")
    try:
        await init_db()
        logger.info("Database tables ready (%s)", get_active_database_url())
    except Exception as exc:
        logger.warning(
            "Could not connect to database at %s (%s). Trying SQLite fallback.",
            config.DATABASE_URL,
            exc,
        )
        try:
            await use_sqlite_fallback()
            await init_db()
            logger.info("Database tables ready (%s)", get_active_database_url())
        except Exception:
            logger.exception(
                "Database initialization failed — session persistence disabled."
            )
    yield
    await close_db()
    logger.info("Confidence Scoring Engine shut down")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Confidence Scoring Engine",
    description=(
        "Real-time interview confidence analysis via MediaPipe. "
        "Send webcam frames over WebSocket, receive a composite 0-100 score."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---- CORS ----
# Permissive for dev (file:// and localhost origins).
# Tighten ALLOWED_ORIGINS via env var for production:
#   ALLOWED_ORIGINS=https://uhired.in,https://app.uhired.in
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Routers ----
from ws.confidence_ws import router as ws_router  # noqa: E402
from api.sessions import router as api_router  # noqa: E402

app.include_router(ws_router)
app.include_router(api_router)


# ---- Health check ----
@app.get("/health", tags=["system"])
async def health():
    """Simple liveness probe."""
    return {"status": "ok", "service": "confidence-engine"}


# ---- Demo frontend (served from same origin as the API/WebSocket) ----
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(_frontend_dir), html=True),
        name="frontend",
    )
