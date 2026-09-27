"""
Async database engine and session factory.

Uses PostgreSQL by default.  If the configured URL is unreachable at startup,
falls back to a local SQLite file so scoring and session persistence still work
during development.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

import config

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


def _sqlite_fallback_url() -> str:
    db_path = Path(__file__).resolve().parent / "confidence.db"
    return f"sqlite+aiosqlite:///{db_path.as_posix()}"


def _make_engine(url: str):
    connect_args = {}
    engine_kwargs = {
        "echo": False,
        "pool_pre_ping": True,
    }
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        engine_kwargs["connect_args"] = connect_args
    else:
        engine_kwargs["pool_size"] = 5
        engine_kwargs["max_overflow"] = 10

    return create_async_engine(url, **engine_kwargs)


async_engine = _make_engine(config.DATABASE_URL)
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
_active_database_url = config.DATABASE_URL


def get_active_database_url() -> str:
    return _active_database_url


async def use_sqlite_fallback() -> None:
    """Re-point the async engine to the local SQLite fallback database."""
    global async_engine, AsyncSessionLocal, _active_database_url

    fallback_url = config.SQLITE_FALLBACK_URL or _sqlite_fallback_url()
    logger.warning(
        "PostgreSQL unavailable — using SQLite fallback at %s", fallback_url
    )

    await async_engine.dispose()
    async_engine = _make_engine(fallback_url)
    AsyncSessionLocal = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    _active_database_url = fallback_url


async def init_db() -> None:
    """Create all tables that don't already exist."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose of the connection pool."""
    await async_engine.dispose()
