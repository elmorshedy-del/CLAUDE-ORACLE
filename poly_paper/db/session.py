"""Async database session factory + init.

Uses sqlalchemy.ext.asyncio. Database URL from DATABASE_URL env var.

Local dev defaults to sqlite+aiosqlite:///./poly_paper.db.
Railway Postgres URLs come in the form postgresql://... — we rewrite to
postgresql+asyncpg:// for the async driver.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .models import Base


def _normalise_db_url(url: str) -> str:
    # Railway gives postgres:// or postgresql:// — async driver needs +asyncpg.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    if url.startswith("postgresql://") and "+asyncpg" not in url:
        url = "postgresql+asyncpg://" + url[len("postgresql://") :]
    if url.startswith("sqlite://") and "+aiosqlite" not in url:
        url = "sqlite+aiosqlite://" + url[len("sqlite://") :]
    return url


DATABASE_URL = _normalise_db_url(
    os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./poly_paper.db")
)

# Echo off by default; set DB_ECHO=1 for SQL debugging.
engine = create_async_engine(
    DATABASE_URL,
    echo=bool(os.environ.get("DB_ECHO")),
    future=True,
    # Pool sizes kept small for Railway; one runner process typically.
    pool_size=5,
    max_overflow=5,
)

SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db() -> None:
    """Create all tables if they don't exist. Idempotent."""
    # Import all models with table definitions before create_all runs, so
    # SQLAlchemy's Base.metadata picks them up. Modules are imported for
    # their side-effect of registering tables with Base.
    from . import models as _models  # noqa: F401
    from .. import weather_calibration as _wc  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def session() -> AsyncIterator[AsyncSession]:
    """Yield an async session. Caller is responsible for commit/rollback."""
    async with SessionLocal() as s:
        yield s
