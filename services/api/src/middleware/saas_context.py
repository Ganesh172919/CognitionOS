"""
SaaS Context Middleware Helpers (API Service)

Provides small repository proxies to allow Starlette/FastAPI middleware to
resolve tenant and API key context without relying on FastAPI dependency injection.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from services.api.src.dependencies.injection import async_session_factory, get_engine
from infrastructure.persistence.api_key_repository import PostgresAPIKeyRepository
from infrastructure.persistence.tenant_repository import PostgreSQLTenantRepository


@dataclass(slots=True)
class TenantRepositoryProxy:
    """Repository proxy that creates a fresh DB session per call."""

    async def get_by_id(self, tenant_id: UUID):
        get_engine()
        async with async_session_factory() as session:
            return await PostgreSQLTenantRepository(session).get_by_id(tenant_id)

    async def get_by_slug(self, slug: str):
        get_engine()
        async with async_session_factory() as session:
            return await PostgreSQLTenantRepository(session).get_by_slug(slug)


@dataclass(slots=True)
class APIKeyRepositoryProxy:
    """Repository proxy that creates a fresh DB session per call."""

    async def get_by_hash(self, key_hash: str):
        get_engine()
        async with async_session_factory() as session:
            return await PostgresAPIKeyRepository(session).get_by_hash(key_hash)

    async def update_last_used(self, api_key_id: UUID) -> None:
        get_engine()
        async with async_session_factory() as session:
            await PostgresAPIKeyRepository(session).update_last_used(api_key_id)
            await session.commit()
