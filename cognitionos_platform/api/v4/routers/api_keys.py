"""
API Keys v4 API.

Tenant-scoped API key management:
- Create keys (returns the raw key only once)
- List keys for a tenant
- Revoke keys
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.middleware.api_key_auth import generate_api_key
from infrastructure.persistence.api_key_repository import PostgresAPIKeyRepository
from services.api.src.auth.dependencies import CurrentUser, get_current_user
from services.api.src.dependencies.injection import get_db_session


router = APIRouter(prefix="/api/v4/api-keys", tags=["API Keys (v4)"])


def _safe_uuid(value: str) -> Optional[UUID]:
    try:
        return UUID(str(value))
    except Exception:
        return None


def _require_tenant_id(request: Request) -> UUID:
    tenant_id = getattr(request.state, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context required (provide X-Tenant-Slug).",
        )
    return tenant_id


def _require_owner_or_admin(request: Request, current_user: CurrentUser) -> None:
    if current_user.has_role("admin"):
        return
    tenant = getattr(request.state, "tenant", None)
    user_uuid = _safe_uuid(current_user.user_id)
    if not tenant or not user_uuid or not getattr(tenant, "owner_user_id", None):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this tenant.")
    if tenant.owner_user_id != user_uuid:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this tenant.")


class CreateAPIKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    scopes: List[str] = Field(default_factory=list)
    rate_limit_per_minute: int = Field(default=60, ge=0, le=100000)
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=3650)


class APIKeyResponse(BaseModel):
    id: UUID
    name: str
    key_prefix: str
    scopes: List[str]
    rate_limit_per_minute: int
    is_active: bool
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: datetime
    created_by: Optional[UUID] = None


class CreatedAPIKeyResponse(APIKeyResponse):
    api_key: str = Field(..., description="Raw API key (only returned once).")


class RevokeResponse(BaseModel):
    success: bool = True


@router.post("", response_model=CreatedAPIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: Request,
    payload: CreateAPIKeyRequest,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> CreatedAPIKeyResponse:
    tenant_id = _require_tenant_id(request)
    _require_owner_or_admin(request, current_user)

    raw_key = generate_api_key()
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_prefix = raw_key[4:12]

    expires_at = None
    if payload.expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=int(payload.expires_in_days))

    created_by = _safe_uuid(current_user.user_id)

    repo = PostgresAPIKeyRepository(session)
    api_key_model = await repo.create(
        tenant_id=tenant_id,
        name=payload.name,
        key_hash=key_hash,
        key_prefix=key_prefix,
        scopes=payload.scopes,
        rate_limit_per_minute=int(payload.rate_limit_per_minute or 0),
        expires_at=expires_at,
        created_by=created_by,
    )
    await session.commit()

    return CreatedAPIKeyResponse(
        id=api_key_model.id,
        name=api_key_model.name,
        key_prefix=api_key_model.key_prefix,
        scopes=list(api_key_model.scopes or []),
        rate_limit_per_minute=int(api_key_model.rate_limit_per_minute or 0),
        is_active=bool(api_key_model.is_active),
        last_used_at=api_key_model.last_used_at,
        expires_at=api_key_model.expires_at,
        created_at=api_key_model.created_at,
        created_by=api_key_model.created_by,
        api_key=raw_key,
    )


@router.get("", response_model=List[APIKeyResponse])
async def list_api_keys(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    include_inactive: bool = False,
    limit: int = 100,
) -> List[APIKeyResponse]:
    tenant_id = _require_tenant_id(request)
    _require_owner_or_admin(request, current_user)

    repo = PostgresAPIKeyRepository(session)
    keys = await repo.list_by_tenant(
        tenant_id=tenant_id,
        include_inactive=bool(include_inactive),
        limit=max(1, min(500, int(limit))),
    )

    return [
        APIKeyResponse(
            id=k.id,
            name=k.name,
            key_prefix=k.key_prefix,
            scopes=list(k.scopes or []),
            rate_limit_per_minute=int(k.rate_limit_per_minute or 0),
            is_active=bool(k.is_active),
            last_used_at=k.last_used_at,
            expires_at=k.expires_at,
            created_at=k.created_at,
            created_by=k.created_by,
        )
        for k in keys
    ]


@router.delete("/{api_key_id}", response_model=RevokeResponse)
async def revoke_api_key(
    request: Request,
    api_key_id: UUID,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> RevokeResponse:
    tenant_id = _require_tenant_id(request)
    _require_owner_or_admin(request, current_user)

    repo = PostgresAPIKeyRepository(session)
    ok = await repo.revoke(api_key_id, tenant_id=tenant_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found.")
    await session.commit()
    return RevokeResponse(success=True)

