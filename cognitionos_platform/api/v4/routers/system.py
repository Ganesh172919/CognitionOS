"""
System-level v4 endpoints (health, build info).

These endpoints are safe to expose without tenant context and are intended
for self-host operational monitoring.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

from core.config import get_config
from services.api.src.dependencies.injection import (
    check_database_health,
    check_redis_health,
    check_rabbitmq_health,
)


router = APIRouter(prefix="/api/v4", tags=["System (v4)"])
config = get_config()


@router.get("/health/live", summary="v4 liveness probe")
async def live() -> dict:
    return {
        "status": "alive",
        "service": config.service_name,
        "version": config.service_version,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health/ready", summary="v4 readiness probe")
async def ready() -> dict:
    db_ok = await check_database_health()
    redis_ok = await check_redis_health()
    rabbit_ok = await check_rabbitmq_health()
    return {
        "status": "ready" if (db_ok and rabbit_ok) else "not_ready",
        "timestamp": datetime.utcnow().isoformat(),
        "dependencies": {
            "database": "healthy" if db_ok else "unhealthy",
            "redis": "healthy" if redis_ok else "degraded",
            "rabbitmq": "healthy" if rabbit_ok else "unhealthy",
        },
    }
