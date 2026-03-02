"""Unified health checks for the platform runtime."""

from __future__ import annotations

from typing import Any, Dict

from services.api.src.dependencies.injection import (
    check_database_health,
    check_redis_health,
    check_rabbitmq_health,
)


async def check_system_health() -> Dict[str, Any]:
    db_ok = await check_database_health()
    redis_ok = await check_redis_health()
    rabbit_ok = await check_rabbitmq_health()
    return {
        "database": db_ok,
        "redis": redis_ok,
        "rabbitmq": rabbit_ok,
        "ok": db_ok and rabbit_ok,
    }
