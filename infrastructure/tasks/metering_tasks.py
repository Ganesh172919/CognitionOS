"""
Usage Metering Async Tasks.

Flush buffered Redis usage aggregates into durable `usage_records`.

This is a pragmatic, production-safe bridge until a dedicated event stream
(Kafka/Redpanda) + OLAP sink is introduced. It avoids per-request DB writes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from infrastructure.tasks.celery_config import celery_app


logger = logging.getLogger(__name__)


async def _flush_api_call_usage_async() -> dict:
    from services.api.src.dependencies.injection import async_session_factory, get_engine

    # Ensure engine + session factory are initialized in this worker process.
    get_engine()

    from infrastructure.persistence.redis_pool import get_redis_client
    from infrastructure.persistence.billing_repository import PostgreSQLUsageRecordRepository
    from core.domain.billing.entities import UsageRecord

    redis = await get_redis_client()

    now = int(time.time())
    current_window_start = now - (now % 60)
    flush_before_window = current_window_start - 60  # only flush completed windows

    processed_keys = 0
    persisted_records = 0
    skipped_keys = 0
    errors = 0

    pattern = "usage:api_calls:tenant:*"
    cursor = 0

    async with async_session_factory() as session:
        usage_repo = PostgreSQLUsageRecordRepository(session)

        while True:
            cursor, keys = await redis.scan(cursor=cursor, match=pattern, count=500)
            for key in keys:
                processed_keys += 1
                try:
                    parts = str(key).split(":")
                    if len(parts) != 5:
                        skipped_keys += 1
                        continue

                    tenant_id = parts[3]
                    window_start = int(parts[4])
                    if window_start > flush_before_window:
                        continue

                    # Atomic read+delete (prefer GETDEL when available).
                    value = None
                    try:
                        value = await redis.getdel(key)
                    except Exception:
                        value = await redis.get(key)
                        if value is not None:
                            await redis.delete(key)

                    if value is None:
                        continue

                    count = int(value)
                    if count <= 0:
                        continue

                    await usage_repo.create(
                        UsageRecord.create(
                            tenant_id=UUID(tenant_id),
                            resource_type="api_calls",
                            quantity=Decimal(str(count)),
                            unit="count",
                            event_id=f"api_calls:{window_start}",
                            metadata={
                                "window_start": window_start,
                                "window_duration_seconds": 60,
                                "source": "redis_aggregate",
                                "timestamp": datetime.utcfromtimestamp(window_start).isoformat(),
                            },
                        )
                    )
                    persisted_records += 1
                except Exception:  # noqa: BLE001
                    errors += 1
                    logger.exception("Failed to flush api_call usage key", extra={"key": str(key)})

            if int(cursor) == 0:
                break

        try:
            await session.commit()
        except Exception:  # noqa: BLE001
            await session.rollback()
            raise

    return {
        "processed_keys": processed_keys,
        "persisted_records": persisted_records,
        "skipped_keys": skipped_keys,
        "errors": errors,
        "flushed_before_window_start": flush_before_window,
    }


@celery_app.task(name="infrastructure.tasks.metering_tasks.flush_api_call_usage", bind=True)
def flush_api_call_usage(self) -> dict:
    """Flush tenant-scoped API call usage aggregates (Redis -> Postgres)."""
    try:
        return asyncio.run(_flush_api_call_usage_async())
    except Exception as exc:  # noqa: BLE001
        logger.error("Usage flush task failed", extra={"error": str(exc)})
        raise

