"""
Admin Dashboard API - Internal metrics and dashboard endpoints.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.src.dependencies.injection import get_db_session
from infrastructure.persistence.billing_repository import (
    PostgreSQLSubscriptionRepository,
    PostgreSQLUsageRecordRepository,
)
from infrastructure.billing.revenue_analytics import RevenueAnalytics

router = APIRouter(prefix="/api/admin", tags=["Admin Dashboard"])


class DashboardMetrics(BaseModel):
    """Dashboard metrics response."""

    mrr: Decimal
    active_subscriptions: int
    total_tenants: int
    usage_executions_today: int
    usage_tokens_today: int
    period_start: datetime
    period_end: datetime


@router.get("/dashboard/metrics", response_model=DashboardMetrics)
async def get_dashboard_metrics(
    session: AsyncSession = Depends(get_db_session),
) -> DashboardMetrics:
    """Get admin dashboard metrics."""
    sub_repo = PostgreSQLSubscriptionRepository(session)
    usage_repo = PostgreSQLUsageRecordRepository(session)
    analytics = RevenueAnalytics(sub_repo, usage_repo)
    now = datetime.utcnow()
    today_start = datetime(now.year, now.month, now.day)
    mrr = await analytics.get_mrr()
    active_subs = await sub_repo.get_active_subscriptions(limit=10000)
    executions_today = Decimal("0")
    tokens_today = Decimal("0")
    for tid in tenant_ids:
        e = await usage_repo.aggregate_usage(
            tenant_id=tid,
            resource_type="executions",
            start_time=today_start,
            end_time=now,
        )
        t = await usage_repo.aggregate_usage(
            tenant_id=tid,
            resource_type="tokens",
            start_time=today_start,
            end_time=now,
        )
        executions_today += (e or Decimal("0"))
        tokens_today += (t or Decimal("0"))
    tenant_ids = {s.tenant_id for s in active_subs}
    return DashboardMetrics(
        mrr=mrr,
        active_subscriptions=len(active_subs),
        total_tenants=len(tenant_ids),
        usage_executions_today=int(executions_today or 0),
        usage_tokens_today=int(tokens_today or 0),
        period_start=today_start,
        period_end=now,
    )
