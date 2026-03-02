"""
Revenue Analytics - MRR, churn, LTV aggregates.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID


@dataclass
class RevenueMetrics:
    """Revenue metrics for a period."""

    mrr: Decimal
    arr: Decimal
    churn_rate: float
    new_mrr: Decimal
    expansion_mrr: Decimal
    contraction_mrr: Decimal
    period_start: datetime
    period_end: datetime


@dataclass
class TenantLTV:
    """LTV metrics for a tenant."""

    tenant_id: UUID
    ltv: Decimal
    avg_revenue_per_month: Decimal
    months_active: int
    predicted_ltv: Decimal


class RevenueAnalytics:
    """
    Revenue analytics for billing.
    Aggregates MRR, churn, LTV from subscription and usage data.
    """

    def __init__(self, subscription_repository, usage_repository):
        self._sub_repo = subscription_repository
        self._usage_repo = usage_repository

    async def get_mrr(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> Decimal:
        """Compute MRR from active subscriptions."""
        subs = await self._sub_repo.get_active_subscriptions(limit=10000)
        mrr = Decimal("0")
        for s in subs:
            if s.amount_cents:
                monthly = self._to_monthly(s.amount_cents, s.billing_cycle or "monthly")
                mrr += monthly
        return mrr

    def _to_monthly(self, amount_cents: int, billing_cycle: str) -> Decimal:
        if billing_cycle == "yearly":
            return Decimal(amount_cents) / Decimal(100) / Decimal(12)
        return Decimal(amount_cents) / Decimal(100)

    async def get_churn_rate(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> float:
        """Compute churn rate (fraction 0-1). Uses canceled vs active ratio."""
        from core.domain.billing.entities import SubscriptionStatus

        active = await self._sub_repo.get_active_subscriptions(limit=10000)
        canceled = await self._sub_repo.get_by_status(SubscriptionStatus.CANCELED, limit=1000)
        total = len(active) + len(canceled)
        if total <= 0:
            return 0.0
        return len(canceled) / total

    async def get_ltv(self, tenant_id: UUID) -> Optional[TenantLTV]:
        """Estimate LTV for a tenant."""
        sub = await self._sub_repo.get_by_tenant(tenant_id)
        if sub is None:
            return None
        amount = Decimal(sub.amount_cents or 0) / Decimal(100)
        monthly = self._to_monthly(sub.amount_cents or 0, sub.billing_cycle or "monthly")
        start = sub.current_period_start or sub.created_at
        months = max(1, (datetime.utcnow() - start).days / 30)
        avg_revenue = (amount * Decimal(months)) / Decimal(str(months))
        predicted_months = 12
        ltv = monthly * Decimal(str(predicted_months))
        return TenantLTV(
            tenant_id=tenant_id,
            ltv=ltv,
            avg_revenue_per_month=avg_revenue,
            months_active=int(months),
            predicted_ltv=ltv,
        )
