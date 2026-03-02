"""
Usage Metering Service — CognitionOS

Tracks granular resource usage for billing:
- API call counting per endpoint
- Token consumption tracking
- Storage usage
- Compute time tracking
- Aggregation by tenant/user/period
- Overage detection
- Usage reports
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class UsageType(str, Enum):
    API_CALL = "api_call"
    TOKEN_INPUT = "token_input"
    TOKEN_OUTPUT = "token_output"
    STORAGE_MB = "storage_mb"
    COMPUTE_SECONDS = "compute_seconds"
    AGENT_EXECUTION = "agent_execution"
    WEBHOOK_DELIVERY = "webhook_delivery"
    PLUGIN_EXECUTION = "plugin_execution"


@dataclass
class UsageRecord:
    record_id: str = field(default_factory=lambda: str(uuid4()))
    tenant_id: str = ""
    user_id: str = ""
    usage_type: UsageType = UsageType.API_CALL
    quantity: Decimal = Decimal("0")
    unit_price: Decimal = Decimal("0")
    total_cost: Decimal = Decimal("0")
    endpoint: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class UsageQuota:
    tenant_id: str
    quotas: Dict[UsageType, Decimal] = field(default_factory=dict)
    period: str = "monthly"  # daily, monthly

    def is_exceeded(self, usage_type: UsageType, current: Decimal) -> bool:
        limit = self.quotas.get(usage_type)
        if limit is None:
            return False
        return current > limit


# Default pricing (per unit)
DEFAULT_PRICING: Dict[UsageType, Decimal] = {
    UsageType.API_CALL: Decimal("0.001"),
    UsageType.TOKEN_INPUT: Decimal("0.00001"),
    UsageType.TOKEN_OUTPUT: Decimal("0.00003"),
    UsageType.STORAGE_MB: Decimal("0.02"),
    UsageType.COMPUTE_SECONDS: Decimal("0.0001"),
    UsageType.AGENT_EXECUTION: Decimal("0.01"),
    UsageType.WEBHOOK_DELIVERY: Decimal("0.0005"),
    UsageType.PLUGIN_EXECUTION: Decimal("0.002"),
}


class UsageMeteringService:
    """Tracks and aggregates resource usage for billing."""

    def __init__(self, *, pricing: Dict[UsageType, Decimal] | None = None) -> None:
        self._pricing = pricing or dict(DEFAULT_PRICING)
        self._records: List[UsageRecord] = []
        self._quotas: Dict[str, UsageQuota] = {}
        self._aggregated: Dict[str, Dict[UsageType, Decimal]] = defaultdict(
            lambda: defaultdict(Decimal))
        self._overage_alerts: List[Dict[str, Any]] = []

    # ---- recording ----
    def record(self, tenant_id: str, usage_type: UsageType, quantity: Decimal, *,
               user_id: str = "", endpoint: str = "",
               metadata: Dict[str, Any] | None = None) -> UsageRecord:
        unit_price = self._pricing.get(usage_type, Decimal("0"))
        total = quantity * unit_price

        record = UsageRecord(
            tenant_id=tenant_id, user_id=user_id, usage_type=usage_type,
            quantity=quantity, unit_price=unit_price, total_cost=total,
            endpoint=endpoint, metadata=metadata or {})
        self._records.append(record)
        self._aggregated[tenant_id][usage_type] += quantity

        # Check quota
        quota = self._quotas.get(tenant_id)
        if quota and quota.is_exceeded(usage_type, self._aggregated[tenant_id][usage_type]):
            self._overage_alerts.append({
                "tenant_id": tenant_id, "usage_type": usage_type.value,
                "current": float(self._aggregated[tenant_id][usage_type]),
                "limit": float(quota.quotas.get(usage_type, 0)),
                "timestamp": datetime.now(timezone.utc).isoformat()})
            logger.warning("Quota exceeded: %s/%s — %s",
                           tenant_id, usage_type.value,
                           self._aggregated[tenant_id][usage_type])

        return record

    # ---- quotas ----
    def set_quota(self, quota: UsageQuota) -> None:
        self._quotas[quota.tenant_id] = quota

    def get_quota(self, tenant_id: str) -> Optional[UsageQuota]:
        return self._quotas.get(tenant_id)

    # ---- aggregation ----
    def get_usage_summary(self, tenant_id: str, *, days: int = 30) -> Dict[str, Any]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        records = [r for r in self._records
                   if r.tenant_id == tenant_id and r.timestamp >= cutoff]

        by_type: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"quantity": 0, "cost": 0})
        total_cost = Decimal("0")

        for r in records:
            by_type[r.usage_type.value]["quantity"] += float(r.quantity)
            by_type[r.usage_type.value]["cost"] += float(r.total_cost)
            total_cost += r.total_cost

        return {
            "tenant_id": tenant_id, "period_days": days,
            "total_cost_usd": float(total_cost),
            "total_records": len(records),
            "by_type": dict(by_type),
            "daily_avg_cost": float(total_cost / max(1, days))}

    def get_usage_by_user(self, tenant_id: str) -> Dict[str, Dict[str, float]]:
        by_user: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "cost": 0})
        for r in self._records:
            if r.tenant_id == tenant_id and r.user_id:
                by_user[r.user_id]["calls"] += 1
                by_user[r.user_id]["cost"] += float(r.total_cost)
        return dict(by_user)

    def get_usage_by_endpoint(self, tenant_id: str, *, top_n: int = 20) -> List[Dict[str, Any]]:
        by_endpoint: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "cost": 0})
        for r in self._records:
            if r.tenant_id == tenant_id and r.endpoint:
                by_endpoint[r.endpoint]["calls"] += 1
                by_endpoint[r.endpoint]["cost"] += float(r.total_cost)
        sorted_eps = sorted(by_endpoint.items(), key=lambda x: -x[1]["calls"])
        return [{"endpoint": ep, **data} for ep, data in sorted_eps[:top_n]]

    # ---- billing invoice ----
    def generate_invoice(self, tenant_id: str, *, days: int = 30) -> Dict[str, Any]:
        summary = self.get_usage_summary(tenant_id, days=days)
        return {
            "invoice_id": str(uuid4())[:8],
            "tenant_id": tenant_id,
            "period_days": days,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "line_items": [
                {"type": t, "quantity": d["quantity"],
                 "unit_price": float(self._pricing.get(UsageType(t), 0)),
                 "total": d["cost"]}
                for t, d in summary["by_type"].items()],
            "subtotal_usd": summary["total_cost_usd"],
            "total_usd": summary["total_cost_usd"]}

    # ---- overage ----
    def get_overage_alerts(self, *, tenant_id: str = "") -> List[Dict[str, Any]]:
        alerts = self._overage_alerts
        if tenant_id:
            alerts = [a for a in alerts if a["tenant_id"] == tenant_id]
        return alerts[-50:]

    # ---- metrics ----
    def get_metrics(self) -> Dict[str, Any]:
        total_cost = sum(float(r.total_cost) for r in self._records)
        return {
            "total_records": len(self._records),
            "total_revenue_usd": round(total_cost, 4),
            "tenants_tracked": len(self._aggregated),
            "overage_alerts": len(self._overage_alerts)}


_service: UsageMeteringService | None = None

def get_usage_metering() -> UsageMeteringService:
    global _service
    if not _service:
        _service = UsageMeteringService()
    return _service
