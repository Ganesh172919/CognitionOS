"""
Tenant Provisioning Service — CognitionOS

Automated tenant lifecycle:
- Tenant creation with resource allocation
- Quota management per plan
- Tenant suspension / reactivation
- Usage tracking per tenant
- Tenant data export
- Customization (branding, features)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger(__name__)


class TenantPlan(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


class TenantStatus(str, Enum):
    PROVISIONING = "provisioning"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEPROVISIONING = "deprovisioning"
    ARCHIVED = "archived"


@dataclass
class TenantQuota:
    max_users: int = 5
    max_agents: int = 3
    max_api_calls_per_day: int = 1000
    max_storage_gb: float = 1.0
    max_executions_per_day: int = 100
    max_plugins: int = 5
    max_workflows: int = 10
    max_integrations: int = 3


PLAN_QUOTAS: Dict[TenantPlan, TenantQuota] = {
    TenantPlan.FREE: TenantQuota(5, 3, 1000, 1, 100, 5, 10, 3),
    TenantPlan.STARTER: TenantQuota(15, 10, 10000, 10, 1000, 20, 50, 10),
    TenantPlan.PRO: TenantQuota(50, 50, 100000, 100, 10000, 100, 200, 50),
    TenantPlan.BUSINESS: TenantQuota(200, 200, 500000, 500, 50000, 500, 1000, 200),
    TenantPlan.ENTERPRISE: TenantQuota(999999, 999999, 999999, 10000, 999999, 999999, 999999, 999999),
}


@dataclass
class Tenant:
    tenant_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    slug: str = ""
    plan: TenantPlan = TenantPlan.FREE
    status: TenantStatus = TenantStatus.PROVISIONING
    owner_user_id: str = ""
    quota: TenantQuota = field(default_factory=TenantQuota)
    features: Set[str] = field(default_factory=set)
    branding: Dict[str, str] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    activated_at: Optional[str] = None
    user_count: int = 0
    agent_count: int = 0
    storage_used_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id, "name": self.name,
            "slug": self.slug, "plan": self.plan.value,
            "status": self.status.value, "owner": self.owner_user_id,
            "user_count": self.user_count, "agent_count": self.agent_count,
            "storage_used_gb": self.storage_used_gb, "created_at": self.created_at}


class TenantProvisioningService:
    """Manages tenant lifecycle and resource allocation."""

    def __init__(self) -> None:
        self._tenants: Dict[str, Tenant] = {}
        self._metrics: Dict[str, int] = defaultdict(int)

    def provision(self, name: str, *, owner_user_id: str,
                   plan: TenantPlan = TenantPlan.FREE,
                   slug: str = "", branding: Dict[str, str] | None = None,
                   features: Set[str] | None = None) -> Tenant:
        tenant = Tenant(
            name=name, slug=slug or name.lower().replace(" ", "-"),
            plan=plan, owner_user_id=owner_user_id,
            quota=PLAN_QUOTAS.get(plan, TenantQuota()),
            features=features or set(), branding=branding or {})
        tenant.status = TenantStatus.ACTIVE
        tenant.activated_at = datetime.now(timezone.utc).isoformat()
        self._tenants[tenant.tenant_id] = tenant
        self._metrics["provisioned"] += 1
        logger.info("Tenant provisioned: %s [%s] plan=%s",
                     name, tenant.tenant_id, plan.value)
        return tenant

    def upgrade_plan(self, tenant_id: str, new_plan: TenantPlan) -> bool:
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False
        tenant.plan = new_plan
        tenant.quota = PLAN_QUOTAS.get(new_plan, tenant.quota)
        self._metrics["upgrades"] += 1
        return True

    def suspend(self, tenant_id: str, *, reason: str = "") -> bool:
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False
        tenant.status = TenantStatus.SUSPENDED
        self._metrics["suspensions"] += 1
        logger.warning("Tenant suspended: %s — %s", tenant_id, reason)
        return True

    def reactivate(self, tenant_id: str) -> bool:
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False
        tenant.status = TenantStatus.ACTIVE
        return True

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        return self._tenants.get(tenant_id)

    def list_tenants(self, *, plan: TenantPlan | None = None,
                      status: TenantStatus | None = None) -> List[Dict[str, Any]]:
        tenants = list(self._tenants.values())
        if plan:
            tenants = [t for t in tenants if t.plan == plan]
        if status:
            tenants = [t for t in tenants if t.status == status]
        return [t.to_dict() for t in tenants]

    def check_quota(self, tenant_id: str, resource: str, requested: int = 1) -> Dict[str, Any]:
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return {"allowed": False, "reason": "tenant_not_found"}
        quota = tenant.quota
        checks = {
            "users": (tenant.user_count + requested, quota.max_users),
            "agents": (tenant.agent_count + requested, quota.max_agents),
            "plugins": (0, quota.max_plugins),
            "workflows": (0, quota.max_workflows)}
        current, limit = checks.get(resource, (0, 999999))
        allowed = current <= limit
        return {"allowed": allowed, "current": current, "limit": limit,
                "resource": resource}

    def get_metrics(self) -> Dict[str, Any]:
        by_plan: Dict[str, int] = defaultdict(int)
        active = 0
        for t in self._tenants.values():
            by_plan[t.plan.value] += 1
            if t.status == TenantStatus.ACTIVE:
                active += 1
        return {**dict(self._metrics), "total_tenants": len(self._tenants),
                "active": active, "by_plan": dict(by_plan)}


_service: TenantProvisioningService | None = None

def get_tenant_provisioning() -> TenantProvisioningService:
    global _service
    if not _service:
        _service = TenantProvisioningService()
    return _service
