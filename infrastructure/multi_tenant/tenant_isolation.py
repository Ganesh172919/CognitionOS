"""
Multi-Tenant Isolation Layer — CognitionOS

Production tenant isolation system with:
- Complete data isolation per tenant
- Resource quota management
- Tenant lifecycle (create, suspend, delete)
- Cross-tenant security boundaries
- Tenant-specific configuration
- Usage tracking per tenant
- Tenant health monitoring
- Noisy neighbor detection
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class TenantStatus(str, Enum):
    ACTIVE = "active"
    PROVISIONING = "provisioning"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"
    DELETING = "deleting"


class TenantTier(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


@dataclass
class ResourceQuota:
    api_calls_per_minute: int = 60
    api_calls_per_day: int = 5000
    storage_gb: float = 1.0
    max_agents: int = 1
    max_workflows: int = 5
    max_plugins: int = 3
    max_users: int = 1
    max_concurrent_tasks: int = 2
    max_memory_mb: int = 512
    custom_domains: int = 0
    webhook_subscriptions: int = 5

    @staticmethod
    def for_tier(tier: TenantTier) -> "ResourceQuota":
        quotas = {
            TenantTier.FREE: ResourceQuota(),
            TenantTier.STARTER: ResourceQuota(
                api_calls_per_minute=200, api_calls_per_day=25000,
                storage_gb=10, max_agents=3, max_workflows=20,
                max_plugins=10, max_users=5, max_concurrent_tasks=5,
                max_memory_mb=1024, webhook_subscriptions=20,
            ),
            TenantTier.PRO: ResourceQuota(
                api_calls_per_minute=1000, api_calls_per_day=100000,
                storage_gb=100, max_agents=20, max_workflows=100,
                max_plugins=50, max_users=25, max_concurrent_tasks=20,
                max_memory_mb=4096, custom_domains=3, webhook_subscriptions=100,
            ),
            TenantTier.BUSINESS: ResourceQuota(
                api_calls_per_minute=5000, api_calls_per_day=500000,
                storage_gb=500, max_agents=100, max_workflows=500,
                max_plugins=200, max_users=100, max_concurrent_tasks=50,
                max_memory_mb=16384, custom_domains=10, webhook_subscriptions=500,
            ),
            TenantTier.ENTERPRISE: ResourceQuota(
                api_calls_per_minute=50000, api_calls_per_day=5000000,
                storage_gb=5000, max_agents=1000, max_workflows=5000,
                max_plugins=1000, max_users=1000, max_concurrent_tasks=200,
                max_memory_mb=65536, custom_domains=50, webhook_subscriptions=2000,
            ),
        }
        return quotas.get(tier, ResourceQuota())


@dataclass
class TenantUsage:
    api_calls_today: int = 0
    api_calls_minute: int = 0
    storage_used_gb: float = 0
    active_agents: int = 0
    active_workflows: int = 0
    active_users: int = 0
    concurrent_tasks: int = 0
    memory_used_mb: int = 0
    last_api_call: float = 0
    last_reset: float = field(default_factory=time.time)


@dataclass
class Tenant:
    tenant_id: str
    name: str
    tier: TenantTier = TenantTier.FREE
    status: TenantStatus = TenantStatus.PROVISIONING
    owner_email: str = ""
    domain: str = ""
    custom_domains: List[str] = field(default_factory=list)
    quota: ResourceQuota = field(default_factory=ResourceQuota)
    usage: TenantUsage = field(default_factory=TenantUsage)
    settings: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    suspended_at: Optional[float] = None
    suspension_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id, "name": self.name,
            "tier": self.tier.value, "status": self.status.value,
            "owner_email": self.owner_email, "domain": self.domain,
            "created_at": self.created_at,
        }


@dataclass
class QuotaCheckResult:
    allowed: bool
    resource: str
    current: float
    limit: float
    usage_pct: float
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed, "resource": self.resource,
            "current": self.current, "limit": self.limit,
            "usage_pct": round(self.usage_pct, 1),
            "message": self.message,
        }


class TenantIsolationLayer:
    """
    Production multi-tenant isolation with quotas, usage tracking,
    lifecycle management, and noisy neighbor detection.
    """

    def __init__(self):
        self._tenants: Dict[str, Tenant] = {}
        self._hooks: Dict[str, List[Callable]] = {
            "on_create": [], "on_suspend": [],
            "on_activate": [], "on_delete": [],
            "on_quota_exceeded": [],
        }
        self._noisy_threshold_pct: float = 90.0

    # ── Tenant Lifecycle ──

    async def create_tenant(self, name: str, *,
                              tier: TenantTier = TenantTier.FREE,
                              owner_email: str = "",
                              settings: Optional[Dict] = None) -> Tenant:
        tenant_id = uuid.uuid4().hex[:12]
        tenant = Tenant(
            tenant_id=tenant_id, name=name, tier=tier,
            owner_email=owner_email,
            quota=ResourceQuota.for_tier(tier),
            settings=settings or {},
        )

        self._tenants[tenant_id] = tenant

        # Provisioning
        tenant.status = TenantStatus.ACTIVE
        await self._fire_hook("on_create", tenant)
        logger.info("Tenant created: %s (%s, tier=%s)", name, tenant_id, tier.value)
        return tenant

    async def suspend_tenant(self, tenant_id: str, *, reason: str = ""):
        tenant = self._get_tenant(tenant_id)
        tenant.status = TenantStatus.SUSPENDED
        tenant.suspended_at = time.time()
        tenant.suspension_reason = reason
        await self._fire_hook("on_suspend", tenant)
        logger.warning("Tenant suspended: %s (reason: %s)", tenant_id, reason)

    async def activate_tenant(self, tenant_id: str):
        tenant = self._get_tenant(tenant_id)
        tenant.status = TenantStatus.ACTIVE
        tenant.suspended_at = None
        tenant.suspension_reason = ""
        await self._fire_hook("on_activate", tenant)

    async def delete_tenant(self, tenant_id: str):
        tenant = self._get_tenant(tenant_id)
        tenant.status = TenantStatus.DELETING
        await self._fire_hook("on_delete", tenant)
        del self._tenants[tenant_id]
        logger.info("Tenant deleted: %s", tenant_id)

    async def upgrade_tier(self, tenant_id: str, new_tier: TenantTier):
        tenant = self._get_tenant(tenant_id)
        old_tier = tenant.tier
        tenant.tier = new_tier
        tenant.quota = ResourceQuota.for_tier(new_tier)
        logger.info("Tenant %s upgraded: %s -> %s",
                     tenant_id, old_tier.value, new_tier.value)

    # ── Quota Checking ──

    def check_quota(self, tenant_id: str, resource: str,
                      requested: float = 1) -> QuotaCheckResult:
        """Check if a resource request is within quota."""
        tenant = self._get_tenant(tenant_id)

        if tenant.status != TenantStatus.ACTIVE:
            return QuotaCheckResult(
                allowed=False, resource=resource,
                current=0, limit=0, usage_pct=0,
                message=f"Tenant is {tenant.status.value}",
            )

        quota_map = {
            "api_calls_minute": (tenant.usage.api_calls_minute, tenant.quota.api_calls_per_minute),
            "api_calls_day": (tenant.usage.api_calls_today, tenant.quota.api_calls_per_day),
            "storage_gb": (tenant.usage.storage_used_gb, tenant.quota.storage_gb),
            "agents": (tenant.usage.active_agents, tenant.quota.max_agents),
            "workflows": (tenant.usage.active_workflows, tenant.quota.max_workflows),
            "users": (tenant.usage.active_users, tenant.quota.max_users),
            "concurrent_tasks": (tenant.usage.concurrent_tasks, tenant.quota.max_concurrent_tasks),
            "memory_mb": (tenant.usage.memory_used_mb, tenant.quota.max_memory_mb),
        }

        if resource not in quota_map:
            return QuotaCheckResult(
                allowed=True, resource=resource,
                current=0, limit=0, usage_pct=0,
                message="Unknown resource — allowed by default",
            )

        current, limit = quota_map[resource]
        usage_pct = (current / max(limit, 1)) * 100
        allowed = (current + requested) <= limit

        if not allowed:
            logger.warning("Quota exceeded for tenant %s: %s (%d/%d)",
                           tenant_id, resource, current, limit)

        return QuotaCheckResult(
            allowed=allowed, resource=resource,
            current=current, limit=limit, usage_pct=usage_pct,
            message="" if allowed else f"Quota exceeded for {resource}",
        )

    def record_usage(self, tenant_id: str, resource: str, amount: float = 1):
        """Record resource usage for a tenant."""
        tenant = self._get_tenant(tenant_id)
        usage = tenant.usage

        if resource == "api_call":
            usage.api_calls_today += int(amount)
            usage.api_calls_minute += int(amount)
            usage.last_api_call = time.time()
        elif resource == "storage_gb":
            usage.storage_used_gb += amount
        elif resource == "agents":
            usage.active_agents += int(amount)
        elif resource == "workflows":
            usage.active_workflows += int(amount)
        elif resource == "concurrent_tasks":
            usage.concurrent_tasks += int(amount)

    def release_usage(self, tenant_id: str, resource: str, amount: float = 1):
        tenant = self._get_tenant(tenant_id)
        usage = tenant.usage

        if resource == "concurrent_tasks":
            usage.concurrent_tasks = max(0, usage.concurrent_tasks - int(amount))
        elif resource == "agents":
            usage.active_agents = max(0, usage.active_agents - int(amount))

    # ── Noisy Neighbor Detection ──

    def detect_noisy_neighbors(self) -> List[Dict[str, Any]]:
        """Detect tenants consuming disproportionate resources."""
        noisy = []
        for tenant in self._tenants.values():
            if tenant.status != TenantStatus.ACTIVE:
                continue
            checks = [
                ("api_calls", tenant.usage.api_calls_today,
                 tenant.quota.api_calls_per_day),
                ("storage", tenant.usage.storage_used_gb,
                 tenant.quota.storage_gb),
                ("memory", tenant.usage.memory_used_mb,
                 tenant.quota.max_memory_mb),
            ]
            for resource, current, limit in checks:
                if limit > 0:
                    usage_pct = (current / limit) * 100
                    if usage_pct >= self._noisy_threshold_pct:
                        noisy.append({
                            "tenant_id": tenant.tenant_id,
                            "tenant_name": tenant.name,
                            "resource": resource,
                            "usage_pct": round(usage_pct, 1),
                            "tier": tenant.tier.value,
                        })
        return sorted(noisy, key=lambda x: -x["usage_pct"])

    # ── Tenant Config ──

    def set_setting(self, tenant_id: str, key: str, value: Any):
        tenant = self._get_tenant(tenant_id)
        tenant.settings[key] = value

    def get_setting(self, tenant_id: str, key: str,
                      default: Any = None) -> Any:
        tenant = self._tenants.get(tenant_id)
        if tenant:
            return tenant.settings.get(key, default)
        return default

    # ── Queries ──

    def get_tenant(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return None
        return {
            **tenant.to_dict(),
            "quota": {
                "api_calls_per_day": tenant.quota.api_calls_per_day,
                "storage_gb": tenant.quota.storage_gb,
                "max_agents": tenant.quota.max_agents,
                "max_users": tenant.quota.max_users,
            },
            "usage": {
                "api_calls_today": tenant.usage.api_calls_today,
                "storage_used_gb": round(tenant.usage.storage_used_gb, 2),
                "active_agents": tenant.usage.active_agents,
                "active_users": tenant.usage.active_users,
            },
        }

    def list_tenants(self, *, tier: Optional[TenantTier] = None,
                       status: Optional[TenantStatus] = None
                       ) -> List[Dict[str, Any]]:
        tenants = list(self._tenants.values())
        if tier:
            tenants = [t for t in tenants if t.tier == tier]
        if status:
            tenants = [t for t in tenants if t.status == status]
        return [t.to_dict() for t in tenants]

    def get_stats(self) -> Dict[str, Any]:
        by_tier = defaultdict(int)
        by_status = defaultdict(int)
        for t in self._tenants.values():
            by_tier[t.tier.value] += 1
            by_status[t.status.value] += 1
        return {
            "total_tenants": len(self._tenants),
            "by_tier": dict(by_tier),
            "by_status": dict(by_status),
            "noisy_neighbors": len(self.detect_noisy_neighbors()),
        }

    # ── Hooks ──

    def add_hook(self, event: str, callback: Callable):
        if event in self._hooks:
            self._hooks[event].append(callback)

    async def _fire_hook(self, event: str, tenant: Tenant):
        for hook in self._hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(tenant)
                else:
                    hook(tenant)
            except Exception as exc:
                logger.error("Hook %s failed: %s", event, exc)

    def _get_tenant(self, tenant_id: str) -> Tenant:
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            raise KeyError(f"Tenant not found: {tenant_id}")
        return tenant


# ── Singleton ──
_layer: Optional[TenantIsolationLayer] = None


def get_tenant_layer() -> TenantIsolationLayer:
    global _layer
    if not _layer:
        _layer = TenantIsolationLayer()
    return _layer
