"""
Subscription Management Engine

Full lifecycle subscription management with tier enforcement,
usage-based billing integration, trial management, upgrade/downgrade
workflows, and revenue tracking.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SubscriptionTier(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class SubscriptionStatus(str, Enum):
    TRIALING = "trialing"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    PAUSED = "paused"
    CANCELED = "canceled"
    EXPIRED = "expired"
    SUSPENDED = "suspended"


class BillingCycle(str, Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    CUSTOM = "custom"


class ChangeType(str, Enum):
    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"
    CANCEL = "cancel"
    REACTIVATE = "reactivate"
    PAUSE = "pause"
    RESUME = "resume"


@dataclass
class TierLimits:
    """Resource limits for a subscription tier."""
    max_agents: int = 1
    max_tasks_per_day: int = 50
    max_workflows: int = 5
    max_plugins: int = 3
    max_api_calls_per_day: int = 1_000
    max_tokens_per_month: int = 100_000
    max_storage_mb: int = 500
    max_team_members: int = 1
    max_concurrent_executions: int = 1
    code_generation_enabled: bool = False
    advanced_analytics: bool = False
    custom_models: bool = False
    priority_support: bool = False
    sso_enabled: bool = False
    audit_logs: bool = False
    custom_branding: bool = False
    dedicated_resources: bool = False
    sla_guarantee: bool = False


@dataclass
class TierPricing:
    """Pricing for a subscription tier."""
    tier: SubscriptionTier
    monthly_price_usd: float
    annual_price_usd: float
    overage_price_per_token: float = 0.0
    overage_price_per_api_call: float = 0.0
    setup_fee: float = 0.0
    limits: TierLimits = field(default_factory=TierLimits)
    features: List[str] = field(default_factory=list)
    trial_days: int = 0


@dataclass
class Subscription:
    """A tenant subscription record."""
    subscription_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    user_id: str = ""
    tier: SubscriptionTier = SubscriptionTier.FREE
    status: SubscriptionStatus = SubscriptionStatus.ACTIVE
    billing_cycle: BillingCycle = BillingCycle.MONTHLY
    stripe_subscription_id: Optional[str] = None
    stripe_customer_id: Optional[str] = None

    # Pricing
    monthly_price: float = 0.0
    discount_percent: float = 0.0
    coupon_code: Optional[str] = None

    # Dates
    created_at: datetime = field(default_factory=datetime.utcnow)
    trial_start: Optional[datetime] = None
    trial_end: Optional[datetime] = None
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    cancel_at_period_end: bool = False

    # Usage
    tokens_used_this_period: int = 0
    api_calls_this_period: int = 0
    storage_used_mb: float = 0.0

    # Custom limits override
    custom_limits: Optional[TierLimits] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status in (SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING)

    @property
    def is_trialing(self) -> bool:
        return self.status == SubscriptionStatus.TRIALING

    @property
    def trial_days_remaining(self) -> int:
        if self.trial_end is None:
            return 0
        remaining = (self.trial_end - datetime.utcnow()).days
        return max(remaining, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subscription_id": self.subscription_id,
            "tenant_id": self.tenant_id,
            "tier": self.tier.value,
            "status": self.status.value,
            "billing_cycle": self.billing_cycle.value,
            "monthly_price": self.monthly_price,
            "is_active": self.is_active,
            "is_trialing": self.is_trialing,
            "trial_days_remaining": self.trial_days_remaining,
            "tokens_used": self.tokens_used_this_period,
            "api_calls_used": self.api_calls_this_period,
            "current_period_start": self.current_period_start.isoformat() if self.current_period_start else None,
            "current_period_end": self.current_period_end.isoformat() if self.current_period_end else None,
            "cancel_at_period_end": self.cancel_at_period_end,
        }


@dataclass
class SubscriptionChange:
    """Records a subscription change event."""
    change_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subscription_id: str = ""
    tenant_id: str = ""
    change_type: ChangeType = ChangeType.UPGRADE
    from_tier: Optional[SubscriptionTier] = None
    to_tier: Optional[SubscriptionTier] = None
    from_status: Optional[SubscriptionStatus] = None
    to_status: Optional[SubscriptionStatus] = None
    reason: str = ""
    proration_amount: float = 0.0
    effective_at: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Default tier pricing
# --------------------------------------------------------------------------

DEFAULT_TIER_PRICING: Dict[SubscriptionTier, TierPricing] = {
    SubscriptionTier.FREE: TierPricing(
        tier=SubscriptionTier.FREE,
        monthly_price_usd=0.0,
        annual_price_usd=0.0,
        trial_days=0,
        limits=TierLimits(
            max_agents=1, max_tasks_per_day=20, max_workflows=3,
            max_plugins=2, max_api_calls_per_day=500, max_tokens_per_month=50_000,
            max_storage_mb=100, max_team_members=1, max_concurrent_executions=1,
        ),
        features=["Basic AI agent", "Community support", "Basic analytics"],
    ),
    SubscriptionTier.STARTER: TierPricing(
        tier=SubscriptionTier.STARTER,
        monthly_price_usd=29.0,
        annual_price_usd=290.0,
        overage_price_per_token=0.00002,
        overage_price_per_api_call=0.001,
        trial_days=14,
        limits=TierLimits(
            max_agents=3, max_tasks_per_day=200, max_workflows=20,
            max_plugins=10, max_api_calls_per_day=5_000, max_tokens_per_month=500_000,
            max_storage_mb=1_000, max_team_members=3, max_concurrent_executions=3,
            code_generation_enabled=True,
        ),
        features=["3 AI agents", "Code generation", "Email support", "Basic analytics", "Webhooks"],
    ),
    SubscriptionTier.PRO: TierPricing(
        tier=SubscriptionTier.PRO,
        monthly_price_usd=99.0,
        annual_price_usd=990.0,
        overage_price_per_token=0.000015,
        overage_price_per_api_call=0.0008,
        trial_days=14,
        limits=TierLimits(
            max_agents=10, max_tasks_per_day=1_000, max_workflows=100,
            max_plugins=50, max_api_calls_per_day=50_000, max_tokens_per_month=5_000_000,
            max_storage_mb=10_000, max_team_members=10, max_concurrent_executions=10,
            code_generation_enabled=True, advanced_analytics=True,
            audit_logs=True,
        ),
        features=["10 AI agents", "Advanced analytics", "Priority support",
                   "Audit logs", "Custom workflows", "API access", "Plugins marketplace"],
    ),
    SubscriptionTier.BUSINESS: TierPricing(
        tier=SubscriptionTier.BUSINESS,
        monthly_price_usd=299.0,
        annual_price_usd=2_990.0,
        overage_price_per_token=0.00001,
        overage_price_per_api_call=0.0005,
        trial_days=30,
        limits=TierLimits(
            max_agents=50, max_tasks_per_day=10_000, max_workflows=500,
            max_plugins=200, max_api_calls_per_day=500_000, max_tokens_per_month=50_000_000,
            max_storage_mb=100_000, max_team_members=50, max_concurrent_executions=50,
            code_generation_enabled=True, advanced_analytics=True,
            custom_models=True, priority_support=True, sso_enabled=True,
            audit_logs=True, custom_branding=True,
        ),
        features=["50 AI agents", "Custom AI models", "SSO", "Priority support",
                   "Custom branding", "Dedicated support", "SLA"],
    ),
    SubscriptionTier.ENTERPRISE: TierPricing(
        tier=SubscriptionTier.ENTERPRISE,
        monthly_price_usd=999.0,
        annual_price_usd=9_990.0,
        overage_price_per_token=0.000005,
        overage_price_per_api_call=0.0002,
        trial_days=30,
        setup_fee=5_000.0,
        limits=TierLimits(
            max_agents=999, max_tasks_per_day=999_999, max_workflows=9_999,
            max_plugins=9_999, max_api_calls_per_day=9_999_999,
            max_tokens_per_month=999_999_999,
            max_storage_mb=999_999, max_team_members=999,
            max_concurrent_executions=200,
            code_generation_enabled=True, advanced_analytics=True,
            custom_models=True, priority_support=True, sso_enabled=True,
            audit_logs=True, custom_branding=True,
            dedicated_resources=True, sla_guarantee=True,
        ),
        features=["Unlimited AI agents", "Dedicated infrastructure", "24/7 support",
                   "Custom SLA", "On-premise option", "Data residency", "HIPAA/SOC2"],
    ),
}


class SubscriptionEngine:
    """
    Production subscription management engine.

    Handles full lifecycle:
    - Trial creation and conversion
    - Upgrade / downgrade with proration
    - Usage tracking and limit enforcement
    - Cancellation and reactivation
    - Revenue analytics
    - Webhook notifications on changes
    """

    def __init__(
        self,
        tier_pricing: Optional[Dict[SubscriptionTier, TierPricing]] = None,
        on_change: Optional[Callable] = None,
    ):
        self._pricing = tier_pricing or DEFAULT_TIER_PRICING
        self._subscriptions: Dict[str, Subscription] = {}
        self._tenant_subscriptions: Dict[str, str] = {}  # tenant_id -> subscription_id
        self._changes: List[SubscriptionChange] = []
        self._on_change = on_change

        # Metrics
        self._total_revenue = 0.0
        self._mrr = 0.0
        self._active_count = 0
        self._trial_count = 0
        self._churn_count = 0

    # -- Subscription lifecycle ---------------------------------------------

    async def create_subscription(
        self,
        tenant_id: str,
        user_id: str,
        tier: SubscriptionTier = SubscriptionTier.FREE,
        billing_cycle: BillingCycle = BillingCycle.MONTHLY,
        coupon_code: Optional[str] = None,
        start_trial: bool = True,
    ) -> Subscription:
        pricing = self._pricing.get(tier)
        if not pricing:
            raise ValueError(f"Unknown tier: {tier}")

        now = datetime.utcnow()
        price = pricing.monthly_price_usd
        if billing_cycle == BillingCycle.ANNUAL:
            price = pricing.annual_price_usd / 12

        discount = 0.0
        if coupon_code:
            discount = self._apply_coupon(coupon_code)

        sub = Subscription(
            tenant_id=tenant_id,
            user_id=user_id,
            tier=tier,
            billing_cycle=billing_cycle,
            monthly_price=price * (1 - discount / 100),
            discount_percent=discount,
            coupon_code=coupon_code,
            current_period_start=now,
            current_period_end=now + timedelta(days=30),
        )

        if start_trial and pricing.trial_days > 0 and tier != SubscriptionTier.FREE:
            sub.status = SubscriptionStatus.TRIALING
            sub.trial_start = now
            sub.trial_end = now + timedelta(days=pricing.trial_days)
            self._trial_count += 1
        else:
            sub.status = SubscriptionStatus.ACTIVE
            self._active_count += 1

        self._subscriptions[sub.subscription_id] = sub
        self._tenant_subscriptions[tenant_id] = sub.subscription_id
        self._recalculate_mrr()

        await self._record_change(SubscriptionChange(
            subscription_id=sub.subscription_id,
            tenant_id=tenant_id,
            change_type=ChangeType.UPGRADE,
            to_tier=tier,
            to_status=sub.status,
            reason="initial_subscription",
        ))

        logger.info("Created subscription %s for tenant %s (tier=%s)",
                     sub.subscription_id, tenant_id, tier.value)
        return sub

    async def upgrade(
        self,
        tenant_id: str,
        new_tier: SubscriptionTier,
        prorate: bool = True,
    ) -> Subscription:
        sub = self._get_tenant_subscription(tenant_id)
        if not sub:
            raise ValueError(f"No subscription found for tenant {tenant_id}")

        old_tier = sub.tier
        pricing = self._pricing.get(new_tier)
        if not pricing:
            raise ValueError(f"Unknown tier: {new_tier}")

        tier_order: List[SubscriptionTier] = [t for t in SubscriptionTier]
        if tier_order.index(new_tier) <= tier_order.index(old_tier):
            raise ValueError("New tier must be higher than current tier for upgrade")

        proration_amount = 0.0
        if prorate and sub.current_period_end:
            days_remaining = (sub.current_period_end - datetime.utcnow()).days
            daily_old = sub.monthly_price / 30
            daily_new = pricing.monthly_price_usd / 30
            proration_amount = (daily_new - daily_old) * days_remaining

        sub.tier = new_tier
        sub.monthly_price = pricing.monthly_price_usd
        if sub.status == SubscriptionStatus.TRIALING:
            sub.status = SubscriptionStatus.ACTIVE

        await self._record_change(SubscriptionChange(
            subscription_id=sub.subscription_id,
            tenant_id=tenant_id,
            change_type=ChangeType.UPGRADE,
            from_tier=old_tier,
            to_tier=new_tier,
            proration_amount=proration_amount,
            reason="user_upgrade",
        ))

        self._recalculate_mrr()
        logger.info("Upgraded tenant %s from %s to %s", tenant_id, old_tier.value, new_tier.value)
        return sub

    async def downgrade(
        self,
        tenant_id: str,
        new_tier: SubscriptionTier,
        effective_at_period_end: bool = True,
    ) -> Subscription:
        sub = self._get_tenant_subscription(tenant_id)
        if not sub:
            raise ValueError(f"No subscription found for tenant {tenant_id}")

        old_tier = sub.tier
        pricing = self._pricing.get(new_tier)
        if not pricing:
            raise ValueError(f"Unknown tier: {new_tier}")

        if effective_at_period_end:
            sub.metadata["pending_downgrade"] = new_tier.value
        else:
            sub.tier = new_tier
            sub.monthly_price = pricing.monthly_price_usd

        await self._record_change(SubscriptionChange(
            subscription_id=sub.subscription_id,
            tenant_id=tenant_id,
            change_type=ChangeType.DOWNGRADE,
            from_tier=old_tier,
            to_tier=new_tier,
            reason="user_downgrade",
        ))

        self._recalculate_mrr()
        return sub

    async def cancel(
        self,
        tenant_id: str,
        reason: str = "",
        at_period_end: bool = True,
    ) -> Subscription:
        sub = self._get_tenant_subscription(tenant_id)
        if not sub:
            raise ValueError(f"No subscription found for tenant {tenant_id}")

        if at_period_end:
            sub.cancel_at_period_end = True
            sub.canceled_at = datetime.utcnow()
        else:
            sub.status = SubscriptionStatus.CANCELED
            sub.canceled_at = datetime.utcnow()
            self._active_count -= 1
            self._churn_count += 1

        await self._record_change(SubscriptionChange(
            subscription_id=sub.subscription_id,
            tenant_id=tenant_id,
            change_type=ChangeType.CANCEL,
            from_tier=sub.tier,
            from_status=SubscriptionStatus.ACTIVE,
            to_status=SubscriptionStatus.CANCELED,
            reason=reason or "user_requested",
        ))

        self._recalculate_mrr()
        return sub

    async def reactivate(self, tenant_id: str) -> Subscription:
        sub = self._get_tenant_subscription(tenant_id)
        if not sub:
            raise ValueError(f"No subscription found for tenant {tenant_id}")

        sub.status = SubscriptionStatus.ACTIVE
        sub.cancel_at_period_end = False
        sub.canceled_at = None
        self._active_count += 1

        await self._record_change(SubscriptionChange(
            subscription_id=sub.subscription_id,
            tenant_id=tenant_id,
            change_type=ChangeType.REACTIVATE,
            to_status=SubscriptionStatus.ACTIVE,
            reason="user_reactivated",
        ))

        self._recalculate_mrr()
        return sub

    async def pause(self, tenant_id: str, resume_date: Optional[datetime] = None) -> Subscription:
        sub = self._get_tenant_subscription(tenant_id)
        if not sub:
            raise ValueError(f"No subscription found for tenant {tenant_id}")

        sub.status = SubscriptionStatus.PAUSED
        if resume_date:
            sub.metadata["resume_date"] = resume_date.isoformat()

        await self._record_change(SubscriptionChange(
            subscription_id=sub.subscription_id,
            tenant_id=tenant_id,
            change_type=ChangeType.PAUSE,
            to_status=SubscriptionStatus.PAUSED,
        ))
        return sub

    # -- Usage tracking -----------------------------------------------------

    async def record_usage(
        self,
        tenant_id: str,
        tokens: int = 0,
        api_calls: int = 0,
        storage_mb: float = 0.0,
    ) -> Dict[str, Any]:
        sub = self._get_tenant_subscription(tenant_id)
        if not sub:
            return {"error": "no_subscription"}

        sub.tokens_used_this_period += tokens
        sub.api_calls_this_period += api_calls
        sub.storage_used_mb += storage_mb

        limits = sub.custom_limits or self._get_tier_limits(sub.tier)
        overage: Dict[str, Any] = {}

        if sub.tokens_used_this_period > limits.max_tokens_per_month:
            overage["tokens"] = sub.tokens_used_this_period - limits.max_tokens_per_month
        if sub.api_calls_this_period > limits.max_api_calls_per_day * 30:
            overage["api_calls"] = sub.api_calls_this_period - (limits.max_api_calls_per_day * 30)
        if sub.storage_used_mb > limits.max_storage_mb:
            overage["storage_mb"] = sub.storage_used_mb - limits.max_storage_mb

        return {
            "tokens_used": sub.tokens_used_this_period,
            "tokens_limit": limits.max_tokens_per_month,
            "tokens_percent": min(sub.tokens_used_this_period / max(limits.max_tokens_per_month, 1) * 100, 100),
            "api_calls_used": sub.api_calls_this_period,
            "storage_used_mb": sub.storage_used_mb,
            "overage": overage,
            "has_overage": len(overage) > 0,
        }

    def check_limit(self, tenant_id: str, resource: str, amount: int = 1) -> Dict[str, Any]:
        sub = self._get_tenant_subscription(tenant_id)
        if not sub or not sub.is_active:
            return {"allowed": False, "reason": "inactive_subscription"}

        limits = sub.custom_limits or self._get_tier_limits(sub.tier)

        limit_map = {
            "agents": limits.max_agents,
            "tasks_per_day": limits.max_tasks_per_day,
            "workflows": limits.max_workflows,
            "plugins": limits.max_plugins,
            "api_calls": limits.max_api_calls_per_day,
            "tokens": limits.max_tokens_per_month,
            "team_members": limits.max_team_members,
            "concurrent": limits.max_concurrent_executions,
            "storage_mb": limits.max_storage_mb,
        }

        max_val = limit_map.get(resource)
        if max_val is None:
            return {"allowed": True, "reason": "unknown_resource"}

        feature_checks = {
            "code_generation": limits.code_generation_enabled,
            "advanced_analytics": limits.advanced_analytics,
            "custom_models": limits.custom_models,
            "sso": limits.sso_enabled,
            "audit_logs": limits.audit_logs,
        }

        if resource in feature_checks:
            return {"allowed": feature_checks[resource], "reason": f"feature_{resource}"}

        return {
            "allowed": amount <= max_val,
            "limit": max_val,
            "reason": "within_limit" if amount <= max_val else "limit_exceeded",
            "upgrade_tier": self._suggest_upgrade_tier(sub.tier, resource),
        }

    # -- Queries ------------------------------------------------------------

    def get_subscription(self, tenant_id: str) -> Optional[Subscription]:
        return self._get_tenant_subscription(tenant_id)

    def get_tier_pricing(self, tier: SubscriptionTier) -> Optional[TierPricing]:
        return self._pricing.get(tier)

    def get_all_pricing(self) -> List[Dict[str, Any]]:
        result = []
        for tier, pricing in self._pricing.items():
            result.append({
                "tier": tier.value,
                "monthly_price": pricing.monthly_price_usd,
                "annual_price": pricing.annual_price_usd,
                "trial_days": pricing.trial_days,
                "features": pricing.features,
                "limits": {
                    "agents": pricing.limits.max_agents,
                    "tasks_per_day": pricing.limits.max_tasks_per_day,
                    "api_calls_per_day": pricing.limits.max_api_calls_per_day,
                    "tokens_per_month": pricing.limits.max_tokens_per_month,
                    "storage_mb": pricing.limits.max_storage_mb,
                    "team_members": pricing.limits.max_team_members,
                },
            })
        return result

    def get_revenue_stats(self) -> Dict[str, Any]:
        tier_counts: Dict[str, int] = {}
        for sub in self._subscriptions.values():
            tier_counts[sub.tier.value] = tier_counts.get(sub.tier.value, 0) + 1

        churn_rate = float(self._churn_count) / max(self._active_count + self._churn_count, 1) * 100
        arpu = float(self._mrr) / max(self._active_count, 1)

        return {
            "mrr": round(float(self._mrr), 2),
            "arr": round(float(self._mrr) * 12, 2),
            "active_subscriptions": self._active_count,
            "trialing": self._trial_count,
            "churned": self._churn_count,
            "churn_rate": round(churn_rate, 2),
            "tier_distribution": tier_counts,
            "total_changes": len(self._changes),
            "arpu": round(arpu, 2),
        }

    def get_change_history(
        self, tenant_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        changes = self._changes
        if tenant_id:
            changes = [c for c in changes if c.tenant_id == tenant_id]
        sliced = changes[-limit:] if limit < len(changes) else changes
        return [
            {
                "change_id": c.change_id,
                "tenant_id": c.tenant_id,
                "change_type": c.change_type.value,
                "from_tier": c.from_tier.value if c.from_tier else None,
                "to_tier": c.to_tier.value if c.to_tier else None,
                "reason": c.reason,
                "proration_amount": c.proration_amount,
                "effective_at": c.effective_at.isoformat(),
            }
            for c in sliced
        ]

    # -- Internal -----------------------------------------------------------

    def _get_tenant_subscription(self, tenant_id: str) -> Optional[Subscription]:
        sub_id = self._tenant_subscriptions.get(tenant_id)
        return self._subscriptions.get(sub_id) if sub_id else None

    def _get_tier_limits(self, tier: SubscriptionTier) -> TierLimits:
        pricing = self._pricing.get(tier)
        return pricing.limits if pricing else TierLimits()

    def _recalculate_mrr(self) -> None:
        self._mrr = sum(
            sub.monthly_price
            for sub in self._subscriptions.values()
            if sub.is_active and not sub.is_trialing
        )

    def _apply_coupon(self, coupon_code: str) -> float:
        coupons = {
            "LAUNCH50": 50.0,
            "STARTUP25": 25.0,
            "ANNUAL20": 20.0,
            "FRIEND10": 10.0,
        }
        return coupons.get(coupon_code.upper(), 0.0)

    def _suggest_upgrade_tier(self, current_tier: SubscriptionTier, resource: str) -> Optional[str]:
        tier_order: List[SubscriptionTier] = [
            SubscriptionTier.FREE, SubscriptionTier.STARTER, SubscriptionTier.PRO,
            SubscriptionTier.BUSINESS, SubscriptionTier.ENTERPRISE,
        ]
        idx = tier_order.index(current_tier)
        if idx < len(tier_order) - 1:
            next_tier = tier_order[idx + 1]
            return next_tier.value
        return None

    async def _record_change(self, change: SubscriptionChange) -> None:
        self._changes.append(change)
        if self._on_change is not None:
            try:
                if asyncio.iscoroutinefunction(self._on_change):
                    await self._on_change(change)
                else:
                    self._on_change(change)
            except Exception:
                logger.exception("Change callback error")
