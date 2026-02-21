"""
Advanced Subscription Management and Monetization Layer
Enterprise-grade subscription system with intelligent pricing, usage tracking, and revenue optimization
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import json


class SubscriptionTier(Enum):
    """Subscription tiers"""
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class BillingCycle(Enum):
    """Billing cycle periods"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    CUSTOM = "custom"


class SubscriptionStatus(Enum):
    """Subscription status"""
    ACTIVE = "active"
    TRIALING = "trialing"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    PENDING_PAYMENT = "pending_payment"


class PaymentMethod(Enum):
    """Payment methods"""
    CREDIT_CARD = "credit_card"
    BANK_TRANSFER = "bank_transfer"
    PAYPAL = "paypal"
    CRYPTO = "crypto"
    INVOICE = "invoice"


@dataclass
class PricingModel:
    """Pricing model configuration"""
    tier: SubscriptionTier
    base_price: Decimal
    billing_cycle: BillingCycle
    included_compute_hours: int
    included_api_calls: int
    included_storage_gb: int
    included_users: int
    overage_compute_price: Decimal  # Per hour
    overage_api_price: Decimal  # Per 1000 calls
    overage_storage_price: Decimal  # Per GB
    overage_user_price: Decimal  # Per user
    features: List[str] = field(default_factory=list)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    support_level: str = "email"
    sla_uptime: float = 99.0

    def calculate_cost(
        self,
        compute_hours: int,
        api_calls: int,
        storage_gb: int,
        users: int
    ) -> Decimal:
        """Calculate total cost based on usage"""
        cost = self.base_price

        # Calculate overages
        if compute_hours > self.included_compute_hours:
            overage = compute_hours - self.included_compute_hours
            cost += Decimal(overage) * self.overage_compute_price

        if api_calls > self.included_api_calls:
            overage = (api_calls - self.included_api_calls) / 1000
            cost += Decimal(overage) * self.overage_api_price

        if storage_gb > self.included_storage_gb:
            overage = storage_gb - self.included_storage_gb
            cost += Decimal(overage) * self.overage_storage_price

        if users > self.included_users:
            overage = users - self.included_users
            cost += Decimal(overage) * self.overage_user_price

        return cost


@dataclass
class Subscription:
    """Customer subscription"""
    subscription_id: str
    tenant_id: str
    tier: SubscriptionTier
    status: SubscriptionStatus
    pricing_model: PricingModel
    billing_cycle: BillingCycle
    current_period_start: datetime
    current_period_end: datetime
    trial_end: Optional[datetime] = None
    auto_renew: bool = True
    payment_method: Optional[PaymentMethod] = None
    discount_percent: Decimal = Decimal("0")
    custom_pricing: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    cancelled_at: Optional[datetime] = None
    cancellation_reason: Optional[str] = None

    def is_active(self) -> bool:
        """Check if subscription is currently active"""
        return self.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]

    def days_remaining(self) -> int:
        """Days remaining in current period"""
        delta = self.current_period_end - datetime.utcnow()
        return max(delta.days, 0)

    def is_trial(self) -> bool:
        """Check if in trial period"""
        return self.status == SubscriptionStatus.TRIALING

    def can_upgrade(self, new_tier: SubscriptionTier) -> bool:
        """Check if can upgrade to new tier"""
        tier_order = [
            SubscriptionTier.FREE,
            SubscriptionTier.STARTER,
            SubscriptionTier.PRO,
            SubscriptionTier.BUSINESS,
            SubscriptionTier.ENTERPRISE
        ]

        if self.tier not in tier_order or new_tier not in tier_order:
            return True  # Custom tiers always allowed

        return tier_order.index(new_tier) > tier_order.index(self.tier)


@dataclass
class UsageRecord:
    """Usage tracking record"""
    record_id: str
    tenant_id: str
    subscription_id: str
    timestamp: datetime
    compute_hours: Decimal
    api_calls: int
    storage_gb: Decimal
    active_users: int
    feature_usage: Dict[str, int] = field(default_factory=dict)
    cost: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Invoice:
    """Billing invoice"""
    invoice_id: str
    tenant_id: str
    subscription_id: str
    billing_period_start: datetime
    billing_period_end: datetime
    subtotal: Decimal
    discount: Decimal
    tax: Decimal
    total: Decimal
    status: str  # pending, paid, failed, refunded
    due_date: datetime
    paid_at: Optional[datetime] = None
    payment_method: Optional[PaymentMethod] = None
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


class PricingStrategy:
    """Intelligent pricing strategy engine"""

    def __init__(self):
        self.tiers = self._initialize_tiers()

    def _initialize_tiers(self) -> Dict[SubscriptionTier, PricingModel]:
        """Initialize default pricing tiers"""
        return {
            SubscriptionTier.FREE: PricingModel(
                tier=SubscriptionTier.FREE,
                base_price=Decimal("0"),
                billing_cycle=BillingCycle.MONTHLY,
                included_compute_hours=10,
                included_api_calls=1000,
                included_storage_gb=1,
                included_users=1,
                overage_compute_price=Decimal("0"),  # No overages on free
                overage_api_price=Decimal("0"),
                overage_storage_price=Decimal("0"),
                overage_user_price=Decimal("0"),
                features=["basic_agent", "email_support"],
                rate_limits={"api_calls_per_minute": 10, "concurrent_agents": 1},
                support_level="email",
                sla_uptime=95.0
            ),
            SubscriptionTier.STARTER: PricingModel(
                tier=SubscriptionTier.STARTER,
                base_price=Decimal("29.99"),
                billing_cycle=BillingCycle.MONTHLY,
                included_compute_hours=100,
                included_api_calls=10000,
                included_storage_gb=10,
                included_users=3,
                overage_compute_price=Decimal("0.50"),
                overage_api_price=Decimal("0.10"),
                overage_storage_price=Decimal("0.20"),
                overage_user_price=Decimal("10.00"),
                features=[
                    "basic_agent",
                    "advanced_agent",
                    "workflow_builder",
                    "email_support"
                ],
                rate_limits={"api_calls_per_minute": 100, "concurrent_agents": 3},
                support_level="email",
                sla_uptime=99.0
            ),
            SubscriptionTier.PRO: PricingModel(
                tier=SubscriptionTier.PRO,
                base_price=Decimal("99.99"),
                billing_cycle=BillingCycle.MONTHLY,
                included_compute_hours=500,
                included_api_calls=100000,
                included_storage_gb=100,
                included_users=10,
                overage_compute_price=Decimal("0.40"),
                overage_api_price=Decimal("0.08"),
                overage_storage_price=Decimal("0.15"),
                overage_user_price=Decimal("8.00"),
                features=[
                    "all_agents",
                    "workflow_builder",
                    "plugin_marketplace",
                    "advanced_analytics",
                    "priority_support",
                    "api_access"
                ],
                rate_limits={"api_calls_per_minute": 1000, "concurrent_agents": 10},
                support_level="priority",
                sla_uptime=99.5
            ),
            SubscriptionTier.BUSINESS: PricingModel(
                tier=SubscriptionTier.BUSINESS,
                base_price=Decimal("299.99"),
                billing_cycle=BillingCycle.MONTHLY,
                included_compute_hours=2000,
                included_api_calls=1000000,
                included_storage_gb=500,
                included_users=50,
                overage_compute_price=Decimal("0.30"),
                overage_api_price=Decimal("0.05"),
                overage_storage_price=Decimal("0.10"),
                overage_user_price=Decimal("5.00"),
                features=[
                    "all_agents",
                    "workflow_builder",
                    "plugin_marketplace",
                    "advanced_analytics",
                    "custom_models",
                    "sso",
                    "audit_logs",
                    "dedicated_support",
                    "api_access"
                ],
                rate_limits={"api_calls_per_minute": 5000, "concurrent_agents": 50},
                support_level="dedicated",
                sla_uptime=99.9
            ),
            SubscriptionTier.ENTERPRISE: PricingModel(
                tier=SubscriptionTier.ENTERPRISE,
                base_price=Decimal("999.99"),
                billing_cycle=BillingCycle.MONTHLY,
                included_compute_hours=10000,
                included_api_calls=10000000,
                included_storage_gb=2000,
                included_users=999,
                overage_compute_price=Decimal("0.20"),
                overage_api_price=Decimal("0.03"),
                overage_storage_price=Decimal("0.05"),
                overage_user_price=Decimal("3.00"),
                features=[
                    "everything",
                    "custom_deployment",
                    "white_label",
                    "custom_sla",
                    "dedicated_account_manager",
                    "24/7_phone_support",
                    "custom_integrations"
                ],
                rate_limits={"api_calls_per_minute": 99999, "concurrent_agents": 999},
                support_level="enterprise",
                sla_uptime=99.99
            )
        }

    def get_pricing_model(self, tier: SubscriptionTier) -> PricingModel:
        """Get pricing model for tier"""
        return self.tiers[tier]

    def calculate_upgrade_cost(
        self,
        current_sub: Subscription,
        new_tier: SubscriptionTier,
        prorate: bool = True
    ) -> Decimal:
        """Calculate cost to upgrade subscription"""
        new_pricing = self.tiers[new_tier]

        if not prorate:
            return new_pricing.base_price

        # Calculate prorated amount
        days_remaining = current_sub.days_remaining()
        total_days = (current_sub.current_period_end - current_sub.current_period_start).days
        proration_factor = Decimal(days_remaining) / Decimal(total_days)

        current_value = current_sub.pricing_model.base_price * proration_factor
        new_value = new_pricing.base_price * proration_factor

        return max(new_value - current_value, Decimal("0"))

    def calculate_discount(
        self,
        base_price: Decimal,
        discount_type: str,
        discount_value: Decimal
    ) -> Decimal:
        """Calculate discount amount"""
        if discount_type == "percent":
            return base_price * (discount_value / Decimal("100"))
        elif discount_type == "fixed":
            return discount_value
        return Decimal("0")


class SubscriptionManager:
    """Advanced subscription management system"""

    def __init__(self):
        self.pricing_strategy = PricingStrategy()
        self.subscriptions: Dict[str, Subscription] = {}
        self.usage_records: Dict[str, List[UsageRecord]] = {}
        self.invoices: Dict[str, List[Invoice]] = {}

    async def create_subscription(
        self,
        tenant_id: str,
        tier: SubscriptionTier,
        billing_cycle: BillingCycle,
        payment_method: Optional[PaymentMethod] = None,
        trial_days: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Subscription:
        """Create new subscription"""
        import uuid

        subscription_id = str(uuid.uuid4())
        pricing_model = self.pricing_strategy.get_pricing_model(tier)

        now = datetime.utcnow()
        period_start = now

        # Calculate period end based on billing cycle
        if billing_cycle == BillingCycle.MONTHLY:
            period_end = now + timedelta(days=30)
        elif billing_cycle == BillingCycle.QUARTERLY:
            period_end = now + timedelta(days=90)
        elif billing_cycle == BillingCycle.ANNUAL:
            period_end = now + timedelta(days=365)
        else:
            period_end = now + timedelta(days=30)

        # Handle trial
        trial_end = None
        status = SubscriptionStatus.ACTIVE

        if trial_days > 0:
            trial_end = now + timedelta(days=trial_days)
            status = SubscriptionStatus.TRIALING

        subscription = Subscription(
            subscription_id=subscription_id,
            tenant_id=tenant_id,
            tier=tier,
            status=status,
            pricing_model=pricing_model,
            billing_cycle=billing_cycle,
            current_period_start=period_start,
            current_period_end=period_end,
            trial_end=trial_end,
            payment_method=payment_method,
            metadata=metadata or {}
        )

        self.subscriptions[subscription_id] = subscription
        return subscription

    async def upgrade_subscription(
        self,
        subscription_id: str,
        new_tier: SubscriptionTier,
        immediate: bool = True
    ) -> Dict[str, Any]:
        """Upgrade subscription to new tier"""
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            raise ValueError("Subscription not found")

        if not subscription.can_upgrade(new_tier):
            raise ValueError("Cannot downgrade to lower tier")

        # Calculate upgrade cost
        upgrade_cost = self.pricing_strategy.calculate_upgrade_cost(
            subscription,
            new_tier,
            prorate=True
        )

        if immediate:
            # Apply upgrade immediately
            old_tier = subscription.tier
            subscription.tier = new_tier
            subscription.pricing_model = self.pricing_strategy.get_pricing_model(new_tier)
            subscription.updated_at = datetime.utcnow()

            # Create upgrade invoice
            invoice = await self._create_upgrade_invoice(
                subscription,
                old_tier,
                new_tier,
                upgrade_cost
            )

            return {
                "success": True,
                "subscription": subscription,
                "upgrade_cost": upgrade_cost,
                "invoice": invoice
            }
        else:
            # Schedule upgrade for next billing period
            subscription.metadata["scheduled_upgrade"] = {
                "new_tier": new_tier.value,
                "scheduled_for": subscription.current_period_end.isoformat()
            }
            subscription.updated_at = datetime.utcnow()

            return {
                "success": True,
                "subscription": subscription,
                "upgrade_cost": Decimal("0"),
                "scheduled_for": subscription.current_period_end
            }

    async def cancel_subscription(
        self,
        subscription_id: str,
        immediate: bool = False,
        reason: Optional[str] = None
    ) -> Subscription:
        """Cancel subscription"""
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            raise ValueError("Subscription not found")

        now = datetime.utcnow()
        subscription.cancelled_at = now
        subscription.cancellation_reason = reason

        if immediate:
            subscription.status = SubscriptionStatus.CANCELLED
            subscription.current_period_end = now
        else:
            # Cancel at end of billing period
            subscription.auto_renew = False
            subscription.metadata["cancel_at_period_end"] = True

        subscription.updated_at = now

        return subscription

    async def record_usage(
        self,
        tenant_id: str,
        subscription_id: str,
        compute_hours: Decimal,
        api_calls: int,
        storage_gb: Decimal,
        active_users: int,
        feature_usage: Optional[Dict[str, int]] = None
    ) -> UsageRecord:
        """Record usage for billing"""
        import uuid

        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            raise ValueError("Subscription not found")

        # Calculate cost
        cost = subscription.pricing_model.calculate_cost(
            int(compute_hours),
            api_calls,
            int(storage_gb),
            active_users
        )

        record = UsageRecord(
            record_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            subscription_id=subscription_id,
            timestamp=datetime.utcnow(),
            compute_hours=compute_hours,
            api_calls=api_calls,
            storage_gb=storage_gb,
            active_users=active_users,
            feature_usage=feature_usage or {},
            cost=cost
        )

        if tenant_id not in self.usage_records:
            self.usage_records[tenant_id] = []

        self.usage_records[tenant_id].append(record)

        return record

    async def generate_invoice(
        self,
        subscription_id: str
    ) -> Invoice:
        """Generate invoice for billing period"""
        import uuid

        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            raise ValueError("Subscription not found")

        # Get usage for period
        usage_records = [
            r for r in self.usage_records.get(subscription.tenant_id, [])
            if r.subscription_id == subscription_id
            and r.timestamp >= subscription.current_period_start
            and r.timestamp <= subscription.current_period_end
        ]

        # Calculate totals
        subtotal = subscription.pricing_model.base_price

        line_items = [{
            "description": f"{subscription.tier.value.title()} Subscription",
            "amount": float(subscription.pricing_model.base_price)
        }]

        # Add usage charges
        if usage_records:
            usage_cost = sum(r.cost for r in usage_records if r.cost)
            subtotal += usage_cost
            line_items.append({
                "description": "Usage Charges",
                "amount": float(usage_cost)
            })

        # Calculate discount
        discount = Decimal("0")
        if subscription.discount_percent > 0:
            discount = subtotal * (subscription.discount_percent / Decimal("100"))

        # Calculate tax (simplified - would use tax service)
        tax = (subtotal - discount) * Decimal("0.08")  # 8% tax

        total = subtotal - discount + tax

        invoice = Invoice(
            invoice_id=str(uuid.uuid4()),
            tenant_id=subscription.tenant_id,
            subscription_id=subscription_id,
            billing_period_start=subscription.current_period_start,
            billing_period_end=subscription.current_period_end,
            subtotal=subtotal,
            discount=discount,
            tax=tax,
            total=total,
            status="pending",
            due_date=subscription.current_period_end + timedelta(days=7),
            payment_method=subscription.payment_method,
            line_items=line_items
        )

        if subscription.tenant_id not in self.invoices:
            self.invoices[subscription.tenant_id] = []

        self.invoices[subscription.tenant_id].append(invoice)

        return invoice

    async def _create_upgrade_invoice(
        self,
        subscription: Subscription,
        old_tier: SubscriptionTier,
        new_tier: SubscriptionTier,
        amount: Decimal
    ) -> Invoice:
        """Create invoice for upgrade"""
        import uuid

        invoice = Invoice(
            invoice_id=str(uuid.uuid4()),
            tenant_id=subscription.tenant_id,
            subscription_id=subscription.subscription_id,
            billing_period_start=datetime.utcnow(),
            billing_period_end=subscription.current_period_end,
            subtotal=amount,
            discount=Decimal("0"),
            tax=amount * Decimal("0.08"),
            total=amount * Decimal("1.08"),
            status="pending",
            due_date=datetime.utcnow() + timedelta(days=7),
            payment_method=subscription.payment_method,
            line_items=[{
                "description": f"Upgrade: {old_tier.value} → {new_tier.value}",
                "amount": float(amount)
            }]
        )

        if subscription.tenant_id not in self.invoices:
            self.invoices[subscription.tenant_id] = []

        self.invoices[subscription.tenant_id].append(invoice)

        return invoice

    async def process_payment(
        self,
        invoice_id: str,
        payment_method: PaymentMethod
    ) -> Dict[str, Any]:
        """Process payment for invoice"""
        # Find invoice
        invoice = None
        for invoices in self.invoices.values():
            for inv in invoices:
                if inv.invoice_id == invoice_id:
                    invoice = inv
                    break

        if not invoice:
            raise ValueError("Invoice not found")

        # Simulate payment processing
        await asyncio.sleep(0.1)

        invoice.status = "paid"
        invoice.paid_at = datetime.utcnow()
        invoice.payment_method = payment_method

        return {
            "success": True,
            "invoice_id": invoice_id,
            "amount_paid": invoice.total,
            "paid_at": invoice.paid_at
        }

    async def get_subscription_metrics(
        self,
        tenant_id: str
    ) -> Dict[str, Any]:
        """Get subscription metrics"""
        subscriptions = [
            s for s in self.subscriptions.values()
            if s.tenant_id == tenant_id
        ]

        if not subscriptions:
            return {}

        current_sub = subscriptions[-1]  # Most recent

        usage_records = self.usage_records.get(tenant_id, [])
        current_period_usage = [
            r for r in usage_records
            if r.timestamp >= current_sub.current_period_start
            and r.timestamp <= current_sub.current_period_end
        ]

        total_compute = sum(r.compute_hours for r in current_period_usage)
        total_api_calls = sum(r.api_calls for r in current_period_usage)
        total_storage = max((r.storage_gb for r in current_period_usage), default=Decimal("0"))

        return {
            "subscription_id": current_sub.subscription_id,
            "tier": current_sub.tier.value,
            "status": current_sub.status.value,
            "days_remaining": current_sub.days_remaining(),
            "usage": {
                "compute_hours": float(total_compute),
                "compute_limit": current_sub.pricing_model.included_compute_hours,
                "api_calls": total_api_calls,
                "api_limit": current_sub.pricing_model.included_api_calls,
                "storage_gb": float(total_storage),
                "storage_limit": current_sub.pricing_model.included_storage_gb
            },
            "features": current_sub.pricing_model.features,
            "rate_limits": current_sub.pricing_model.rate_limits
        }
