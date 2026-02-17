"""
Billing Domain - Domain Services

Domain services contain business logic that doesn't naturally fit in entities.
These are stateless and operate on domain entities using repository interfaces.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Optional
from uuid import UUID

from .entities import (
    Subscription,
    SubscriptionStatus,
    SubscriptionTier,
    Invoice,
    InvoiceStatus,
    UsageRecord,
    EntitlementCheck,
)
from .repositories import (
    SubscriptionRepository,
    InvoiceRepository,
    UsageRecordRepository,
)


class EntitlementService:
    """
    Domain service for checking tenant entitlements.
    
    Validates whether a tenant can use a resource based on their
    subscription tier and current usage.
    """
    
    TIER_LIMITS = {
        SubscriptionTier.FREE: {
            "executions": Decimal("100"),
            "tokens": Decimal("100000"),
            "storage_mb": Decimal("1000"),
            "api_calls": Decimal("1000"),
            "agents": Decimal("2"),
            "workflows": Decimal("5"),
        },
        SubscriptionTier.PRO: {
            "executions": Decimal("10000"),
            "tokens": Decimal("10000000"),
            "storage_mb": Decimal("10000"),
            "api_calls": Decimal("100000"),
            "agents": Decimal("10"),
            "workflows": Decimal("100"),
        },
        SubscriptionTier.TEAM: {
            "executions": Decimal("100000"),
            "tokens": Decimal("100000000"),
            "storage_mb": Decimal("100000"),
            "api_calls": Decimal("1000000"),
            "agents": Decimal("50"),
            "workflows": Decimal("500"),
        },
        SubscriptionTier.ENTERPRISE: {
            "executions": None,  # Unlimited
            "tokens": None,  # Unlimited
            "storage_mb": None,  # Unlimited
            "api_calls": None,  # Unlimited
            "agents": None,  # Unlimited
            "workflows": None,  # Unlimited
        },
    }
    
    def __init__(
        self,
        subscription_repository: SubscriptionRepository,
        usage_repository: UsageRecordRepository,
    ):
        """
        Initialize entitlement service.
        
        Args:
            subscription_repository: Repository for subscription data
            usage_repository: Repository for usage data
        """
        self.subscription_repository = subscription_repository
        self.usage_repository = usage_repository
    
    async def check_entitlement(
        self,
        tenant_id: UUID,
        resource_type: str,
        quantity: Decimal = Decimal("1"),
    ) -> EntitlementCheck:
        """
        Check if tenant is entitled to use a resource.
        
        Args:
            tenant_id: Tenant identifier
            resource_type: Type of resource to check
            quantity: Quantity of resource requested
            
        Returns:
            EntitlementCheck with result and details
        """
        subscription = await self.subscription_repository.get_by_tenant(tenant_id)
        
        if not subscription:
            return EntitlementCheck(
                allowed=False,
                reason="No active subscription found",
            )
        
        if not subscription.is_active():
            return EntitlementCheck(
                allowed=False,
                reason=f"Subscription is not active (status: {subscription.status.value})",
            )
        
        limit = self.TIER_LIMITS.get(subscription.tier, {}).get(resource_type)
        
        if limit is None:
            return EntitlementCheck(
                allowed=True,
                reason="Unlimited usage for tier",
                limit=None,
            )
        
        now = datetime.utcnow()
        period_start = subscription.current_period_start
        period_end = subscription.current_period_end
        
        current_usage = await self.usage_repository.aggregate_usage(
            tenant_id=tenant_id,
            resource_type=resource_type,
            start_time=period_start,
            end_time=now,
        )
        
        remaining = limit - current_usage
        
        if current_usage + quantity > limit:
            return EntitlementCheck(
                allowed=False,
                reason=f"Usage limit exceeded for {resource_type}",
                current_usage=current_usage,
                limit=limit,
                remaining=remaining,
            )
        
        return EntitlementCheck(
            allowed=True,
            current_usage=current_usage,
            limit=limit,
            remaining=remaining,
        )
    
    async def get_tier_limits(self, tier: SubscriptionTier) -> Dict[str, Optional[Decimal]]:
        """
        Get resource limits for a subscription tier.
        
        Args:
            tier: Subscription tier
            
        Returns:
            Dictionary of resource type to limit
        """
        return self.TIER_LIMITS.get(tier, {})


class UsageMeteringService:
    """
    Domain service for metering and recording resource usage.
    
    Tracks tenant resource consumption for billing purposes.
    """
    
    def __init__(self, usage_repository: UsageRecordRepository):
        """
        Initialize usage metering service.
        
        Args:
            usage_repository: Repository for usage records
        """
        self.usage_repository = usage_repository
    
    async def record_usage(
        self,
        tenant_id: UUID,
        resource_type: str,
        quantity: Decimal,
        unit: str = "count",
        metadata: Optional[Dict] = None,
    ) -> UsageRecord:
        """
        Record resource usage for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            resource_type: Type of resource consumed
            quantity: Quantity consumed
            unit: Unit of measurement
            metadata: Optional metadata about the usage
            
        Returns:
            Created usage record
        """
        usage_record = UsageRecord.create(
            tenant_id=tenant_id,
            resource_type=resource_type,
            quantity=quantity,
            unit=unit,
            metadata=metadata,
        )
        
        return await self.usage_repository.create(usage_record)
    
    async def get_monthly_usage(
        self,
        tenant_id: UUID,
        resource_type: Optional[str] = None,
    ) -> Dict[str, Decimal]:
        """
        Get usage for the current month.
        
        Args:
            tenant_id: Tenant identifier
            resource_type: Optional resource type filter
            
        Returns:
            Dictionary of resource type to usage quantity
        """
        now = datetime.utcnow()
        month_start = datetime(now.year, now.month, 1)
        
        if now.month == 12:
            month_end = datetime(now.year + 1, 1, 1)
        else:
            month_end = datetime(now.year, now.month + 1, 1)
        
        if resource_type:
            usage = await self.usage_repository.aggregate_usage(
                tenant_id=tenant_id,
                resource_type=resource_type,
                start_time=month_start,
                end_time=month_end,
            )
            return {resource_type: usage}
        
        records = await self.usage_repository.get_by_tenant(
            tenant_id=tenant_id,
            start_time=month_start,
            end_time=month_end,
        )
        
        usage_by_type: Dict[str, Decimal] = {}
        for record in records:
            if record.resource_type not in usage_by_type:
                usage_by_type[record.resource_type] = Decimal("0")
            usage_by_type[record.resource_type] += record.quantity
        
        return usage_by_type
    
    async def get_usage_for_period(
        self,
        tenant_id: UUID,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Decimal]:
        """
        Get usage aggregated by resource type for a time period.
        
        Args:
            tenant_id: Tenant identifier
            start_time: Start of period
            end_time: End of period
            
        Returns:
            Dictionary of resource type to usage quantity
        """
        records = await self.usage_repository.get_by_tenant(
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time,
        )
        
        usage_by_type: Dict[str, Decimal] = {}
        for record in records:
            if record.resource_type not in usage_by_type:
                usage_by_type[record.resource_type] = Decimal("0")
            usage_by_type[record.resource_type] += record.quantity
        
        return usage_by_type


class BillingService:
    """
    Domain service for billing operations.
    
    Manages subscription lifecycle, upgrades, and invoice generation.
    """
    
    def __init__(
        self,
        subscription_repository: SubscriptionRepository,
        invoice_repository: InvoiceRepository,
        usage_repository: UsageRecordRepository,
    ):
        """
        Initialize billing service.
        
        Args:
            subscription_repository: Repository for subscriptions
            invoice_repository: Repository for invoices
            usage_repository: Repository for usage records
        """
        self.subscription_repository = subscription_repository
        self.invoice_repository = invoice_repository
        self.usage_repository = usage_repository
    
    async def create_subscription(
        self,
        tenant_id: UUID,
        tier: SubscriptionTier,
        billing_cycle: str = "monthly",
        stripe_subscription_id: Optional[str] = None,
        stripe_customer_id: Optional[str] = None,
        trial: bool = False,
        trial_days: int = 14,
    ) -> Subscription:
        """
        Create a new subscription for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            tier: Subscription tier
            billing_cycle: Billing cycle (monthly or yearly)
            stripe_subscription_id: Optional Stripe subscription ID
            stripe_customer_id: Optional Stripe customer ID
            trial: Whether to start as trial
            trial_days: Number of trial days if trial is True
            
        Returns:
            Created subscription
        """
        existing = await self.subscription_repository.get_by_tenant(tenant_id)
        if existing and existing.is_active():
            raise ValueError(
                f"Cannot create subscription: Tenant {tenant_id} already has an active subscription. "
                f"Cancel or modify the existing subscription first."
            )
        
        if trial:
            subscription = Subscription.create_trial(
                tenant_id=tenant_id,
                tier=tier,
                trial_days=trial_days,
            )
        else:
            subscription = Subscription.create(
                tenant_id=tenant_id,
                tier=tier,
                billing_cycle=billing_cycle,
                stripe_subscription_id=stripe_subscription_id,
                stripe_customer_id=stripe_customer_id,
            )
        
        return await self.subscription_repository.create(subscription)
    
    async def upgrade_subscription(
        self,
        tenant_id: UUID,
        new_tier: SubscriptionTier,
    ) -> Subscription:
        """
        Upgrade a tenant's subscription to a higher tier.
        
        Args:
            tenant_id: Tenant identifier
            new_tier: New subscription tier
            
        Returns:
            Updated subscription
        """
        subscription = await self.subscription_repository.get_by_tenant(tenant_id)
        
        if not subscription:
            raise ValueError(
                f"No subscription found for tenant {tenant_id}. "
                f"Create a subscription before attempting to upgrade."
            )
        
        if not subscription.is_active():
            raise ValueError(f"Cannot upgrade inactive subscription (status: {subscription.status.value})")
        
        tier_order = {
            SubscriptionTier.FREE: 0,
            SubscriptionTier.PRO: 1,
            SubscriptionTier.TEAM: 2,
            SubscriptionTier.ENTERPRISE: 3,
        }
        
        current_level = tier_order.get(subscription.tier, -1)
        new_level = tier_order.get(new_tier, -1)
        
        if new_level <= current_level:
            raise ValueError(
                f"Cannot upgrade from {subscription.tier.value} to {new_tier.value}. "
                f"Use downgrade for lower tiers."
            )
        
        subscription.upgrade(new_tier)
        
        return await self.subscription_repository.update(subscription)
    
    async def downgrade_subscription(
        self,
        tenant_id: UUID,
        new_tier: SubscriptionTier,
        immediate: bool = False,
    ) -> Subscription:
        """
        Downgrade a tenant's subscription to a lower tier.
        
        Args:
            tenant_id: Tenant identifier
            new_tier: New subscription tier
            immediate: Whether to apply immediately or at period end
            
        Returns:
            Updated subscription
        """
        subscription = await self.subscription_repository.get_by_tenant(tenant_id)
        
        if not subscription:
            raise ValueError(f"No subscription found for tenant {tenant_id}")
        
        tier_order = {
            SubscriptionTier.FREE: 0,
            SubscriptionTier.PRO: 1,
            SubscriptionTier.TEAM: 2,
            SubscriptionTier.ENTERPRISE: 3,
        }
        
        current_level = tier_order.get(subscription.tier, -1)
        new_level = tier_order.get(new_tier, -1)
        
        if new_level >= current_level:
            raise ValueError(
                f"Cannot downgrade from {subscription.tier.value} to {new_tier.value}. "
                f"Use upgrade for higher tiers."
            )
        
        subscription.downgrade(new_tier)
        
        if immediate:
            subscription.updated_at = datetime.utcnow()
        
        return await self.subscription_repository.update(subscription)
    
    async def cancel_subscription(
        self,
        tenant_id: UUID,
        immediate: bool = False,
    ) -> Subscription:
        """
        Cancel a tenant's subscription.
        
        Args:
            tenant_id: Tenant identifier
            immediate: Whether to cancel immediately or at period end
            
        Returns:
            Updated subscription
        """
        subscription = await self.subscription_repository.get_by_tenant(tenant_id)
        
        if not subscription:
            raise ValueError(f"No subscription found for tenant {tenant_id}")
        
        if subscription.status == SubscriptionStatus.CANCELED:
            raise ValueError("Subscription is already canceled")
        
        subscription.cancel(immediate=immediate)
        
        return await self.subscription_repository.update(subscription)
    
    async def reactivate_subscription(self, tenant_id: UUID) -> Subscription:
        """
        Reactivate a canceled subscription.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Updated subscription
        """
        subscription = await self.subscription_repository.get_by_tenant(tenant_id)
        
        if not subscription:
            raise ValueError(f"No subscription found for tenant {tenant_id}")
        
        if subscription.status != SubscriptionStatus.CANCELED:
            raise ValueError("Only canceled subscriptions can be reactivated")
        
        subscription.reactivate()
        
        return await self.subscription_repository.update(subscription)
    
    async def generate_invoice(
        self,
        tenant_id: UUID,
        subscription_id: Optional[UUID] = None,
    ) -> Invoice:
        """
        Generate an invoice for a tenant's subscription.
        
        Args:
            tenant_id: Tenant identifier
            subscription_id: Optional specific subscription ID
            
        Returns:
            Created invoice
        """
        if subscription_id:
            subscription = await self.subscription_repository.get_by_id(subscription_id)
        else:
            subscription = await self.subscription_repository.get_by_tenant(tenant_id)
        
        if not subscription:
            raise ValueError(f"No subscription found for tenant {tenant_id}")
        
        if subscription.tenant_id != tenant_id:
            raise ValueError(f"Subscription {subscription_id} does not belong to tenant {tenant_id}")
        
        invoice = Invoice.create(
            tenant_id=tenant_id,
            subscription_id=subscription.id,
            amount_cents=subscription.amount_cents,
            period_start=subscription.current_period_start,
            period_end=subscription.current_period_end,
        )
        
        usage_data = await self.usage_repository.get_by_tenant(
            tenant_id=tenant_id,
            start_time=subscription.current_period_start,
            end_time=subscription.current_period_end,
        )
        
        line_items = []
        line_items.append({
            "description": f"{subscription.tier.value.title()} Plan - {subscription.billing_cycle}",
            "amount_cents": subscription.amount_cents,
            "quantity": 1,
        })
        
        usage_by_type: Dict[str, Decimal] = {}
        for record in usage_data:
            if record.resource_type not in usage_by_type:
                usage_by_type[record.resource_type] = Decimal("0")
            usage_by_type[record.resource_type] += record.quantity
        
        for resource_type, quantity in usage_by_type.items():
            line_items.append({
                "description": f"Usage: {resource_type}",
                "quantity": str(quantity),
                "amount_cents": 0,
            })
        
        invoice.line_items = line_items
        
        return await self.invoice_repository.create(invoice)
    
    async def mark_invoice_paid(self, invoice_id: UUID) -> Invoice:
        """
        Mark an invoice as paid.
        
        Args:
            invoice_id: Invoice identifier
            
        Returns:
            Updated invoice
        """
        invoice = await self.invoice_repository.get_by_id(invoice_id)
        
        if not invoice:
            raise ValueError(f"Invoice {invoice_id} not found")
        
        if invoice.status == InvoiceStatus.PAID:
            raise ValueError("Invoice is already marked as paid")
        
        invoice.mark_paid()
        
        return await self.invoice_repository.update(invoice)
    
    async def void_invoice(self, invoice_id: UUID) -> Invoice:
        """
        Void an invoice.
        
        Args:
            invoice_id: Invoice identifier
            
        Returns:
            Updated invoice
        """
        invoice = await self.invoice_repository.get_by_id(invoice_id)
        
        if not invoice:
            raise ValueError(f"Invoice {invoice_id} not found")
        
        if invoice.status == InvoiceStatus.PAID:
            raise ValueError("Cannot void a paid invoice")
        
        invoice.mark_void()
        
        return await self.invoice_repository.update(invoice)
    
    async def handle_trial_expiration(self, tenant_id: UUID) -> Subscription:
        """
        Handle trial subscription expiration.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Updated subscription
        """
        subscription = await self.subscription_repository.get_by_tenant(tenant_id)
        
        if not subscription:
            raise ValueError(f"No subscription found for tenant {tenant_id}")
        
        if not subscription.is_trial():
            raise ValueError("Subscription is not in trial period")
        
        if not subscription.trial_expired():
            raise ValueError("Trial period has not expired yet")
        
        if subscription.payment_method:
            subscription.status = SubscriptionStatus.ACTIVE
            now = datetime.utcnow()
            
            if subscription.billing_cycle == "yearly":
                subscription.current_period_end = now + timedelta(days=365)
            else:
                subscription.current_period_end = now + timedelta(days=30)
            
            subscription.amount_cents = Subscription._calculate_amount(
                subscription.tier,
                subscription.billing_cycle
            )
        else:
            subscription.status = SubscriptionStatus.UNPAID
        
        subscription.updated_at = datetime.utcnow()
        
        return await self.subscription_repository.update(subscription)
