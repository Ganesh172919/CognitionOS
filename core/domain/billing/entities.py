"""Billing domain entities."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4


class SubscriptionStatus(str, Enum):
    """Subscription status enumeration."""
    
    ACTIVE = "active"
    TRIALING = "trialing"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    UNPAID = "unpaid"
    PAUSED = "paused"


class SubscriptionTier(str, Enum):
    """Subscription tier enumeration."""
    
    FREE = "free"
    PRO = "pro"
    TEAM = "team"
    ENTERPRISE = "enterprise"


class InvoiceStatus(str, Enum):
    """Invoice status enumeration."""
    
    DRAFT = "draft"
    OPEN = "open"
    PAID = "paid"
    VOID = "void"
    UNCOLLECTIBLE = "uncollectible"


@dataclass
class PaymentMethod:
    """Payment method details."""
    
    id: str
    type: str  # 'card', 'bank_account', 'paypal'
    last4: str
    brand: Optional[str] = None
    exp_month: Optional[int] = None
    exp_year: Optional[int] = None
    is_default: bool = False


@dataclass
class Subscription:
    """
    Subscription entity representing a tenant's billing subscription.
    
    Manages subscription lifecycle, tier changes, and billing cycles.
    """
    
    id: UUID
    tenant_id: UUID
    tier: SubscriptionTier
    status: SubscriptionStatus
    stripe_subscription_id: Optional[str]
    stripe_customer_id: Optional[str]
    current_period_start: datetime
    current_period_end: datetime
    trial_start: Optional[datetime]
    trial_end: Optional[datetime]
    canceled_at: Optional[datetime]
    cancel_at_period_end: bool
    amount_cents: int  # Amount in cents
    currency: str
    billing_cycle: str  # 'monthly', 'yearly'
    payment_method: Optional[PaymentMethod]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create_trial(
        cls,
        tenant_id: UUID,
        tier: SubscriptionTier = SubscriptionTier.PRO,
        trial_days: int = 14,
    ) -> "Subscription":
        """Create a new trial subscription."""
        now = datetime.utcnow()
        from datetime import timedelta
        
        trial_end = now + timedelta(days=trial_days)
        
        # Trial pricing (free during trial)
        amount_cents = 0
        
        return cls(
            id=uuid4(),
            tenant_id=tenant_id,
            tier=tier,
            status=SubscriptionStatus.TRIALING,
            stripe_subscription_id=None,
            stripe_customer_id=None,
            current_period_start=now,
            current_period_end=trial_end,
            trial_start=now,
            trial_end=trial_end,
            canceled_at=None,
            cancel_at_period_end=False,
            amount_cents=amount_cents,
            currency="usd",
            billing_cycle="monthly",
            payment_method=None,
            created_at=now,
            updated_at=now,
        )
    
    @classmethod
    def create(
        cls,
        tenant_id: UUID,
        tier: SubscriptionTier,
        billing_cycle: str = "monthly",
        stripe_subscription_id: Optional[str] = None,
        stripe_customer_id: Optional[str] = None,
    ) -> "Subscription":
        """Create a new active subscription."""
        now = datetime.utcnow()
        from datetime import timedelta
        
        # Calculate billing period
        if billing_cycle == "yearly":
            period_end = now + timedelta(days=365)
        else:
            period_end = now + timedelta(days=30)
        
        # Calculate amount based on tier and cycle
        amount_cents = cls._calculate_amount(tier, billing_cycle)
        
        return cls(
            id=uuid4(),
            tenant_id=tenant_id,
            tier=tier,
            status=SubscriptionStatus.ACTIVE,
            stripe_subscription_id=stripe_subscription_id,
            stripe_customer_id=stripe_customer_id,
            current_period_start=now,
            current_period_end=period_end,
            trial_start=None,
            trial_end=None,
            canceled_at=None,
            cancel_at_period_end=False,
            amount_cents=amount_cents,
            currency="usd",
            billing_cycle=billing_cycle,
            payment_method=None,
            created_at=now,
            updated_at=now,
        )
    
    @staticmethod
    def _calculate_amount(tier: SubscriptionTier, billing_cycle: str) -> int:
        """Calculate subscription amount in cents."""
        # Monthly pricing in cents
        monthly_prices = {
            SubscriptionTier.FREE: 0,
            SubscriptionTier.PRO: 4900,  # $49/month
            SubscriptionTier.TEAM: 19900,  # $199/month
            SubscriptionTier.ENTERPRISE: 0,  # Custom pricing
        }
        
        monthly_amount = monthly_prices.get(tier, 0)
        
        if billing_cycle == "yearly":
            # 2 months free for annual billing
            return monthly_amount * 10
        
        return monthly_amount
    
    def upgrade(self, new_tier: SubscriptionTier) -> None:
        """Upgrade to a higher tier."""
        self.tier = new_tier
        self.amount_cents = self._calculate_amount(new_tier, self.billing_cycle)
        self.updated_at = datetime.utcnow()
    
    def downgrade(self, new_tier: SubscriptionTier) -> None:
        """Downgrade to a lower tier (applies at period end)."""
        self.tier = new_tier
        self.amount_cents = self._calculate_amount(new_tier, self.billing_cycle)
        self.updated_at = datetime.utcnow()
    
    def cancel(self, immediate: bool = False) -> None:
        """Cancel subscription."""
        now = datetime.utcnow()
        self.canceled_at = now
        
        if immediate:
            self.status = SubscriptionStatus.CANCELED
            self.current_period_end = now
        else:
            self.cancel_at_period_end = True
        
        self.updated_at = now
    
    def reactivate(self) -> None:
        """Reactivate a canceled subscription."""
        self.status = SubscriptionStatus.ACTIVE
        self.canceled_at = None
        self.cancel_at_period_end = False
        self.updated_at = datetime.utcnow()
    
    def mark_past_due(self) -> None:
        """Mark subscription as past due."""
        self.status = SubscriptionStatus.PAST_DUE
        self.updated_at = datetime.utcnow()
    
    def is_active(self) -> bool:
        """Check if subscription is active."""
        return self.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]
    
    def is_trial(self) -> bool:
        """Check if subscription is in trial."""
        return self.status == SubscriptionStatus.TRIALING
    
    def trial_expired(self) -> bool:
        """Check if trial has expired."""
        if self.trial_end is None:
            return False
        return datetime.utcnow() > self.trial_end


@dataclass
class Invoice:
    """Invoice entity for billing records."""
    
    id: UUID
    tenant_id: UUID
    subscription_id: UUID
    status: InvoiceStatus
    amount_cents: int
    amount_paid_cents: int
    amount_due_cents: int
    currency: str
    stripe_invoice_id: Optional[str]
    invoice_number: str
    period_start: datetime
    period_end: datetime
    due_date: Optional[datetime]
    paid_at: Optional[datetime]
    created_at: datetime
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        tenant_id: UUID,
        subscription_id: UUID,
        amount_cents: int,
        period_start: datetime,
        period_end: datetime,
    ) -> "Invoice":
        """Create a new invoice."""
        now = datetime.utcnow()
        from datetime import timedelta
        
        invoice_number = f"INV-{now.year}{now.month:02d}-{uuid4().hex[:8].upper()}"
        
        return cls(
            id=uuid4(),
            tenant_id=tenant_id,
            subscription_id=subscription_id,
            status=InvoiceStatus.OPEN,
            amount_cents=amount_cents,
            amount_paid_cents=0,
            amount_due_cents=amount_cents,
            currency="usd",
            stripe_invoice_id=None,
            invoice_number=invoice_number,
            period_start=period_start,
            period_end=period_end,
            due_date=now + timedelta(days=7),
            paid_at=None,
            created_at=now,
        )
    
    def mark_paid(self) -> None:
        """Mark invoice as paid."""
        self.status = InvoiceStatus.PAID
        self.amount_paid_cents = self.amount_due_cents
        self.amount_due_cents = 0
        self.paid_at = datetime.utcnow()
    
    def mark_void(self) -> None:
        """Void the invoice."""
        self.status = InvoiceStatus.VOID
        self.amount_due_cents = 0


@dataclass
class UsageRecord:
    """Usage metering record for tracking consumption."""
    
    id: UUID
    tenant_id: UUID
    resource_type: str  # 'executions', 'tokens', 'storage', 'api_calls'
    quantity: Decimal
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        tenant_id: UUID,
        resource_type: str,
        quantity: Decimal,
        unit: str = "count",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "UsageRecord":
        """Create a new usage record."""
        return cls(
            id=uuid4(),
            tenant_id=tenant_id,
            resource_type=resource_type,
            quantity=quantity,
            unit=unit,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )


@dataclass
class EntitlementCheck:
    """Result of an entitlement check."""
    
    allowed: bool
    reason: Optional[str] = None
    current_usage: Optional[Decimal] = None
    limit: Optional[Decimal] = None
    remaining: Optional[Decimal] = None
