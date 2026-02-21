"""
Billing Integration Layer

Abstract billing provider interface with Stripe and Paddle implementations.
Supports subscription management, usage-based billing, and payment processing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)


class BillingProviderType(str, Enum):
    """Supported billing providers"""
    STRIPE = "stripe"
    PADDLE = "paddle"
    CHARGEBEE = "chargebee"


class PaymentStatus(str, Enum):
    """Payment status"""
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REFUNDED = "refunded"
    DISPUTED = "disputed"


class SubscriptionStatus(str, Enum):
    """Subscription status"""
    ACTIVE = "active"
    TRIALING = "trialing"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    UNPAID = "unpaid"


@dataclass
class PaymentMethod:
    """Payment method information"""
    id: str
    type: str  # card, bank_account, etc.
    last4: Optional[str] = None
    brand: Optional[str] = None
    exp_month: Optional[int] = None
    exp_year: Optional[int] = None
    is_default: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Invoice:
    """Invoice information"""
    id: str
    tenant_id: str
    amount_due: float
    amount_paid: float
    currency: str
    status: PaymentStatus
    created_at: datetime
    due_date: datetime
    paid_at: Optional[datetime] = None
    invoice_url: Optional[str] = None
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BillingSubscription:
    """Subscription information"""
    id: str
    tenant_id: str
    plan_id: str
    status: SubscriptionStatus
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool = False
    canceled_at: Optional[datetime] = None
    trial_end: Optional[datetime] = None
    amount: float = 0.0
    currency: str = "usd"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BillingProvider(ABC):
    """Abstract billing provider interface"""

    @abstractmethod
    async def create_customer(
        self,
        tenant_id: str,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a customer in the billing system"""
        pass

    @abstractmethod
    async def create_subscription(
        self,
        tenant_id: str,
        plan_id: str,
        payment_method_id: Optional[str] = None,
        trial_days: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BillingSubscription:
        """Create a new subscription"""
        pass

    @abstractmethod
    async def update_subscription(
        self,
        subscription_id: str,
        plan_id: Optional[str] = None,
        payment_method_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BillingSubscription:
        """Update an existing subscription"""
        pass

    @abstractmethod
    async def cancel_subscription(
        self,
        subscription_id: str,
        immediate: bool = False
    ) -> BillingSubscription:
        """Cancel a subscription"""
        pass

    @abstractmethod
    async def get_subscription(self, subscription_id: str) -> BillingSubscription:
        """Get subscription details"""
        pass

    @abstractmethod
    async def add_payment_method(
        self,
        tenant_id: str,
        payment_method_token: str,
        set_default: bool = True
    ) -> PaymentMethod:
        """Add a payment method"""
        pass

    @abstractmethod
    async def create_invoice(
        self,
        tenant_id: str,
        line_items: List[Dict[str, Any]],
        due_date: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Invoice:
        """Create an invoice for usage-based billing"""
        pass

    @abstractmethod
    async def record_usage(
        self,
        subscription_id: str,
        usage_records: List[Dict[str, Any]]
    ) -> bool:
        """Record usage for metered billing"""
        pass


class StripeProvider(BillingProvider):
    """Stripe billing provider implementation"""

    def __init__(self, api_key: str, webhook_secret: Optional[str] = None):
        self.api_key = api_key
        self.webhook_secret = webhook_secret
        # In production, would initialize Stripe client
        # import stripe
        # stripe.api_key = api_key

    async def create_customer(
        self,
        tenant_id: str,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create Stripe customer"""
        logger.info(f"Creating Stripe customer for tenant {tenant_id}")

        # Stripe implementation would be:
        # customer = stripe.Customer.create(
        #     email=email,
        #     name=name,
        #     metadata={'tenant_id': tenant_id, **(metadata or {})}
        # )
        # return customer.id

        # Mock implementation
        return f"cus_{tenant_id}_stripe"

    async def create_subscription(
        self,
        tenant_id: str,
        plan_id: str,
        payment_method_id: Optional[str] = None,
        trial_days: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BillingSubscription:
        """Create Stripe subscription"""
        logger.info(f"Creating Stripe subscription for tenant {tenant_id}, plan {plan_id}")

        # Stripe implementation:
        # customer_id = await self._get_or_create_customer(tenant_id)
        # subscription = stripe.Subscription.create(
        #     customer=customer_id,
        #     items=[{'price': plan_id}],
        #     default_payment_method=payment_method_id,
        #     trial_period_days=trial_days if trial_days > 0 else None,
        #     metadata={'tenant_id': tenant_id, **(metadata or {})}
        # )

        # Mock implementation
        now = datetime.utcnow()
        return BillingSubscription(
            id=f"sub_{tenant_id}_stripe",
            tenant_id=tenant_id,
            plan_id=plan_id,
            status=SubscriptionStatus.TRIALING if trial_days > 0 else SubscriptionStatus.ACTIVE,
            current_period_start=now,
            current_period_end=now + timedelta(days=30),
            trial_end=now + timedelta(days=trial_days) if trial_days > 0 else None,
            metadata=metadata or {}
        )

    async def update_subscription(
        self,
        subscription_id: str,
        plan_id: Optional[str] = None,
        payment_method_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BillingSubscription:
        """Update Stripe subscription"""
        logger.info(f"Updating Stripe subscription {subscription_id}")

        # Stripe implementation:
        # update_params = {}
        # if plan_id:
        #     subscription = stripe.Subscription.retrieve(subscription_id)
        #     update_params['items'] = [{'id': subscription['items']['data'][0].id, 'price': plan_id}]
        #     update_params['proration_behavior'] = 'always_invoice'
        # if payment_method_id:
        #     update_params['default_payment_method'] = payment_method_id
        # if metadata:
        #     update_params['metadata'] = metadata
        #
        # updated_subscription = stripe.Subscription.modify(subscription_id, **update_params)

        # Mock implementation
        return await self.get_subscription(subscription_id)

    async def cancel_subscription(
        self,
        subscription_id: str,
        immediate: bool = False
    ) -> BillingSubscription:
        """Cancel Stripe subscription"""
        logger.info(f"Canceling Stripe subscription {subscription_id}, immediate={immediate}")

        # Stripe implementation:
        # if immediate:
        #     subscription = stripe.Subscription.delete(subscription_id)
        # else:
        #     subscription = stripe.Subscription.modify(
        #         subscription_id,
        #         cancel_at_period_end=True
        #     )

        subscription = await self.get_subscription(subscription_id)
        subscription.cancel_at_period_end = not immediate
        subscription.canceled_at = datetime.utcnow() if immediate else None
        subscription.status = SubscriptionStatus.CANCELED if immediate else SubscriptionStatus.ACTIVE

        return subscription

    async def get_subscription(self, subscription_id: str) -> BillingSubscription:
        """Get Stripe subscription"""
        # Stripe implementation:
        # subscription = stripe.Subscription.retrieve(subscription_id)

        # Mock implementation
        now = datetime.utcnow()
        return BillingSubscription(
            id=subscription_id,
            tenant_id="tenant_123",
            plan_id="plan_pro",
            status=SubscriptionStatus.ACTIVE,
            current_period_start=now,
            current_period_end=now + timedelta(days=30)
        )

    async def add_payment_method(
        self,
        tenant_id: str,
        payment_method_token: str,
        set_default: bool = True
    ) -> PaymentMethod:
        """Add Stripe payment method"""
        logger.info(f"Adding payment method for tenant {tenant_id}")

        # Stripe implementation:
        # customer_id = await self._get_or_create_customer(tenant_id)
        # payment_method = stripe.PaymentMethod.attach(
        #     payment_method_token,
        #     customer=customer_id
        # )
        # if set_default:
        #     stripe.Customer.modify(
        #         customer_id,
        #         invoice_settings={'default_payment_method': payment_method.id}
        #     )

        return PaymentMethod(
            id=payment_method_token,
            type="card",
            last4="4242",
            brand="visa",
            is_default=set_default
        )

    async def create_invoice(
        self,
        tenant_id: str,
        line_items: List[Dict[str, Any]],
        due_date: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Invoice:
        """Create Stripe invoice"""
        logger.info(f"Creating invoice for tenant {tenant_id}")

        # Stripe implementation:
        # customer_id = await self._get_or_create_customer(tenant_id)
        # invoice = stripe.Invoice.create(
        #     customer=customer_id,
        #     auto_advance=True,
        #     collection_method='charge_automatically',
        #     due_date=int(due_date.timestamp()) if due_date else None,
        #     metadata={'tenant_id': tenant_id, **(metadata or {})}
        # )
        #
        # for item in line_items:
        #     stripe.InvoiceItem.create(
        #         customer=customer_id,
        #         invoice=invoice.id,
        #         **item
        #     )
        #
        # invoice = stripe.Invoice.finalize_invoice(invoice.id)

        total_amount = sum(item.get("amount", 0) for item in line_items)
        now = datetime.utcnow()

        return Invoice(
            id=f"inv_{tenant_id}_stripe",
            tenant_id=tenant_id,
            amount_due=total_amount,
            amount_paid=0.0,
            currency="usd",
            status=PaymentStatus.PENDING,
            created_at=now,
            due_date=due_date or now + timedelta(days=30),
            line_items=line_items,
            metadata=metadata or {}
        )

    async def record_usage(
        self,
        subscription_id: str,
        usage_records: List[Dict[str, Any]]
    ) -> bool:
        """Record usage for Stripe metered billing"""
        logger.info(f"Recording usage for subscription {subscription_id}")

        # Stripe implementation:
        # for record in usage_records:
        #     stripe.SubscriptionItem.create_usage_record(
        #         record['subscription_item_id'],
        #         quantity=record['quantity'],
        #         timestamp=int(record['timestamp'].timestamp()),
        #         action='increment'
        #     )

        return True


class PaddleProvider(BillingProvider):
    """Paddle billing provider implementation"""

    def __init__(self, vendor_id: str, vendor_auth_code: str, sandbox: bool = False):
        self.vendor_id = vendor_id
        self.vendor_auth_code = vendor_auth_code
        self.sandbox = sandbox
        # In production, would initialize Paddle client

    async def create_customer(
        self,
        tenant_id: str,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create Paddle customer"""
        logger.info(f"Creating Paddle customer for tenant {tenant_id}")
        return f"cus_{tenant_id}_paddle"

    async def create_subscription(
        self,
        tenant_id: str,
        plan_id: str,
        payment_method_id: Optional[str] = None,
        trial_days: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BillingSubscription:
        """Create Paddle subscription"""
        logger.info(f"Creating Paddle subscription for tenant {tenant_id}")

        now = datetime.utcnow()
        return BillingSubscription(
            id=f"sub_{tenant_id}_paddle",
            tenant_id=tenant_id,
            plan_id=plan_id,
            status=SubscriptionStatus.ACTIVE,
            current_period_start=now,
            current_period_end=now + timedelta(days=30),
            metadata=metadata or {}
        )

    async def update_subscription(
        self,
        subscription_id: str,
        plan_id: Optional[str] = None,
        payment_method_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BillingSubscription:
        """Update Paddle subscription"""
        return await self.get_subscription(subscription_id)

    async def cancel_subscription(
        self,
        subscription_id: str,
        immediate: bool = False
    ) -> BillingSubscription:
        """Cancel Paddle subscription"""
        subscription = await self.get_subscription(subscription_id)
        subscription.status = SubscriptionStatus.CANCELED
        return subscription

    async def get_subscription(self, subscription_id: str) -> BillingSubscription:
        """Get Paddle subscription"""
        now = datetime.utcnow()
        return BillingSubscription(
            id=subscription_id,
            tenant_id="tenant_123",
            plan_id="plan_pro",
            status=SubscriptionStatus.ACTIVE,
            current_period_start=now,
            current_period_end=now + timedelta(days=30)
        )

    async def add_payment_method(
        self,
        tenant_id: str,
        payment_method_token: str,
        set_default: bool = True
    ) -> PaymentMethod:
        """Add Paddle payment method"""
        return PaymentMethod(
            id=payment_method_token,
            type="card",
            is_default=set_default
        )

    async def create_invoice(
        self,
        tenant_id: str,
        line_items: List[Dict[str, Any]],
        due_date: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Invoice:
        """Create Paddle invoice"""
        total_amount = sum(item.get("amount", 0) for item in line_items)
        now = datetime.utcnow()

        return Invoice(
            id=f"inv_{tenant_id}_paddle",
            tenant_id=tenant_id,
            amount_due=total_amount,
            amount_paid=0.0,
            currency="usd",
            status=PaymentStatus.PENDING,
            created_at=now,
            due_date=due_date or now + timedelta(days=30),
            line_items=line_items
        )

    async def record_usage(
        self,
        subscription_id: str,
        usage_records: List[Dict[str, Any]]
    ) -> bool:
        """Record usage for Paddle"""
        return True


class BillingOrchestrator:
    """
    Orchestrates billing operations across providers

    Handles provider selection, failover, and unified billing operations.
    """

    def __init__(self, primary_provider: BillingProvider):
        self.primary_provider = primary_provider
        self.fallback_providers: List[BillingProvider] = []

    def add_fallback_provider(self, provider: BillingProvider):
        """Add a fallback billing provider"""
        self.fallback_providers.append(provider)

    async def create_subscription(
        self,
        tenant_id: str,
        plan_id: str,
        **kwargs
    ) -> BillingSubscription:
        """Create subscription with automatic failover"""
        try:
            return await self.primary_provider.create_subscription(
                tenant_id=tenant_id,
                plan_id=plan_id,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Primary provider failed: {e}")

            # Try fallback providers
            for provider in self.fallback_providers:
                try:
                    logger.info(f"Attempting fallback provider: {type(provider).__name__}")
                    return await provider.create_subscription(
                        tenant_id=tenant_id,
                        plan_id=plan_id,
                        **kwargs
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback provider failed: {fallback_error}")
                    continue

            # All providers failed
            raise Exception("All billing providers failed")

    async def handle_upgrade(
        self,
        tenant_id: str,
        current_subscription_id: str,
        new_plan_id: str,
        proration: bool = True
    ) -> BillingSubscription:
        """Handle subscription upgrade"""
        logger.info(f"Upgrading subscription {current_subscription_id} to plan {new_plan_id}")

        # Calculate proration if needed
        if proration:
            # Would calculate prorated amount
            pass

        return await self.primary_provider.update_subscription(
            subscription_id=current_subscription_id,
            plan_id=new_plan_id
        )

    async def handle_downgrade(
        self,
        tenant_id: str,
        current_subscription_id: str,
        new_plan_id: str,
        immediate: bool = False
    ) -> BillingSubscription:
        """Handle subscription downgrade"""
        logger.info(f"Downgrading subscription {current_subscription_id} to plan {new_plan_id}")

        if immediate:
            # Immediate downgrade
            return await self.primary_provider.update_subscription(
                subscription_id=current_subscription_id,
                plan_id=new_plan_id
            )
        else:
            # Schedule downgrade for end of billing period
            # Would need to store scheduled change
            logger.info("Scheduled downgrade at end of billing period")
            return await self.primary_provider.get_subscription(current_subscription_id)

    async def process_usage_billing(
        self,
        tenant_id: str,
        subscription_id: str,
        usage_data: Dict[str, float],
        unit_prices: Dict[str, float]
    ) -> Invoice:
        """Process usage-based billing"""
        logger.info(f"Processing usage billing for tenant {tenant_id}")

        # Create invoice line items from usage
        line_items = []
        for resource, quantity in usage_data.items():
            unit_price = unit_prices.get(resource, 0.0)
            amount = quantity * unit_price

            line_items.append({
                "description": f"{resource} usage",
                "quantity": quantity,
                "unit_amount": unit_price,
                "amount": amount
            })

        return await self.primary_provider.create_invoice(
            tenant_id=tenant_id,
            line_items=line_items
        )

    async def handle_failed_payment(
        self,
        tenant_id: str,
        subscription_id: str,
        invoice_id: str
    ) -> Dict[str, Any]:
        """Handle failed payment with retry logic"""
        logger.warning(f"Handling failed payment for tenant {tenant_id}")

        # Implement retry logic
        retry_result = {
            "success": False,
            "retries_remaining": 3,
            "next_retry_at": datetime.utcnow() + timedelta(days=1)
        }

        # Would implement actual retry logic with exponential backoff
        # Would send notification to customer
        # Would eventually suspend service if retries exhausted

        return retry_result

    async def generate_revenue_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate revenue report"""
        logger.info(f"Generating revenue report from {start_date} to {end_date}")

        # Would query billing provider for invoices and payments
        report = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "metrics": {
                "total_revenue": 0.0,
                "mrr": 0.0,  # Monthly Recurring Revenue
                "arr": 0.0,  # Annual Recurring Revenue
                "new_customers": 0,
                "churned_customers": 0,
                "upgrades": 0,
                "downgrades": 0
            },
            "by_plan": {},
            "by_region": {}
        }

        return report
