"""
Billing Provider Infrastructure

Abstract billing provider with Stripe implementation and mock provider for development.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List
from uuid import UUID

from core.domain.billing.entities import (
    Subscription,
    SubscriptionTier,
    PaymentMethod,
)

logger = logging.getLogger(__name__)


class BillingProviderError(Exception):
    """Base exception for billing provider errors."""
    pass


class BillingProvider(ABC):
    """
    Abstract base class for billing providers.
    
    Defines interface for external billing system integration.
    """
    
    @abstractmethod
    async def create_customer(
        self,
        tenant_id: UUID,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a customer in the billing system.
        
        Args:
            tenant_id: Internal tenant identifier
            email: Customer email
            name: Customer name
            metadata: Additional metadata
            
        Returns:
            External customer ID
        """
        pass
    
    @abstractmethod
    async def create_subscription(
        self,
        customer_id: str,
        price_id: str,
        trial_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a subscription for a customer.
        
        Args:
            customer_id: External customer ID
            price_id: Price/plan identifier
            trial_days: Optional trial period in days
            metadata: Additional metadata
            
        Returns:
            Subscription data including external subscription ID
        """
        pass
    
    @abstractmethod
    async def update_subscription(
        self,
        subscription_id: str,
        new_price_id: str,
        prorate: bool = True,
    ) -> Dict[str, Any]:
        """
        Update a subscription to a new price/plan.
        
        Args:
            subscription_id: External subscription ID
            new_price_id: New price/plan identifier
            prorate: Whether to prorate the change
            
        Returns:
            Updated subscription data
        """
        pass
    
    @abstractmethod
    async def cancel_subscription(
        self,
        subscription_id: str,
        immediate: bool = False,
    ) -> Dict[str, Any]:
        """
        Cancel a subscription.
        
        Args:
            subscription_id: External subscription ID
            immediate: If True, cancel immediately; if False, at period end
            
        Returns:
            Updated subscription data
        """
        pass
    
    @abstractmethod
    async def create_payment_method(
        self,
        customer_id: str,
        payment_method_data: Dict[str, Any],
    ) -> PaymentMethod:
        """
        Create and attach a payment method to a customer.
        
        Args:
            customer_id: External customer ID
            payment_method_data: Payment method details
            
        Returns:
            PaymentMethod entity
        """
        pass
    
    @abstractmethod
    async def create_invoice(
        self,
        customer_id: str,
        line_items: List[Dict[str, Any]],
        auto_charge: bool = False,
    ) -> Dict[str, Any]:
        """
        Create an invoice for a customer.
        
        Args:
            customer_id: External customer ID
            line_items: List of line items
            auto_charge: Whether to automatically charge
            
        Returns:
            Invoice data
        """
        pass
    
    @abstractmethod
    async def get_subscription(self, subscription_id: str) -> Dict[str, Any]:
        """
        Retrieve subscription details.
        
        Args:
            subscription_id: External subscription ID
            
        Returns:
            Subscription data
        """
        pass
    
    @abstractmethod
    async def get_customer(self, customer_id: str) -> Dict[str, Any]:
        """
        Retrieve customer details.
        
        Args:
            customer_id: External customer ID
            
        Returns:
            Customer data
        """
        pass


class StripeBillingProvider(BillingProvider):
    """
    Stripe implementation of billing provider.
    
    Integrates with Stripe API for payment processing and subscription management.
    """
    
    def __init__(self, api_key: str, webhook_secret: Optional[str] = None):
        """
        Initialize Stripe billing provider.
        
        Args:
            api_key: Stripe API secret key
            webhook_secret: Optional webhook signing secret
        """
        try:
            import stripe
            self.stripe = stripe
            self.stripe.api_key = api_key
            self.webhook_secret = webhook_secret
            logger.info("Stripe billing provider initialized")
        except ImportError:
            logger.error("Stripe library not installed. Run: pip install stripe")
            raise BillingProviderError("Stripe library not available")
    
    async def create_customer(
        self,
        tenant_id: UUID,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a Stripe customer."""
        try:
            customer_metadata = metadata or {}
            customer_metadata["tenant_id"] = str(tenant_id)
            
            customer = await self._async_stripe_call(
                self.stripe.Customer.create,
                email=email,
                name=name,
                metadata=customer_metadata,
            )
            
            logger.info(f"Created Stripe customer: {customer.id} for tenant {tenant_id}")
            return customer.id
            
        except Exception as e:
            logger.error(f"Error creating Stripe customer for tenant {tenant_id}: {e}")
            raise BillingProviderError(f"Failed to create customer: {str(e)}")
    
    async def create_subscription(
        self,
        customer_id: str,
        price_id: str,
        trial_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a Stripe subscription."""
        try:
            params = {
                "customer": customer_id,
                "items": [{"price": price_id}],
                "metadata": metadata or {},
            }
            
            if trial_days is not None:
                params["trial_period_days"] = trial_days
            
            subscription = await self._async_stripe_call(
                self.stripe.Subscription.create,
                **params
            )
            
            logger.info(f"Created Stripe subscription: {subscription.id} for customer {customer_id}")
            
            return {
                "id": subscription.id,
                "status": subscription.status,
                "current_period_start": datetime.fromtimestamp(subscription.current_period_start),
                "current_period_end": datetime.fromtimestamp(subscription.current_period_end),
                "trial_start": datetime.fromtimestamp(subscription.trial_start) if subscription.trial_start else None,
                "trial_end": datetime.fromtimestamp(subscription.trial_end) if subscription.trial_end else None,
                "cancel_at_period_end": subscription.cancel_at_period_end,
            }
            
        except Exception as e:
            logger.error(f"Error creating Stripe subscription for customer {customer_id}: {e}")
            raise BillingProviderError(f"Failed to create subscription: {str(e)}")
    
    async def update_subscription(
        self,
        subscription_id: str,
        new_price_id: str,
        prorate: bool = True,
    ) -> Dict[str, Any]:
        """Update a Stripe subscription."""
        try:
            subscription = await self._async_stripe_call(
                self.stripe.Subscription.retrieve,
                subscription_id
            )
            
            updated = await self._async_stripe_call(
                self.stripe.Subscription.modify,
                subscription_id,
                items=[{
                    "id": subscription["items"]["data"][0].id,
                    "price": new_price_id,
                }],
                proration_behavior="create_prorations" if prorate else "none",
            )
            
            logger.info(f"Updated Stripe subscription: {subscription_id} to price {new_price_id}")
            
            return {
                "id": updated.id,
                "status": updated.status,
                "current_period_start": datetime.fromtimestamp(updated.current_period_start),
                "current_period_end": datetime.fromtimestamp(updated.current_period_end),
            }
            
        except Exception as e:
            logger.error(f"Error updating Stripe subscription {subscription_id}: {e}")
            raise BillingProviderError(f"Failed to update subscription: {str(e)}")
    
    async def cancel_subscription(
        self,
        subscription_id: str,
        immediate: bool = False,
    ) -> Dict[str, Any]:
        """Cancel a Stripe subscription."""
        try:
            if immediate:
                subscription = await self._async_stripe_call(
                    self.stripe.Subscription.cancel,
                    subscription_id
                )
            else:
                subscription = await self._async_stripe_call(
                    self.stripe.Subscription.modify,
                    subscription_id,
                    cancel_at_period_end=True
                )
            
            logger.info(f"Canceled Stripe subscription: {subscription_id} (immediate={immediate})")
            
            return {
                "id": subscription.id,
                "status": subscription.status,
                "canceled_at": datetime.fromtimestamp(subscription.canceled_at) if subscription.canceled_at else None,
                "cancel_at_period_end": subscription.cancel_at_period_end,
            }
            
        except Exception as e:
            logger.error(f"Error canceling Stripe subscription {subscription_id}: {e}")
            raise BillingProviderError(f"Failed to cancel subscription: {str(e)}")
    
    async def create_payment_method(
        self,
        customer_id: str,
        payment_method_data: Dict[str, Any],
    ) -> PaymentMethod:
        """Create and attach a payment method."""
        try:
            payment_method = await self._async_stripe_call(
                self.stripe.PaymentMethod.create,
                type=payment_method_data.get("type", "card"),
                card=payment_method_data.get("card"),
            )
            
            await self._async_stripe_call(
                self.stripe.PaymentMethod.attach,
                payment_method.id,
                customer=customer_id
            )
            
            await self._async_stripe_call(
                self.stripe.Customer.modify,
                customer_id,
                invoice_settings={
                    "default_payment_method": payment_method.id
                }
            )
            
            logger.info(f"Created payment method: {payment_method.id} for customer {customer_id}")
            
            return PaymentMethod(
                id=payment_method.id,
                type=payment_method.type,
                last4=payment_method.card.last4 if payment_method.card else "",
                brand=payment_method.card.brand if payment_method.card else None,
                exp_month=payment_method.card.exp_month if payment_method.card else None,
                exp_year=payment_method.card.exp_year if payment_method.card else None,
                is_default=True,
            )
            
        except Exception as e:
            logger.error(f"Error creating payment method for customer {customer_id}: {e}")
            raise BillingProviderError(f"Failed to create payment method: {str(e)}")
    
    async def create_invoice(
        self,
        customer_id: str,
        line_items: List[Dict[str, Any]],
        auto_charge: bool = False,
    ) -> Dict[str, Any]:
        """Create a Stripe invoice."""
        try:
            invoice = await self._async_stripe_call(
                self.stripe.Invoice.create,
                customer=customer_id,
                auto_advance=auto_charge,
            )
            
            for item in line_items:
                await self._async_stripe_call(
                    self.stripe.InvoiceItem.create,
                    customer=customer_id,
                    invoice=invoice.id,
                    description=item.get("description"),
                    amount=item.get("amount_cents"),
                    currency=item.get("currency", "usd"),
                )
            
            if auto_charge:
                invoice = await self._async_stripe_call(
                    self.stripe.Invoice.finalize_invoice,
                    invoice.id
                )
            
            logger.info(f"Created Stripe invoice: {invoice.id} for customer {customer_id}")
            
            return {
                "id": invoice.id,
                "status": invoice.status,
                "amount_due": invoice.amount_due,
                "amount_paid": invoice.amount_paid,
            }
            
        except Exception as e:
            logger.error(f"Error creating Stripe invoice for customer {customer_id}: {e}")
            raise BillingProviderError(f"Failed to create invoice: {str(e)}")
    
    async def get_subscription(self, subscription_id: str) -> Dict[str, Any]:
        """Retrieve Stripe subscription."""
        try:
            subscription = await self._async_stripe_call(
                self.stripe.Subscription.retrieve,
                subscription_id
            )
            
            return {
                "id": subscription.id,
                "status": subscription.status,
                "customer": subscription.customer,
                "current_period_start": datetime.fromtimestamp(subscription.current_period_start),
                "current_period_end": datetime.fromtimestamp(subscription.current_period_end),
                "cancel_at_period_end": subscription.cancel_at_period_end,
            }
            
        except Exception as e:
            logger.error(f"Error retrieving Stripe subscription {subscription_id}: {e}")
            raise BillingProviderError(f"Failed to retrieve subscription: {str(e)}")
    
    async def get_customer(self, customer_id: str) -> Dict[str, Any]:
        """Retrieve Stripe customer."""
        try:
            customer = await self._async_stripe_call(
                self.stripe.Customer.retrieve,
                customer_id
            )
            
            return {
                "id": customer.id,
                "email": customer.email,
                "name": customer.name,
                "metadata": customer.metadata,
            }
            
        except Exception as e:
            logger.error(f"Error retrieving Stripe customer {customer_id}: {e}")
            raise BillingProviderError(f"Failed to retrieve customer: {str(e)}")
    
    async def _async_stripe_call(self, func, *args, **kwargs):
        """
        Wrapper to handle Stripe synchronous API in async context.
        
        In production, use stripe's async client if available.
        """
        import asyncio
        return await asyncio.to_thread(func, *args, **kwargs)


class MockBillingProvider(BillingProvider):
    """
    Mock billing provider for local development and testing.
    
    Simulates billing operations without external API calls.
    """
    
    def __init__(self):
        """Initialize mock billing provider."""
        self._customers: Dict[str, Dict[str, Any]] = {}
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._invoices: Dict[str, Dict[str, Any]] = {}
        self._payment_methods: Dict[str, PaymentMethod] = {}
        self._counter = 0
        logger.info("Mock billing provider initialized")
    
    def _generate_id(self, prefix: str) -> str:
        """Generate a mock ID."""
        self._counter += 1
        return f"{prefix}_{self._counter:08d}"
    
    async def create_customer(
        self,
        tenant_id: UUID,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a mock customer."""
        customer_id = self._generate_id("cus")
        
        self._customers[customer_id] = {
            "id": customer_id,
            "email": email,
            "name": name,
            "metadata": {**(metadata or {}), "tenant_id": str(tenant_id)},
            "created": datetime.utcnow(),
        }
        
        logger.info(f"Mock: Created customer {customer_id} for tenant {tenant_id}")
        return customer_id
    
    async def create_subscription(
        self,
        customer_id: str,
        price_id: str,
        trial_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a mock subscription."""
        from datetime import timedelta
        
        subscription_id = self._generate_id("sub")
        now = datetime.utcnow()
        
        if trial_days:
            trial_end = now + timedelta(days=trial_days)
            status = "trialing"
        else:
            trial_end = None
            status = "active"
        
        period_end = now + timedelta(days=30)
        
        subscription = {
            "id": subscription_id,
            "customer": customer_id,
            "price_id": price_id,
            "status": status,
            "current_period_start": now,
            "current_period_end": period_end,
            "trial_start": now if trial_days else None,
            "trial_end": trial_end,
            "cancel_at_period_end": False,
            "canceled_at": None,
            "metadata": metadata or {},
        }
        
        self._subscriptions[subscription_id] = subscription
        
        logger.info(f"Mock: Created subscription {subscription_id} for customer {customer_id}")
        return subscription
    
    async def update_subscription(
        self,
        subscription_id: str,
        new_price_id: str,
        prorate: bool = True,
    ) -> Dict[str, Any]:
        """Update a mock subscription."""
        if subscription_id not in self._subscriptions:
            raise BillingProviderError(f"Subscription {subscription_id} not found")
        
        subscription = self._subscriptions[subscription_id]
        subscription["price_id"] = new_price_id
        
        logger.info(f"Mock: Updated subscription {subscription_id} to price {new_price_id}")
        return subscription
    
    async def cancel_subscription(
        self,
        subscription_id: str,
        immediate: bool = False,
    ) -> Dict[str, Any]:
        """Cancel a mock subscription."""
        if subscription_id not in self._subscriptions:
            raise BillingProviderError(f"Subscription {subscription_id} not found")
        
        subscription = self._subscriptions[subscription_id]
        subscription["canceled_at"] = datetime.utcnow()
        
        if immediate:
            subscription["status"] = "canceled"
            subscription["current_period_end"] = datetime.utcnow()
        else:
            subscription["cancel_at_period_end"] = True
        
        logger.info(f"Mock: Canceled subscription {subscription_id} (immediate={immediate})")
        return subscription
    
    async def create_payment_method(
        self,
        customer_id: str,
        payment_method_data: Dict[str, Any],
    ) -> PaymentMethod:
        """Create a mock payment method."""
        pm_id = self._generate_id("pm")
        
        payment_method = PaymentMethod(
            id=pm_id,
            type=payment_method_data.get("type", "card"),
            last4=payment_method_data.get("last4", "4242"),
            brand=payment_method_data.get("brand", "visa"),
            exp_month=payment_method_data.get("exp_month", 12),
            exp_year=payment_method_data.get("exp_year", datetime.utcnow().year + 5),
            is_default=True,
        )
        
        self._payment_methods[pm_id] = payment_method
        
        logger.info(f"Mock: Created payment method {pm_id} for customer {customer_id}")
        return payment_method
    
    async def create_invoice(
        self,
        customer_id: str,
        line_items: List[Dict[str, Any]],
        auto_charge: bool = False,
    ) -> Dict[str, Any]:
        """Create a mock invoice."""
        invoice_id = self._generate_id("in")
        
        total_amount = sum(item.get("amount_cents", 0) for item in line_items)
        
        invoice = {
            "id": invoice_id,
            "customer": customer_id,
            "status": "paid" if auto_charge else "open",
            "amount_due": total_amount,
            "amount_paid": total_amount if auto_charge else 0,
            "line_items": line_items,
            "created": datetime.utcnow(),
        }
        
        self._invoices[invoice_id] = invoice
        
        logger.info(f"Mock: Created invoice {invoice_id} for customer {customer_id}")
        return invoice
    
    async def get_subscription(self, subscription_id: str) -> Dict[str, Any]:
        """Retrieve a mock subscription."""
        if subscription_id not in self._subscriptions:
            raise BillingProviderError(f"Subscription {subscription_id} not found")
        
        return self._subscriptions[subscription_id]
    
    async def get_customer(self, customer_id: str) -> Dict[str, Any]:
        """Retrieve a mock customer."""
        if customer_id not in self._customers:
            raise BillingProviderError(f"Customer {customer_id} not found")
        
        return self._customers[customer_id]
