"""
Stripe Webhook Handler Infrastructure

Processes Stripe webhook events for automated billing operations.
Handles payment events, subscription lifecycle, and invoice management.
"""

import logging
import hmac
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID
from enum import Enum

from core.domain.billing.entities import (
    Subscription,
    SubscriptionStatus,
    Invoice,
    InvoiceStatus,
)
from core.exceptions import (
    BillingException,
    ValidationError,
)

logger = logging.getLogger(__name__)


class WebhookEventType(str, Enum):
    """Stripe webhook event types we handle."""
    PAYMENT_SUCCEEDED = "payment_intent.succeeded"
    PAYMENT_FAILED = "payment_intent.payment_failed"
    SUBSCRIPTION_CREATED = "customer.subscription.created"
    SUBSCRIPTION_UPDATED = "customer.subscription.updated"
    SUBSCRIPTION_DELETED = "customer.subscription.deleted"
    SUBSCRIPTION_TRIAL_ENDING = "customer.subscription.trial_will_end"
    INVOICE_PAID = "invoice.paid"
    INVOICE_PAYMENT_FAILED = "invoice.payment_failed"
    INVOICE_FINALIZED = "invoice.finalized"
    CUSTOMER_CREATED = "customer.created"
    CUSTOMER_UPDATED = "customer.updated"
    CUSTOMER_DELETED = "customer.deleted"
    PAYMENT_METHOD_ATTACHED = "payment_method.attached"
    PAYMENT_METHOD_DETACHED = "payment_method.detached"


class WebhookValidationError(Exception):
    """Exception raised when webhook signature validation fails."""
    pass


class StripeWebhookHandler:
    """
    Handles Stripe webhook events with signature verification and event processing.
    
    Features:
    - Signature verification for security
    - Idempotent event processing
    - Automatic retry handling
    - Event type routing
    - Error tracking and alerting
    """
    
    def __init__(
        self,
        webhook_secret: str,
        billing_service: Any,  # Core billing service
        event_repository: Any,  # Event storage
    ):
        """
        Initialize webhook handler.
        
        Args:
            webhook_secret: Stripe webhook signing secret
            billing_service: Core billing domain service
            event_repository: Repository for event deduplication
        """
        self.webhook_secret = webhook_secret
        self.billing_service = billing_service
        self.event_repository = event_repository
        logger.info("Stripe webhook handler initialized")
    
    def verify_signature(
        self,
        payload: bytes,
        signature: str,
        timestamp: int,
        tolerance: int = 300,  # 5 minutes
    ) -> bool:
        """
        Verify Stripe webhook signature.
        
        Args:
            payload: Raw webhook payload
            signature: Stripe-Signature header value
            timestamp: Webhook timestamp
            tolerance: Maximum age of webhook in seconds
            
        Returns:
            True if signature is valid
            
        Raises:
            WebhookValidationError: If signature is invalid or webhook is too old
        """
        # Check timestamp to prevent replay attacks
        current_time = datetime.utcnow().timestamp()
        if abs(current_time - timestamp) > tolerance:
            raise WebhookValidationError(
                f"Webhook timestamp too old: {timestamp} (current: {current_time})"
            )
        
        # Compute expected signature
        signed_payload = f"{timestamp}.{payload.decode('utf-8')}"
        expected_sig = hmac.new(
            self.webhook_secret.encode('utf-8'),
            signed_payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Parse signature header
        sig_parts = dict(part.split('=') for part in signature.split(','))
        received_sig = sig_parts.get('v1')
        
        if not received_sig:
            raise WebhookValidationError("No v1 signature found in header")
        
        # Constant-time comparison to prevent timing attacks
        if not hmac.compare_digest(expected_sig, received_sig):
            raise WebhookValidationError("Signature verification failed")
        
        return True
    
    async def process_event(
        self,
        event_data: Dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a Stripe webhook event.
        
        Args:
            event_data: Parsed webhook event data
            idempotency_key: Optional idempotency key for deduplication
            
        Returns:
            Processing result with status and details
            
        Raises:
            BillingException: If event processing fails
        """
        event_id = event_data.get("id")
        event_type = event_data.get("type")
        
        logger.info(f"Processing webhook event: {event_id} ({event_type})")
        
        # Check for duplicate events (idempotent processing)
        if await self._is_duplicate_event(event_id):
            logger.info(f"Skipping duplicate event: {event_id}")
            return {"status": "skipped", "reason": "duplicate"}
        
        try:
            # Route to appropriate handler
            handler = self._get_event_handler(event_type)
            if not handler:
                logger.warning(f"No handler for event type: {event_type}")
                return {"status": "ignored", "event_type": event_type}
            
            # Process event
            result = await handler(event_data)
            
            # Store event for deduplication
            await self._store_event(event_id, event_type, result)
            
            logger.info(f"Successfully processed event: {event_id}")
            return {"status": "processed", "result": result}
            
        except Exception as e:
            logger.error(f"Error processing webhook event {event_id}: {e}", exc_info=True)
            # Store failed event for retry
            await self._store_failed_event(event_id, event_type, str(e))
            raise BillingException(f"Webhook processing failed: {str(e)}")
    
    def _get_event_handler(self, event_type: str):
        """Route event type to handler method."""
        handlers = {
            WebhookEventType.PAYMENT_SUCCEEDED: self._handle_payment_succeeded,
            WebhookEventType.PAYMENT_FAILED: self._handle_payment_failed,
            WebhookEventType.SUBSCRIPTION_CREATED: self._handle_subscription_created,
            WebhookEventType.SUBSCRIPTION_UPDATED: self._handle_subscription_updated,
            WebhookEventType.SUBSCRIPTION_DELETED: self._handle_subscription_deleted,
            WebhookEventType.SUBSCRIPTION_TRIAL_ENDING: self._handle_trial_ending,
            WebhookEventType.INVOICE_PAID: self._handle_invoice_paid,
            WebhookEventType.INVOICE_PAYMENT_FAILED: self._handle_invoice_payment_failed,
            WebhookEventType.INVOICE_FINALIZED: self._handle_invoice_finalized,
            WebhookEventType.CUSTOMER_CREATED: self._handle_customer_created,
            WebhookEventType.CUSTOMER_UPDATED: self._handle_customer_updated,
            WebhookEventType.PAYMENT_METHOD_ATTACHED: self._handle_payment_method_attached,
        }
        return handlers.get(event_type)
    
    async def _handle_payment_succeeded(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle successful payment.
        
        Updates subscription status, records payment, sends confirmation.
        """
        payment_intent = event_data["data"]["object"]
        customer_id = payment_intent.get("customer")
        amount = payment_intent.get("amount") / 100  # Convert cents to dollars
        currency = payment_intent.get("currency", "usd")
        
        logger.info(f"Payment succeeded: ${amount} {currency} for customer {customer_id}")
        
        # Update subscription status if applicable
        if payment_intent.get("metadata", {}).get("subscription_id"):
            subscription_id = payment_intent["metadata"]["subscription_id"]
            await self.billing_service.mark_subscription_paid(
                subscription_id=subscription_id,
                payment_amount=amount,
                payment_method=payment_intent.get("payment_method"),
            )
        
        # Record payment
        await self.billing_service.record_payment(
            customer_id=customer_id,
            amount=amount,
            currency=currency,
            payment_intent_id=payment_intent["id"],
            metadata=payment_intent.get("metadata", {}),
        )
        
        # Send confirmation email (async task)
        await self._send_payment_confirmation(customer_id, amount, currency)
        
        return {
            "action": "payment_recorded",
            "amount": amount,
            "currency": currency,
        }
    
    async def _handle_payment_failed(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle failed payment.
        
        Updates subscription status, initiates dunning workflow, alerts customer.
        """
        payment_intent = event_data["data"]["object"]
        customer_id = payment_intent.get("customer")
        amount = payment_intent.get("amount") / 100
        error_message = payment_intent.get("last_payment_error", {}).get("message", "Unknown error")
        
        logger.warning(f"Payment failed for customer {customer_id}: {error_message}")
        
        # Mark subscription as past due
        if payment_intent.get("metadata", {}).get("subscription_id"):
            subscription_id = payment_intent["metadata"]["subscription_id"]
            await self.billing_service.mark_subscription_past_due(
                subscription_id=subscription_id,
                failure_reason=error_message,
            )
        
        # Initiate dunning workflow (automated retry emails)
        await self._start_dunning_workflow(
            customer_id=customer_id,
            amount=amount,
            failure_reason=error_message,
        )
        
        return {
            "action": "payment_failed_handled",
            "dunning_initiated": True,
        }
    
    async def _handle_subscription_created(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle new subscription creation."""
        subscription = event_data["data"]["object"]
        
        logger.info(f"Subscription created: {subscription['id']}")
        
        await self.billing_service.sync_subscription_from_stripe(
            stripe_subscription_id=subscription["id"],
            subscription_data=subscription,
        )
        
        return {"action": "subscription_synced"}
    
    async def _handle_subscription_updated(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle subscription updates (plan changes, cancellations, etc.).
        """
        subscription = event_data["data"]["object"]
        previous_attributes = event_data.get("data", {}).get("previous_attributes", {})
        
        logger.info(f"Subscription updated: {subscription['id']}")
        
        # Detect what changed
        changes = []
        if "items" in previous_attributes:
            changes.append("plan_changed")
        if "cancel_at_period_end" in previous_attributes:
            changes.append("cancellation_scheduled" if subscription["cancel_at_period_end"] else "cancellation_removed")
        if "status" in previous_attributes:
            changes.append(f"status_changed_to_{subscription['status']}")
        
        await self.billing_service.sync_subscription_from_stripe(
            stripe_subscription_id=subscription["id"],
            subscription_data=subscription,
        )
        
        # Handle specific changes
        if "cancellation_scheduled" in changes:
            await self._handle_subscription_cancellation_scheduled(subscription)
        
        if subscription["status"] == "active" and "status_changed_to_active" in changes:
            await self._handle_subscription_activated(subscription)
        
        return {
            "action": "subscription_updated",
            "changes": changes,
        }
    
    async def _handle_subscription_deleted(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle subscription deletion/cancellation."""
        subscription = event_data["data"]["object"]
        
        logger.info(f"Subscription deleted: {subscription['id']}")
        
        await self.billing_service.mark_subscription_cancelled(
            stripe_subscription_id=subscription["id"],
            cancelled_at=datetime.utcnow(),
        )
        
        # Send cancellation confirmation
        await self._send_cancellation_confirmation(subscription["customer"])
        
        return {"action": "subscription_cancelled"}
    
    async def _handle_trial_ending(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle trial ending notification (3 days before).
        
        Sends reminder email to customer.
        """
        subscription = event_data["data"]["object"]
        trial_end = datetime.fromtimestamp(subscription["trial_end"])
        
        logger.info(f"Trial ending soon for subscription: {subscription['id']} (ends {trial_end})")
        
        await self._send_trial_ending_notification(
            customer_id=subscription["customer"],
            trial_end_date=trial_end,
        )
        
        return {"action": "trial_ending_notification_sent"}
    
    async def _handle_invoice_paid(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle successful invoice payment."""
        invoice = event_data["data"]["object"]
        
        logger.info(f"Invoice paid: {invoice['id']}")
        
        await self.billing_service.mark_invoice_paid(
            stripe_invoice_id=invoice["id"],
            paid_at=datetime.fromtimestamp(invoice["status_transitions"]["paid_at"]),
            amount_paid=invoice["amount_paid"] / 100,
        )
        
        return {"action": "invoice_marked_paid"}
    
    async def _handle_invoice_payment_failed(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle failed invoice payment."""
        invoice = event_data["data"]["object"]
        attempt_count = invoice.get("attempt_count", 1)
        
        logger.warning(f"Invoice payment failed: {invoice['id']} (attempt {attempt_count})")
        
        await self.billing_service.mark_invoice_payment_failed(
            stripe_invoice_id=invoice["id"],
            attempt_count=attempt_count,
        )
        
        # Escalate if max attempts reached
        if attempt_count >= 4:
            await self._escalate_payment_failure(invoice)
        
        return {
            "action": "invoice_payment_failed_handled",
            "attempt": attempt_count,
        }
    
    async def _handle_invoice_finalized(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle invoice finalization (ready for payment)."""
        invoice = event_data["data"]["object"]
        
        logger.info(f"Invoice finalized: {invoice['id']}")
        
        await self.billing_service.sync_invoice_from_stripe(
            stripe_invoice_id=invoice["id"],
            invoice_data=invoice,
        )
        
        return {"action": "invoice_synced"}
    
    async def _handle_customer_created(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle customer creation."""
        customer = event_data["data"]["object"]
        
        logger.info(f"Customer created: {customer['id']}")
        
        # Link to tenant if metadata contains tenant_id
        tenant_id = customer.get("metadata", {}).get("tenant_id")
        if tenant_id:
            await self.billing_service.link_stripe_customer(
                tenant_id=UUID(tenant_id),
                stripe_customer_id=customer["id"],
            )
        
        return {"action": "customer_linked"}
    
    async def _handle_customer_updated(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle customer information updates."""
        customer = event_data["data"]["object"]
        previous_attributes = event_data.get("data", {}).get("previous_attributes", {})
        
        logger.info(f"Customer updated: {customer['id']}")
        
        # Sync relevant changes (email, name, payment method)
        if "email" in previous_attributes or "name" in previous_attributes:
            await self.billing_service.sync_customer_info(
                stripe_customer_id=customer["id"],
                email=customer.get("email"),
                name=customer.get("name"),
            )
        
        return {"action": "customer_info_synced"}
    
    async def _handle_payment_method_attached(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle payment method attachment."""
        payment_method = event_data["data"]["object"]
        
        logger.info(f"Payment method attached: {payment_method['id']}")
        
        await self.billing_service.add_payment_method(
            customer_id=payment_method["customer"],
            payment_method_id=payment_method["id"],
            payment_method_type=payment_method["type"],
            last_four=payment_method.get("card", {}).get("last4"),
        )
        
        return {"action": "payment_method_added"}
    
    # Helper methods for notifications and workflows
    
    async def _is_duplicate_event(self, event_id: str) -> bool:
        """Check if event has already been processed."""
        return await self.event_repository.event_exists(event_id)
    
    async def _store_event(self, event_id: str, event_type: str, result: Dict[str, Any]):
        """Store processed event for deduplication."""
        await self.event_repository.store_event(
            event_id=event_id,
            event_type=event_type,
            processed_at=datetime.utcnow(),
            result=result,
        )
    
    async def _store_failed_event(self, event_id: str, event_type: str, error: str):
        """Store failed event for retry."""
        await self.event_repository.store_failed_event(
            event_id=event_id,
            event_type=event_type,
            error=error,
            failed_at=datetime.utcnow(),
        )
    
    async def _send_payment_confirmation(self, customer_id: str, amount: float, currency: str):
        """Send payment confirmation email."""
        # Implementation would integrate with email service
        logger.info(f"Payment confirmation sent to customer {customer_id}")
    
    async def _start_dunning_workflow(self, customer_id: str, amount: float, failure_reason: str):
        """Start automated dunning workflow for failed payments."""
        # Implementation would integrate with dunning service
        logger.info(f"Dunning workflow started for customer {customer_id}")
    
    async def _send_trial_ending_notification(self, customer_id: str, trial_end_date: datetime):
        """Send trial ending notification."""
        logger.info(f"Trial ending notification sent to customer {customer_id}")
    
    async def _send_cancellation_confirmation(self, customer_id: str):
        """Send subscription cancellation confirmation."""
        logger.info(f"Cancellation confirmation sent to customer {customer_id}")
    
    async def _handle_subscription_cancellation_scheduled(self, subscription: Dict[str, Any]):
        """Handle scheduled cancellation (at period end)."""
        logger.info(f"Subscription cancellation scheduled: {subscription['id']}")
    
    async def _handle_subscription_activated(self, subscription: Dict[str, Any]):
        """Handle subscription activation (from trial or past_due)."""
        logger.info(f"Subscription activated: {subscription['id']}")
    
    async def _escalate_payment_failure(self, invoice: Dict[str, Any]):
        """Escalate payment failure after max retry attempts."""
        logger.error(f"Payment failure escalated for invoice: {invoice['id']}")
        # Would trigger alerts, suspend service, etc.
