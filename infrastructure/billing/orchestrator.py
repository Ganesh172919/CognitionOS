"""
Billing Orchestrator (Infrastructure/Application Service)

Bridges domain billing entities/repositories with external billing providers (Stripe/mock).

This module intentionally lives in the infrastructure layer to avoid leaking provider concerns
into the domain layer while still offering a cohesive API for:
- Checkout / billing portal sessions
- Webhook-driven subscription and invoice synchronization
- Customer linking and payment method updates
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple
from uuid import UUID, uuid4

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_config
from core.domain.billing.entities import (
    Invoice,
    InvoiceStatus,
    PaymentMethod,
    Subscription,
    SubscriptionStatus,
    SubscriptionTier,
)
from core.domain.tenant.entities import Tenant
from infrastructure.billing.provider import BillingProviderError, MockBillingProvider, StripeBillingProvider
from infrastructure.persistence.billing_models import SubscriptionModel
from infrastructure.persistence.billing_repository import (
    PostgreSQLInvoiceRepository,
    PostgreSQLSubscriptionRepository,
    PostgreSQLUsageRecordRepository,
)
from infrastructure.persistence.tenant_repository import PostgreSQLTenantRepository


logger = logging.getLogger(__name__)


def _utc_from_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc).replace(tzinfo=None)
    except Exception:
        return None


class BillingOrchestrator:
    """
    Provider-aware billing orchestrator.

    Used by API routes (checkout/portal) and webhook handlers (sync).
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.config = get_config()

        self.tenant_repo = PostgreSQLTenantRepository(session)
        self.subscription_repo = PostgreSQLSubscriptionRepository(session)
        self.invoice_repo = PostgreSQLInvoiceRepository(session)
        self.usage_repo = PostgreSQLUsageRecordRepository(session)

        provider = (self.config.billing.provider or "mock").strip().lower()
        if provider == "stripe":
            api_key = self.config.billing.stripe.api_key
            if not api_key:
                raise ValueError("STRIPE_API_KEY must be set when BILLING_PROVIDER=stripe")
            self.billing_provider = StripeBillingProvider(
                api_key=api_key,
                webhook_secret=self.config.billing.stripe.webhook_secret,
            )
        elif provider == "mock":
            self.billing_provider = MockBillingProvider()
        else:
            raise ValueError(f"Unknown billing provider: {provider}. Use 'stripe' or 'mock'.")

    # ---------------------------------------------------------------------
    # Checkout / Portal
    # ---------------------------------------------------------------------

    async def create_checkout_session(
        self,
        *,
        tenant_id: UUID,
        tier: SubscriptionTier,
        billing_cycle: str = "monthly",
        success_url: Optional[str] = None,
        cancel_url: Optional[str] = None,
        trial_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Create a Stripe Checkout session (or mock equivalent) for a subscription.

        Returns a dict that includes a URL the frontend can redirect to.
        """
        tenant = await self._require_tenant(tenant_id)

        price_id = self._resolve_price_id(tier=tier, billing_cycle=billing_cycle)
        if not price_id:
            raise ValueError(
                f"Missing Stripe price for tier={tier.value} cycle={billing_cycle}. "
                "Set STRIPE_PRICE_* env vars."
            )

        customer_id = await self._ensure_customer(tenant)
        success_url = success_url or self.config.billing.stripe.default_success_url
        cancel_url = cancel_url or self.config.billing.stripe.default_cancel_url

        metadata = {
            "tenant_id": str(tenant.id),
            "tenant_slug": tenant.slug,
            "tier": tier.value,
            "billing_cycle": billing_cycle,
        }

        if hasattr(self.billing_provider, "create_checkout_session"):
            session_data = await self.billing_provider.create_checkout_session(
                customer_id=customer_id,
                price_id=price_id,
                success_url=success_url,
                cancel_url=cancel_url,
                trial_days=trial_days,
                metadata=metadata,
            )
            return session_data

        # Mock fallback: immediately create/upgrade internal subscription and return a local URL.
        now = datetime.utcnow()
        if billing_cycle == "yearly":
            period_end = now + timedelta(days=365)
        else:
            period_end = now + timedelta(days=30)

        subscription = await self._upsert_internal_subscription(
            tenant=tenant,
            tier=tier,
            billing_cycle=billing_cycle,
            stripe_customer_id=customer_id,
            stripe_subscription_id=None,
            status=SubscriptionStatus.ACTIVE,
            current_period_start=now,
            current_period_end=period_end,
            trial_start=None,
            trial_end=None,
            cancel_at_period_end=False,
            canceled_at=None,
            amount_cents=Subscription._calculate_amount(tier, billing_cycle),
            currency="usd",
            metadata={"source": "mock_checkout"},
        )
        return {"id": f"mock_cs_{subscription.id}", "url": success_url, "provider": "mock"}

    async def create_billing_portal_session(
        self,
        *,
        tenant_id: UUID,
        return_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a customer billing portal session (Stripe) or mock equivalent."""
        tenant = await self._require_tenant(tenant_id)
        customer_id = await self._ensure_customer(tenant)
        return_url = return_url or self.config.billing.stripe.default_portal_return_url

        if hasattr(self.billing_provider, "create_billing_portal_session"):
            return await self.billing_provider.create_billing_portal_session(
                customer_id=customer_id,
                return_url=return_url,
            )

        return {"id": f"mock_portal_{tenant.id}", "url": return_url, "provider": "mock"}

    # ---------------------------------------------------------------------
    # Webhook sync hooks (called by StripeWebhookHandler)
    # ---------------------------------------------------------------------

    async def link_stripe_customer(self, *, tenant_id: UUID, stripe_customer_id: str) -> Tenant:
        """Persist Stripe customer id on tenant metadata and current subscription when present."""
        tenant = await self._require_tenant(tenant_id)
        tenant.metadata = {**(tenant.metadata or {}), "stripe_customer_id": stripe_customer_id}
        tenant.updated_at = datetime.utcnow()
        updated_tenant = await self.tenant_repo.update(tenant)

        subscription = await self.subscription_repo.get_by_tenant(tenant_id)
        if subscription:
            subscription.stripe_customer_id = stripe_customer_id
            subscription.updated_at = datetime.utcnow()
            await self.subscription_repo.update(subscription)

        return updated_tenant

    async def sync_customer_info(
        self,
        *,
        stripe_customer_id: str,
        email: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Best-effort sync of customer info.

        We update the tenant billing_email if we can locate the owning tenant via:
        - subscriptions.stripe_customer_id
        - tenant.metadata.stripe_customer_id
        """
        tenant = await self._find_tenant_by_customer_id(stripe_customer_id)
        if not tenant:
            return {"action": "customer_not_linked"}

        if email:
            tenant.billing_email = email
        tenant.metadata = {**(tenant.metadata or {}), "stripe_customer_name": name} if name else (tenant.metadata or {})
        tenant.updated_at = datetime.utcnow()
        await self.tenant_repo.update(tenant)
        return {"action": "customer_synced", "tenant_id": str(tenant.id)}

    async def add_payment_method(
        self,
        *,
        customer_id: str,
        payment_method_id: str,
        payment_method_type: str,
        last_four: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Attach payment method metadata to current subscription (best-effort)."""
        tenant = await self._find_tenant_by_customer_id(customer_id)
        if not tenant:
            return {"action": "payment_method_ignored", "reason": "unknown_customer"}

        subscription = await self.subscription_repo.get_by_tenant(tenant.id)
        if not subscription:
            return {"action": "payment_method_ignored", "reason": "no_subscription"}

        subscription.payment_method = PaymentMethod(
            id=payment_method_id,
            type=payment_method_type,
            last4=last_four or "",
            brand=None,
            exp_month=None,
            exp_year=None,
            is_default=True,
        )
        subscription.updated_at = datetime.utcnow()
        await self.subscription_repo.update(subscription)
        return {"action": "payment_method_updated"}

    async def mark_subscription_paid(
        self,
        *,
        subscription_id: str,
        payment_amount: float,
        payment_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mark a subscription as active after successful payment (best-effort)."""
        subscription = await self._find_subscription_by_any_id(subscription_id)
        if not subscription:
            return {"action": "subscription_not_found"}

        subscription.status = SubscriptionStatus.ACTIVE
        subscription.updated_at = datetime.utcnow()
        subscription.metadata = {
            **(subscription.metadata or {}),
            "last_payment": {
                "amount": payment_amount,
                "payment_method": payment_method,
                "at": datetime.utcnow().isoformat(),
            },
        }
        await self.subscription_repo.update(subscription)
        return {"action": "subscription_marked_paid"}

    async def mark_subscription_past_due(
        self,
        *,
        subscription_id: str,
        failure_reason: str,
    ) -> Dict[str, Any]:
        """Mark a subscription as past due (best-effort)."""
        subscription = await self._find_subscription_by_any_id(subscription_id)
        if not subscription:
            return {"action": "subscription_not_found"}

        subscription.status = SubscriptionStatus.PAST_DUE
        subscription.updated_at = datetime.utcnow()
        subscription.metadata = {
            **(subscription.metadata or {}),
            "last_payment_failure": {"reason": failure_reason, "at": datetime.utcnow().isoformat()},
        }
        await self.subscription_repo.update(subscription)
        return {"action": "subscription_marked_past_due"}

    async def mark_subscription_cancelled(
        self,
        *,
        stripe_subscription_id: str,
        cancelled_at: datetime,
    ) -> Dict[str, Any]:
        """Mark subscription cancelled by Stripe subscription id (best-effort)."""
        subscription = await self.subscription_repo.get_by_stripe_subscription_id(stripe_subscription_id)
        if not subscription:
            return {"action": "subscription_not_found"}

        subscription.status = SubscriptionStatus.CANCELED
        subscription.canceled_at = cancelled_at
        subscription.updated_at = datetime.utcnow()
        await self.subscription_repo.update(subscription)
        return {"action": "subscription_marked_cancelled"}

    async def mark_invoice_paid(
        self,
        *,
        stripe_invoice_id: str,
        paid_at: datetime,
        amount_paid: float,
    ) -> Dict[str, Any]:
        """Mark an invoice paid (best-effort)."""
        invoice = await self.invoice_repo.get_by_stripe_invoice_id(stripe_invoice_id)
        if not invoice:
            return {"action": "invoice_not_found"}

        invoice.status = InvoiceStatus.PAID
        invoice.paid_at = paid_at
        invoice.amount_paid_cents = int(amount_paid * 100)
        invoice.amount_due_cents = 0
        await self.invoice_repo.update(invoice)
        return {"action": "invoice_marked_paid"}

    async def mark_invoice_payment_failed(
        self,
        *,
        stripe_invoice_id: str,
        attempt_count: int,
    ) -> Dict[str, Any]:
        """Record invoice payment failure attempt (best-effort)."""
        invoice = await self.invoice_repo.get_by_stripe_invoice_id(stripe_invoice_id)
        if not invoice:
            return {"action": "invoice_not_found"}

        invoice.metadata = {
            **(invoice.metadata or {}),
            "payment_failed": {"attempt_count": int(attempt_count), "at": datetime.utcnow().isoformat()},
        }
        await self.invoice_repo.update(invoice)
        return {"action": "invoice_payment_failure_recorded"}

    async def record_payment(
        self,
        *,
        customer_id: str,
        amount: float,
        currency: str,
        payment_intent_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record a payment event in the billing audit log (immutable).

        The schema exists in migration 009 as `billing_audit_log`.
        """
        tenant = await self._find_tenant_by_customer_id(customer_id)
        if not tenant:
            return {"action": "payment_recorded_unlinked"}

        entity_id = uuid4()
        await self.session.execute(
            text(
                """
                INSERT INTO billing_audit_log (id, tenant_id, event_type, entity_type, entity_id, changes, timestamp, metadata)
                VALUES (:id, :tenant_id, :event_type, :entity_type, :entity_id, :changes, NOW(), :metadata)
                """
            ),
            {
                "id": entity_id,
                "tenant_id": tenant.id,
                "event_type": "payment_succeeded",
                "entity_type": "payment_intent",
                "entity_id": entity_id,
                "changes": None,
                "metadata": {
                    "payment_intent_id": payment_intent_id,
                    "customer_id": customer_id,
                    "amount": amount,
                    "currency": currency,
                    "metadata": metadata or {},
                },
            },
        )
        await self.session.flush()
        return {"action": "payment_audited", "tenant_id": str(tenant.id)}

    async def sync_subscription_from_stripe(
        self,
        *,
        stripe_subscription_id: str,
        subscription_data: Dict[str, Any],
    ) -> Subscription:
        """Upsert subscription state based on Stripe subscription object."""
        existing = await self.subscription_repo.get_by_stripe_subscription_id(stripe_subscription_id)

        tenant_id = self._extract_tenant_id_from_stripe(subscription_data) or (
            await self._find_tenant_id_by_customer_id(str(subscription_data.get("customer") or ""))
        )
        if not tenant_id:
            raise ValueError("Cannot resolve tenant_id from Stripe subscription metadata/customer id")

        tenant = await self._require_tenant(tenant_id)

        tier, billing_cycle = self._infer_tier_from_stripe(subscription_data)
        status = self._map_stripe_subscription_status(str(subscription_data.get("status") or ""))

        period_start = _utc_from_ts(subscription_data.get("current_period_start")) or datetime.utcnow()
        period_end = _utc_from_ts(subscription_data.get("current_period_end")) or datetime.utcnow()
        trial_start = _utc_from_ts(subscription_data.get("trial_start"))
        trial_end = _utc_from_ts(subscription_data.get("trial_end"))

        cancel_at_period_end = bool(subscription_data.get("cancel_at_period_end") or False)
        canceled_at = _utc_from_ts(subscription_data.get("canceled_at"))

        amount_cents, currency = self._infer_amount_currency_from_stripe(subscription_data, tier, billing_cycle)

        upserted = await self._upsert_internal_subscription(
            tenant=tenant,
            tier=tier,
            billing_cycle=billing_cycle,
            stripe_customer_id=str(subscription_data.get("customer") or "") or None,
            stripe_subscription_id=stripe_subscription_id,
            status=status,
            current_period_start=period_start,
            current_period_end=period_end,
            trial_start=trial_start,
            trial_end=trial_end,
            cancel_at_period_end=cancel_at_period_end,
            canceled_at=canceled_at,
            amount_cents=amount_cents,
            currency=currency,
            metadata={"stripe": {"raw": {"id": stripe_subscription_id}}},
            existing_id=(existing.id if existing else None),
        )

        # Keep tenant tier in sync for fast access and UI.
        if tenant.subscription_tier != upserted.tier.value:
            tenant.upgrade_tier(upserted.tier.value)
            await self.tenant_repo.update(tenant)

        return upserted

    async def sync_invoice_from_stripe(
        self,
        *,
        stripe_invoice_id: str,
        invoice_data: Dict[str, Any],
    ) -> Invoice:
        """Upsert invoice state based on Stripe invoice object."""
        existing = await self.invoice_repo.get_by_stripe_invoice_id(stripe_invoice_id)

        tenant_id = self._extract_tenant_id_from_stripe(invoice_data) or (
            await self._find_tenant_id_by_customer_id(str(invoice_data.get("customer") or ""))
        )
        if not tenant_id:
            raise ValueError("Cannot resolve tenant_id from Stripe invoice metadata/customer id")

        subscription_id = None
        if invoice_data.get("subscription"):
            subscription = await self.subscription_repo.get_by_stripe_subscription_id(str(invoice_data["subscription"]))
            subscription_id = subscription.id if subscription else None

        if not subscription_id:
            # Fallback to current subscription.
            current = await self.subscription_repo.get_by_tenant(tenant_id)
            subscription_id = current.id if current else None
        if not subscription_id:
            raise ValueError("Cannot resolve internal subscription_id for Stripe invoice")

        status = self._map_stripe_invoice_status(str(invoice_data.get("status") or "open"))

        amount_cents = int(invoice_data.get("amount_due") or invoice_data.get("amount_paid") or 0)
        amount_paid_cents = int(invoice_data.get("amount_paid") or 0)
        amount_due_cents = int(invoice_data.get("amount_remaining") or (amount_cents - amount_paid_cents))
        currency = str(invoice_data.get("currency") or "usd")

        period_start = _utc_from_ts(invoice_data.get("period_start")) or datetime.utcnow()
        period_end = _utc_from_ts(invoice_data.get("period_end")) or datetime.utcnow()

        due_date = _utc_from_ts(invoice_data.get("due_date"))
        paid_at = _utc_from_ts(invoice_data.get("status_transitions", {}).get("paid_at")) or _utc_from_ts(
            invoice_data.get("paid_at")
        )

        invoice_number = str(invoice_data.get("number") or invoice_data.get("id") or "").strip()
        if not invoice_number:
            invoice_number = f"STRIPE-{stripe_invoice_id}"

        line_items = []
        try:
            lines = (invoice_data.get("lines") or {}).get("data") or []
            for line in lines:
                line_items.append(
                    {
                        "description": line.get("description") or line.get("price", {}).get("nickname") or "line_item",
                        "quantity": int(line.get("quantity") or 1),
                        "amount_cents": int(line.get("amount") or 0),
                        "currency": currency,
                        "period": line.get("period"),
                    }
                )
        except Exception:
            line_items = []

        entity = Invoice(
            id=existing.id if existing else uuid4(),
            tenant_id=tenant_id,
            subscription_id=subscription_id,
            status=status,
            amount_cents=amount_cents,
            amount_paid_cents=amount_paid_cents,
            amount_due_cents=amount_due_cents,
            currency=currency,
            stripe_invoice_id=stripe_invoice_id,
            invoice_number=invoice_number,
            period_start=period_start,
            period_end=period_end,
            due_date=due_date,
            paid_at=paid_at,
            created_at=existing.created_at if existing else datetime.utcnow(),
            line_items=line_items,
            metadata={"stripe": {"raw": {"id": stripe_invoice_id}}},
        )

        if existing:
            return await self.invoice_repo.update(entity)
        return await self.invoice_repo.create(entity)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    async def _require_tenant(self, tenant_id: UUID) -> Tenant:
        tenant = await self.tenant_repo.get_by_id(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant not found: {tenant_id}")
        return tenant

    def _resolve_price_id(self, *, tier: SubscriptionTier, billing_cycle: str) -> Optional[str]:
        cycle = str(billing_cycle or "monthly").strip().lower()
        key = f"{tier.value}_{cycle}"
        return self.config.billing.stripe_price_map().get(key)

    async def _ensure_customer(self, tenant: Tenant) -> str:
        # Prefer tenant metadata mapping.
        existing = (tenant.metadata or {}).get("stripe_customer_id")
        if existing:
            return str(existing)

        # Fall back to current subscription mapping.
        subscription = await self.subscription_repo.get_by_tenant(tenant.id)
        if subscription and subscription.stripe_customer_id:
            tenant.metadata = {**(tenant.metadata or {}), "stripe_customer_id": subscription.stripe_customer_id}
            tenant.updated_at = datetime.utcnow()
            await self.tenant_repo.update(tenant)
            return str(subscription.stripe_customer_id)

        # Create customer in provider.
        try:
            customer_id = await self.billing_provider.create_customer(
                tenant_id=tenant.id,
                email=tenant.billing_email or "",
                name=tenant.name,
                metadata={"tenant_id": str(tenant.id), "tenant_slug": tenant.slug},
            )
        except BillingProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BillingProviderError(f"Failed to create customer: {exc}") from exc

        tenant.metadata = {**(tenant.metadata or {}), "stripe_customer_id": customer_id}
        tenant.updated_at = datetime.utcnow()
        await self.tenant_repo.update(tenant)
        return str(customer_id)

    async def _find_tenant_id_by_customer_id(self, customer_id: str) -> Optional[UUID]:
        tenant = await self._find_tenant_by_customer_id(customer_id)
        return tenant.id if tenant else None

    async def _find_tenant_by_customer_id(self, customer_id: str) -> Optional[Tenant]:
        if not customer_id:
            return None

        # 1) via subscriptions table (fast).
        stmt = select(SubscriptionModel.tenant_id).where(SubscriptionModel.stripe_customer_id == customer_id).limit(1)
        result = await self.session.execute(stmt)
        tenant_id = result.scalar_one_or_none()
        if tenant_id:
            return await self.tenant_repo.get_by_id(tenant_id)

        # 2) via tenant metadata jsonb.
        try:
            from infrastructure.persistence.tenant_models import TenantModel

            stmt2 = select(TenantModel.id).where(TenantModel.tenant_metadata["stripe_customer_id"].astext == customer_id)
            result2 = await self.session.execute(stmt2)
            tid = result2.scalar_one_or_none()
            if tid:
                return await self.tenant_repo.get_by_id(tid)
        except Exception:
            return None

        return None

    async def _find_subscription_by_any_id(self, value: str) -> Optional[Subscription]:
        """Resolve either a Stripe subscription id or an internal UUID to a Subscription entity."""
        if not value:
            return None
        subscription = await self.subscription_repo.get_by_stripe_subscription_id(value)
        if subscription:
            return subscription
        try:
            return await self.subscription_repo.get_by_id(UUID(str(value)))
        except Exception:
            return None

    def _extract_tenant_id_from_stripe(self, obj: Dict[str, Any]) -> Optional[UUID]:
        raw = (obj.get("metadata") or {}).get("tenant_id")
        if not raw:
            return None
        try:
            return UUID(str(raw))
        except Exception:
            return None

    def _infer_tier_from_stripe(self, subscription_data: Dict[str, Any]) -> Tuple[SubscriptionTier, str]:
        # Prefer explicit metadata.
        tier_meta = (subscription_data.get("metadata") or {}).get("tier")
        billing_cycle = str((subscription_data.get("metadata") or {}).get("billing_cycle") or "monthly").lower()
        if tier_meta:
            try:
                return SubscriptionTier(str(tier_meta)), billing_cycle
            except Exception:
                pass

        # Infer from price id.
        price_id = None
        try:
            items = (subscription_data.get("items") or {}).get("data") or []
            if items:
                price_id = (items[0].get("price") or {}).get("id") or items[0].get("price_id")
        except Exception:
            price_id = None

        for key, value in self.config.billing.stripe_price_map().items():
            if value and price_id and value == price_id:
                tier_key, cycle_key = key.split("_", 1)
                return SubscriptionTier(tier_key), cycle_key

        return SubscriptionTier.FREE, "monthly"

    def _map_stripe_subscription_status(self, status: str) -> SubscriptionStatus:
        status = (status or "").strip().lower()
        mapping = {
            "active": SubscriptionStatus.ACTIVE,
            "trialing": SubscriptionStatus.TRIALING,
            "past_due": SubscriptionStatus.PAST_DUE,
            "canceled": SubscriptionStatus.CANCELED,
            "unpaid": SubscriptionStatus.UNPAID,
            "paused": SubscriptionStatus.PAUSED,
            "incomplete": SubscriptionStatus.PAST_DUE,
            "incomplete_expired": SubscriptionStatus.UNPAID,
        }
        return mapping.get(status, SubscriptionStatus.PAUSED)

    def _map_stripe_invoice_status(self, status: str) -> InvoiceStatus:
        status = (status or "").strip().lower()
        mapping = {
            "draft": InvoiceStatus.DRAFT,
            "open": InvoiceStatus.OPEN,
            "paid": InvoiceStatus.PAID,
            "void": InvoiceStatus.VOID,
            "uncollectible": InvoiceStatus.UNCOLLECTIBLE,
        }
        return mapping.get(status, InvoiceStatus.OPEN)

    def _infer_amount_currency_from_stripe(
        self,
        subscription_data: Dict[str, Any],
        tier: SubscriptionTier,
        billing_cycle: str,
    ) -> Tuple[int, str]:
        # Try price unit amount.
        try:
            items = (subscription_data.get("items") or {}).get("data") or []
            if items:
                price = items[0].get("price") or {}
                unit_amount = price.get("unit_amount")
                currency = str(price.get("currency") or "usd")
                if unit_amount is not None:
                    return int(unit_amount), currency
        except Exception:
            pass
        return int(Subscription._calculate_amount(tier, billing_cycle)), "usd"

    async def _upsert_internal_subscription(
        self,
        *,
        tenant: Tenant,
        tier: SubscriptionTier,
        billing_cycle: str,
        stripe_customer_id: Optional[str],
        stripe_subscription_id: Optional[str],
        status: SubscriptionStatus,
        current_period_start: datetime,
        current_period_end: datetime,
        trial_start: Optional[datetime],
        trial_end: Optional[datetime],
        cancel_at_period_end: bool,
        canceled_at: Optional[datetime],
        amount_cents: int,
        currency: str,
        metadata: Optional[Dict[str, Any]] = None,
        existing_id: Optional[UUID] = None,
    ) -> Subscription:
        now = datetime.utcnow()
        entity = Subscription(
            id=existing_id or uuid4(),
            tenant_id=tenant.id,
            tier=tier,
            status=status,
            stripe_subscription_id=stripe_subscription_id,
            stripe_customer_id=stripe_customer_id,
            current_period_start=current_period_start,
            current_period_end=current_period_end,
            trial_start=trial_start,
            trial_end=trial_end,
            canceled_at=canceled_at,
            cancel_at_period_end=cancel_at_period_end,
            amount_cents=amount_cents,
            currency=currency,
            billing_cycle=billing_cycle,
            payment_method=None,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

        if existing_id:
            # Preserve created_at from existing row.
            existing = await self.subscription_repo.get_by_id(existing_id)
            if existing:
                entity.created_at = existing.created_at

        if existing_id:
            return await self.subscription_repo.update(entity)
        return await self.subscription_repo.create(entity)
