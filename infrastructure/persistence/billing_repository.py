"""
Billing Repository Implementations

PostgreSQL implementations of billing repositories.
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Optional, List
from uuid import UUID

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from core.domain.billing.entities import (
    Subscription,
    SubscriptionStatus,
    SubscriptionTier,
    Invoice,
    InvoiceStatus,
    UsageRecord,
    PaymentMethod,
)
from core.domain.billing.repositories import (
    SubscriptionRepository,
    InvoiceRepository,
    UsageRecordRepository,
)
from infrastructure.persistence.billing_models import (
    SubscriptionModel,
    InvoiceModel,
    UsageRecordModel,
)


logger = logging.getLogger(__name__)


class PostgreSQLSubscriptionRepository(SubscriptionRepository):
    """PostgreSQL implementation of SubscriptionRepository"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, subscription: Subscription) -> Subscription:
        """Create a new subscription"""
        try:
            subscription_model = self._to_model(subscription)
            self.session.add(subscription_model)
            await self.session.flush()
            await self.session.refresh(subscription_model)
            
            logger.info(f"Created subscription: {subscription.id}")
            return self._to_entity(subscription_model)
        except Exception as e:
            logger.error(f"Error creating subscription {subscription.id}: {e}")
            raise
    
    async def get_by_id(self, subscription_id: UUID) -> Optional[Subscription]:
        """Retrieve subscription by ID"""
        try:
            stmt = select(SubscriptionModel).where(SubscriptionModel.id == subscription_id)
            result = await self.session.execute(stmt)
            subscription_model = result.scalar_one_or_none()
            
            if subscription_model is None:
                return None
            
            return self._to_entity(subscription_model)
        except Exception as e:
            logger.error(f"Error fetching subscription {subscription_id}: {e}")
            raise
    
    async def get_by_tenant(self, tenant_id: UUID) -> Optional[Subscription]:
        """Get active subscription for a tenant"""
        try:
            stmt = (
                select(SubscriptionModel)
                .where(
                    and_(
                        SubscriptionModel.tenant_id == tenant_id,
                        SubscriptionModel.status == SubscriptionStatus.ACTIVE
                    )
                )
            )
            result = await self.session.execute(stmt)
            subscription_model = result.scalar_one_or_none()
            
            if subscription_model is None:
                return None
            
            return self._to_entity(subscription_model)
        except Exception as e:
            logger.error(f"Error fetching subscription for tenant {tenant_id}: {e}")
            raise
    
    async def get_active_subscriptions(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Subscription]:
        """Get all active subscriptions"""
        try:
            stmt = (
                select(SubscriptionModel)
                .where(SubscriptionModel.status == SubscriptionStatus.ACTIVE)
                .offset(offset)
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            subscription_models = result.scalars().all()
            
            return [self._to_entity(model) for model in subscription_models]
        except Exception as e:
            logger.error(f"Error fetching active subscriptions: {e}")
            raise
    
    async def get_by_status(
        self,
        status: SubscriptionStatus,
        limit: int = 100
    ) -> List[Subscription]:
        """Get subscriptions by status"""
        try:
            stmt = (
                select(SubscriptionModel)
                .where(SubscriptionModel.status == status)
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            subscription_models = result.scalars().all()
            
            return [self._to_entity(model) for model in subscription_models]
        except Exception as e:
            logger.error(f"Error fetching subscriptions by status {status}: {e}")
            raise
    
    async def get_expiring_trials(self, before: datetime) -> List[Subscription]:
        """Get trial subscriptions expiring before specified date"""
        try:
            stmt = (
                select(SubscriptionModel)
                .where(
                    and_(
                        SubscriptionModel.status == SubscriptionStatus.TRIALING,
                        SubscriptionModel.trial_end.isnot(None),
                        SubscriptionModel.trial_end <= before
                    )
                )
            )
            result = await self.session.execute(stmt)
            subscription_models = result.scalars().all()
            
            return [self._to_entity(model) for model in subscription_models]
        except Exception as e:
            logger.error(f"Error fetching expiring trials: {e}")
            raise
    
    async def get_by_stripe_subscription_id(
        self,
        stripe_subscription_id: str
    ) -> Optional[Subscription]:
        """Get subscription by Stripe subscription ID"""
        try:
            stmt = (
                select(SubscriptionModel)
                .where(SubscriptionModel.stripe_subscription_id == stripe_subscription_id)
            )
            result = await self.session.execute(stmt)
            subscription_model = result.scalar_one_or_none()
            
            if subscription_model is None:
                return None
            
            return self._to_entity(subscription_model)
        except Exception as e:
            logger.error(f"Error fetching subscription by Stripe ID {stripe_subscription_id}: {e}")
            raise
    
    async def update(self, subscription: Subscription) -> Subscription:
        """Update an existing subscription"""
        try:
            stmt = select(SubscriptionModel).where(SubscriptionModel.id == subscription.id)
            result = await self.session.execute(stmt)
            subscription_model = result.scalar_one_or_none()
            
            if subscription_model is None:
                raise ValueError(f"Subscription not found: {subscription.id}")
            
            # Update fields
            subscription_model.tenant_id = subscription.tenant_id
            subscription_model.tier = subscription.tier.value
            subscription_model.status = subscription.status
            subscription_model.stripe_subscription_id = subscription.stripe_subscription_id
            subscription_model.stripe_customer_id = subscription.stripe_customer_id
            subscription_model.current_period_start = subscription.current_period_start
            subscription_model.current_period_end = subscription.current_period_end
            subscription_model.trial_start = subscription.trial_start
            subscription_model.trial_end = subscription.trial_end
            subscription_model.canceled_at = subscription.canceled_at
            subscription_model.cancel_at_period_end = subscription.cancel_at_period_end
            subscription_model.amount_cents = subscription.amount_cents
            subscription_model.currency = subscription.currency
            subscription_model.billing_cycle = subscription.billing_cycle
            subscription_model.payment_method = self._payment_method_to_dict(subscription.payment_method) if subscription.payment_method else None
            subscription_model.updated_at = subscription.updated_at
            subscription_model.metadata = subscription.metadata
            
            await self.session.flush()
            await self.session.refresh(subscription_model)
            
            logger.info(f"Updated subscription: {subscription.id}")
            return self._to_entity(subscription_model)
        except Exception as e:
            logger.error(f"Error updating subscription {subscription.id}: {e}")
            raise
    
    async def delete(self, subscription_id: UUID) -> bool:
        """Delete a subscription"""
        try:
            stmt = select(SubscriptionModel).where(SubscriptionModel.id == subscription_id)
            result = await self.session.execute(stmt)
            subscription_model = result.scalar_one_or_none()
            
            if subscription_model is None:
                return False
            
            await self.session.delete(subscription_model)
            await self.session.flush()
            
            logger.info(f"Deleted subscription: {subscription_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting subscription {subscription_id}: {e}")
            raise
    
    def _to_entity(self, model: SubscriptionModel) -> Subscription:
        """Convert model to entity"""
        payment_method = self._dict_to_payment_method(model.payment_method) if model.payment_method else None
        
        return Subscription(
            id=model.id,
            tenant_id=model.tenant_id,
            tier=SubscriptionTier(model.tier),
            status=model.status,
            stripe_subscription_id=model.stripe_subscription_id,
            stripe_customer_id=model.stripe_customer_id,
            current_period_start=model.current_period_start,
            current_period_end=model.current_period_end,
            trial_start=model.trial_start,
            trial_end=model.trial_end,
            canceled_at=model.canceled_at,
            cancel_at_period_end=model.cancel_at_period_end,
            amount_cents=model.amount_cents,
            currency=model.currency,
            billing_cycle=model.billing_cycle,
            payment_method=payment_method,
            created_at=model.created_at,
            updated_at=model.updated_at,
            metadata=model.metadata or {},
        )
    
    def _to_model(self, entity: Subscription) -> SubscriptionModel:
        """Convert entity to model"""
        return SubscriptionModel(
            id=entity.id,
            tenant_id=entity.tenant_id,
            tier=entity.tier.value,
            status=entity.status,
            stripe_subscription_id=entity.stripe_subscription_id,
            stripe_customer_id=entity.stripe_customer_id,
            current_period_start=entity.current_period_start,
            current_period_end=entity.current_period_end,
            trial_start=entity.trial_start,
            trial_end=entity.trial_end,
            canceled_at=entity.canceled_at,
            cancel_at_period_end=entity.cancel_at_period_end,
            amount_cents=entity.amount_cents,
            currency=entity.currency,
            billing_cycle=entity.billing_cycle,
            payment_method=self._payment_method_to_dict(entity.payment_method) if entity.payment_method else None,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            metadata=entity.metadata,
        )
    
    def _payment_method_to_dict(self, payment_method: PaymentMethod) -> dict:
        """Convert PaymentMethod to dict for JSON storage"""
        return {
            "id": payment_method.id,
            "type": payment_method.type,
            "last4": payment_method.last4,
            "brand": payment_method.brand,
            "exp_month": payment_method.exp_month,
            "exp_year": payment_method.exp_year,
            "is_default": payment_method.is_default,
        }
    
    def _dict_to_payment_method(self, data: dict) -> PaymentMethod:
        """Convert dict to PaymentMethod"""
        return PaymentMethod(
            id=data["id"],
            type=data["type"],
            last4=data["last4"],
            brand=data.get("brand"),
            exp_month=data.get("exp_month"),
            exp_year=data.get("exp_year"),
            is_default=data.get("is_default", False),
        )


class PostgreSQLInvoiceRepository(InvoiceRepository):
    """PostgreSQL implementation of InvoiceRepository"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, invoice: Invoice) -> Invoice:
        """Create a new invoice"""
        try:
            invoice_model = self._to_model(invoice)
            self.session.add(invoice_model)
            await self.session.flush()
            await self.session.refresh(invoice_model)
            
            logger.info(f"Created invoice: {invoice.id}")
            return self._to_entity(invoice_model)
        except Exception as e:
            logger.error(f"Error creating invoice {invoice.id}: {e}")
            raise
    
    async def get_by_id(self, invoice_id: UUID) -> Optional[Invoice]:
        """Retrieve invoice by ID"""
        try:
            stmt = select(InvoiceModel).where(InvoiceModel.id == invoice_id)
            result = await self.session.execute(stmt)
            invoice_model = result.scalar_one_or_none()
            
            if invoice_model is None:
                return None
            
            return self._to_entity(invoice_model)
        except Exception as e:
            logger.error(f"Error fetching invoice {invoice_id}: {e}")
            raise
    
    async def get_by_tenant(
        self,
        tenant_id: UUID,
        limit: int = 100,
        offset: int = 0
    ) -> List[Invoice]:
        """Get invoices for a tenant"""
        try:
            stmt = (
                select(InvoiceModel)
                .where(InvoiceModel.tenant_id == tenant_id)
                .order_by(InvoiceModel.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            invoice_models = result.scalars().all()
            
            return [self._to_entity(model) for model in invoice_models]
        except Exception as e:
            logger.error(f"Error fetching invoices for tenant {tenant_id}: {e}")
            raise
    
    async def get_unpaid_invoices(
        self,
        tenant_id: Optional[UUID] = None
    ) -> List[Invoice]:
        """Get all unpaid invoices"""
        try:
            conditions = [InvoiceModel.status.in_([InvoiceStatus.OPEN, InvoiceStatus.DRAFT])]
            if tenant_id:
                conditions.append(InvoiceModel.tenant_id == tenant_id)
            
            stmt = select(InvoiceModel).where(and_(*conditions))
            result = await self.session.execute(stmt)
            invoice_models = result.scalars().all()
            
            return [self._to_entity(model) for model in invoice_models]
        except Exception as e:
            logger.error(f"Error fetching unpaid invoices: {e}")
            raise
    
    async def get_by_subscription(
        self,
        subscription_id: UUID,
        limit: int = 100
    ) -> List[Invoice]:
        """Get invoices for a subscription"""
        try:
            stmt = (
                select(InvoiceModel)
                .where(InvoiceModel.subscription_id == subscription_id)
                .order_by(InvoiceModel.created_at.desc())
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            invoice_models = result.scalars().all()
            
            return [self._to_entity(model) for model in invoice_models]
        except Exception as e:
            logger.error(f"Error fetching invoices for subscription {subscription_id}: {e}")
            raise
    
    async def get_overdue_invoices(self, as_of: datetime) -> List[Invoice]:
        """Get overdue invoices"""
        try:
            stmt = (
                select(InvoiceModel)
                .where(
                    and_(
                        InvoiceModel.status == InvoiceStatus.OPEN,
                        InvoiceModel.due_date.isnot(None),
                        InvoiceModel.due_date < as_of
                    )
                )
            )
            result = await self.session.execute(stmt)
            invoice_models = result.scalars().all()
            
            return [self._to_entity(model) for model in invoice_models]
        except Exception as e:
            logger.error(f"Error fetching overdue invoices: {e}")
            raise
    
    async def get_by_stripe_invoice_id(
        self,
        stripe_invoice_id: str
    ) -> Optional[Invoice]:
        """Get invoice by Stripe invoice ID"""
        try:
            stmt = (
                select(InvoiceModel)
                .where(InvoiceModel.stripe_invoice_id == stripe_invoice_id)
            )
            result = await self.session.execute(stmt)
            invoice_model = result.scalar_one_or_none()
            
            if invoice_model is None:
                return None
            
            return self._to_entity(invoice_model)
        except Exception as e:
            logger.error(f"Error fetching invoice by Stripe ID {stripe_invoice_id}: {e}")
            raise
    
    async def update(self, invoice: Invoice) -> Invoice:
        """Update an existing invoice"""
        try:
            stmt = select(InvoiceModel).where(InvoiceModel.id == invoice.id)
            result = await self.session.execute(stmt)
            invoice_model = result.scalar_one_or_none()
            
            if invoice_model is None:
                raise ValueError(f"Invoice not found: {invoice.id}")
            
            # Update fields
            invoice_model.tenant_id = invoice.tenant_id
            invoice_model.subscription_id = invoice.subscription_id
            invoice_model.status = invoice.status
            invoice_model.invoice_number = invoice.invoice_number
            invoice_model.amount_cents = invoice.amount_cents
            invoice_model.amount_paid_cents = invoice.amount_paid_cents
            invoice_model.amount_due_cents = invoice.amount_due_cents
            invoice_model.currency = invoice.currency
            invoice_model.stripe_invoice_id = invoice.stripe_invoice_id
            invoice_model.period_start = invoice.period_start
            invoice_model.period_end = invoice.period_end
            invoice_model.due_date = invoice.due_date
            invoice_model.paid_at = invoice.paid_at
            invoice_model.line_items = invoice.line_items
            invoice_model.metadata = invoice.metadata
            
            await self.session.flush()
            await self.session.refresh(invoice_model)
            
            logger.info(f"Updated invoice: {invoice.id}")
            return self._to_entity(invoice_model)
        except Exception as e:
            logger.error(f"Error updating invoice {invoice.id}: {e}")
            raise
    
    async def delete(self, invoice_id: UUID) -> bool:
        """Delete an invoice"""
        try:
            stmt = select(InvoiceModel).where(InvoiceModel.id == invoice_id)
            result = await self.session.execute(stmt)
            invoice_model = result.scalar_one_or_none()
            
            if invoice_model is None:
                return False
            
            await self.session.delete(invoice_model)
            await self.session.flush()
            
            logger.info(f"Deleted invoice: {invoice_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting invoice {invoice_id}: {e}")
            raise
    
    def _to_entity(self, model: InvoiceModel) -> Invoice:
        """Convert model to entity"""
        return Invoice(
            id=model.id,
            tenant_id=model.tenant_id,
            subscription_id=model.subscription_id,
            status=model.status,
            amount_cents=model.amount_cents,
            amount_paid_cents=model.amount_paid_cents,
            amount_due_cents=model.amount_due_cents,
            currency=model.currency,
            stripe_invoice_id=model.stripe_invoice_id,
            invoice_number=model.invoice_number,
            period_start=model.period_start,
            period_end=model.period_end,
            due_date=model.due_date,
            paid_at=model.paid_at,
            created_at=model.created_at,
            line_items=model.line_items or [],
            metadata=model.metadata or {},
        )
    
    def _to_model(self, entity: Invoice) -> InvoiceModel:
        """Convert entity to model"""
        return InvoiceModel(
            id=entity.id,
            tenant_id=entity.tenant_id,
            subscription_id=entity.subscription_id,
            status=entity.status,
            invoice_number=entity.invoice_number,
            amount_cents=entity.amount_cents,
            amount_paid_cents=entity.amount_paid_cents,
            amount_due_cents=entity.amount_due_cents,
            currency=entity.currency,
            stripe_invoice_id=entity.stripe_invoice_id,
            period_start=entity.period_start,
            period_end=entity.period_end,
            due_date=entity.due_date,
            paid_at=entity.paid_at,
            created_at=entity.created_at,
            line_items=entity.line_items,
            metadata=entity.metadata,
        )


class PostgreSQLUsageRecordRepository(UsageRecordRepository):
    """PostgreSQL implementation of UsageRecordRepository"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, usage_record: UsageRecord) -> UsageRecord:
        """Create a new usage record"""
        try:
            usage_record_model = self._to_model(usage_record)
            self.session.add(usage_record_model)
            await self.session.flush()
            await self.session.refresh(usage_record_model)
            
            return self._to_entity(usage_record_model)
        except Exception as e:
            logger.error(f"Error creating usage record {usage_record.id}: {e}")
            raise
    
    async def get_by_id(self, record_id: UUID) -> Optional[UsageRecord]:
        """Retrieve usage record by ID"""
        try:
            stmt = select(UsageRecordModel).where(UsageRecordModel.id == record_id)
            result = await self.session.execute(stmt)
            usage_record_model = result.scalar_one_or_none()
            
            if usage_record_model is None:
                return None
            
            return self._to_entity(usage_record_model)
        except Exception as e:
            logger.error(f"Error fetching usage record {record_id}: {e}")
            raise
    
    async def get_by_tenant(
        self,
        tenant_id: UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[UsageRecord]:
        """Get usage records for a tenant within a time range"""
        try:
            conditions = [UsageRecordModel.tenant_id == tenant_id]
            if start_time:
                conditions.append(UsageRecordModel.timestamp >= start_time)
            if end_time:
                conditions.append(UsageRecordModel.timestamp <= end_time)
            
            stmt = (
                select(UsageRecordModel)
                .where(and_(*conditions))
                .order_by(UsageRecordModel.timestamp.desc())
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            usage_record_models = result.scalars().all()
            
            return [self._to_entity(model) for model in usage_record_models]
        except Exception as e:
            logger.error(f"Error fetching usage records for tenant {tenant_id}: {e}")
            raise
    
    async def aggregate_usage(
        self,
        tenant_id: UUID,
        resource_type: str,
        start_time: datetime,
        end_time: datetime
    ) -> Decimal:
        """Aggregate usage for a tenant and resource type over a period"""
        try:
            stmt = (
                select(func.sum(UsageRecordModel.quantity))
                .where(
                    and_(
                        UsageRecordModel.tenant_id == tenant_id,
                        UsageRecordModel.resource_type == resource_type,
                        UsageRecordModel.timestamp >= start_time,
                        UsageRecordModel.timestamp <= end_time
                    )
                )
            )
            result = await self.session.execute(stmt)
            total = result.scalar()
            
            return Decimal(total) if total else Decimal("0")
        except Exception as e:
            logger.error(f"Error aggregating usage for tenant {tenant_id}: {e}")
            raise
    
    async def get_by_resource_type(
        self,
        tenant_id: UUID,
        resource_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[UsageRecord]:
        """Get usage records filtered by resource type"""
        try:
            conditions = [
                UsageRecordModel.tenant_id == tenant_id,
                UsageRecordModel.resource_type == resource_type
            ]
            if start_time:
                conditions.append(UsageRecordModel.timestamp >= start_time)
            if end_time:
                conditions.append(UsageRecordModel.timestamp <= end_time)
            
            stmt = (
                select(UsageRecordModel)
                .where(and_(*conditions))
                .order_by(UsageRecordModel.timestamp.desc())
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            usage_record_models = result.scalars().all()
            
            return [self._to_entity(model) for model in usage_record_models]
        except Exception as e:
            logger.error(f"Error fetching usage records by resource type {resource_type}: {e}")
            raise
    
    async def bulk_create(self, usage_records: List[UsageRecord]) -> List[UsageRecord]:
        """Create multiple usage records in bulk"""
        try:
            usage_record_models = [self._to_model(record) for record in usage_records]
            self.session.add_all(usage_record_models)
            await self.session.flush()
            
            # Refresh all models
            for model in usage_record_models:
                await self.session.refresh(model)
            
            logger.info(f"Bulk created {len(usage_records)} usage records")
            return [self._to_entity(model) for model in usage_record_models]
        except Exception as e:
            logger.error(f"Error bulk creating usage records: {e}")
            raise
    
    async def delete_older_than(self, before: datetime) -> int:
        """Delete usage records older than specified date"""
        try:
            stmt = select(UsageRecordModel).where(UsageRecordModel.timestamp < before)
            result = await self.session.execute(stmt)
            usage_record_models = result.scalars().all()
            
            count = len(usage_record_models)
            for model in usage_record_models:
                await self.session.delete(model)
            
            await self.session.flush()
            
            logger.info(f"Deleted {count} usage records older than {before}")
            return count
        except Exception as e:
            logger.error(f"Error deleting old usage records: {e}")
            raise
    
    def _to_entity(self, model: UsageRecordModel) -> UsageRecord:
        """Convert model to entity"""
        return UsageRecord(
            id=model.id,
            tenant_id=model.tenant_id,
            resource_type=model.resource_type,
            quantity=model.quantity,
            unit=model.unit,
            timestamp=model.timestamp,
            metadata=model.metadata or {},
        )
    
    def _to_model(self, entity: UsageRecord) -> UsageRecordModel:
        """Convert entity to model"""
        return UsageRecordModel(
            id=entity.id,
            tenant_id=entity.tenant_id,
            resource_type=entity.resource_type,
            quantity=entity.quantity,
            unit=entity.unit,
            timestamp=entity.timestamp,
            metadata=entity.metadata,
        )
