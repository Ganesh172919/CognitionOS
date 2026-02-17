"""
Billing Domain - Repository Interfaces

Abstract repository interfaces for billing aggregate persistence.
Implementations provided by infrastructure layer.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import List, Optional
from uuid import UUID

from .entities import (
    Subscription,
    SubscriptionStatus,
    SubscriptionTier,
    Invoice,
    InvoiceStatus,
    UsageRecord,
)


class SubscriptionRepository(ABC):
    """
    Repository interface for Subscription aggregate.
    
    Handles persistence of tenant subscriptions and billing cycles.
    """
    
    @abstractmethod
    async def create(self, subscription: Subscription) -> Subscription:
        """
        Create a new subscription.
        
        Args:
            subscription: Subscription entity to create
            
        Returns:
            Created subscription
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, subscription_id: UUID) -> Optional[Subscription]:
        """
        Retrieve subscription by ID.
        
        Args:
            subscription_id: Subscription identifier
            
        Returns:
            Subscription if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_by_tenant(self, tenant_id: UUID) -> Optional[Subscription]:
        """
        Get active subscription for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Active subscription if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_active_subscriptions(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Subscription]:
        """
        Get all active subscriptions.
        
        Args:
            limit: Maximum number of subscriptions to return
            offset: Number of subscriptions to skip
            
        Returns:
            List of active subscriptions
        """
        pass
    
    @abstractmethod
    async def get_by_status(
        self,
        status: SubscriptionStatus,
        limit: int = 100
    ) -> List[Subscription]:
        """
        Get subscriptions by status.
        
        Args:
            status: Subscription status filter
            limit: Maximum number of subscriptions to return
            
        Returns:
            List of subscriptions matching status
        """
        pass
    
    @abstractmethod
    async def get_expiring_trials(
        self,
        before: datetime
    ) -> List[Subscription]:
        """
        Get trial subscriptions expiring before specified date.
        
        Args:
            before: Date threshold for trial expiration
            
        Returns:
            List of expiring trial subscriptions
        """
        pass
    
    @abstractmethod
    async def get_by_stripe_subscription_id(
        self,
        stripe_subscription_id: str
    ) -> Optional[Subscription]:
        """
        Get subscription by Stripe subscription ID.
        
        Args:
            stripe_subscription_id: Stripe subscription identifier
            
        Returns:
            Subscription if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def update(self, subscription: Subscription) -> Subscription:
        """
        Update an existing subscription.
        
        Args:
            subscription: Subscription entity to update
            
        Returns:
            Updated subscription
        """
        pass
    
    @abstractmethod
    async def delete(self, subscription_id: UUID) -> bool:
        """
        Delete a subscription.
        
        Args:
            subscription_id: Subscription identifier
            
        Returns:
            True if deleted, False if not found
        """
        pass


class InvoiceRepository(ABC):
    """
    Repository interface for Invoice aggregate.
    
    Handles persistence of billing invoices and payment records.
    """
    
    @abstractmethod
    async def create(self, invoice: Invoice) -> Invoice:
        """
        Create a new invoice.
        
        Args:
            invoice: Invoice entity to create
            
        Returns:
            Created invoice
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, invoice_id: UUID) -> Optional[Invoice]:
        """
        Retrieve invoice by ID.
        
        Args:
            invoice_id: Invoice identifier
            
        Returns:
            Invoice if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_by_tenant(
        self,
        tenant_id: UUID,
        limit: int = 100,
        offset: int = 0
    ) -> List[Invoice]:
        """
        Get invoices for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            limit: Maximum number of invoices to return
            offset: Number of invoices to skip
            
        Returns:
            List of invoices for the tenant
        """
        pass
    
    @abstractmethod
    async def get_unpaid_invoices(
        self,
        tenant_id: Optional[UUID] = None
    ) -> List[Invoice]:
        """
        Get all unpaid invoices.
        
        Args:
            tenant_id: Optional tenant filter
            
        Returns:
            List of unpaid invoices
        """
        pass
    
    @abstractmethod
    async def get_by_subscription(
        self,
        subscription_id: UUID,
        limit: int = 100
    ) -> List[Invoice]:
        """
        Get invoices for a subscription.
        
        Args:
            subscription_id: Subscription identifier
            limit: Maximum number of invoices to return
            
        Returns:
            List of invoices for the subscription
        """
        pass
    
    @abstractmethod
    async def get_overdue_invoices(
        self,
        as_of: datetime
    ) -> List[Invoice]:
        """
        Get overdue invoices.
        
        Args:
            as_of: Date to check overdue status against
            
        Returns:
            List of overdue invoices
        """
        pass
    
    @abstractmethod
    async def get_by_stripe_invoice_id(
        self,
        stripe_invoice_id: str
    ) -> Optional[Invoice]:
        """
        Get invoice by Stripe invoice ID.
        
        Args:
            stripe_invoice_id: Stripe invoice identifier
            
        Returns:
            Invoice if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def update(self, invoice: Invoice) -> Invoice:
        """
        Update an existing invoice.
        
        Args:
            invoice: Invoice entity to update
            
        Returns:
            Updated invoice
        """
        pass
    
    @abstractmethod
    async def delete(self, invoice_id: UUID) -> bool:
        """
        Delete an invoice.
        
        Args:
            invoice_id: Invoice identifier
            
        Returns:
            True if deleted, False if not found
        """
        pass


class UsageRecordRepository(ABC):
    """
    Repository interface for UsageRecord aggregate.
    
    Handles persistence of usage metering records for billing.
    """
    
    @abstractmethod
    async def create(self, usage_record: UsageRecord) -> UsageRecord:
        """
        Create a new usage record.
        
        Args:
            usage_record: UsageRecord entity to create
            
        Returns:
            Created usage record
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, record_id: UUID) -> Optional[UsageRecord]:
        """
        Retrieve usage record by ID.
        
        Args:
            record_id: Usage record identifier
            
        Returns:
            UsageRecord if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_by_tenant(
        self,
        tenant_id: UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[UsageRecord]:
        """
        Get usage records for a tenant within a time range.
        
        Args:
            tenant_id: Tenant identifier
            start_time: Optional start of time range
            end_time: Optional end of time range
            limit: Maximum number of records to return
            
        Returns:
            List of usage records
        """
        pass
    
    @abstractmethod
    async def aggregate_usage(
        self,
        tenant_id: UUID,
        resource_type: str,
        start_time: datetime,
        end_time: datetime
    ) -> Decimal:
        """
        Aggregate usage for a tenant and resource type over a period.
        
        Args:
            tenant_id: Tenant identifier
            resource_type: Type of resource to aggregate
            start_time: Start of aggregation period
            end_time: End of aggregation period
            
        Returns:
            Total usage quantity
        """
        pass
    
    @abstractmethod
    async def get_by_resource_type(
        self,
        tenant_id: UUID,
        resource_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[UsageRecord]:
        """
        Get usage records filtered by resource type.
        
        Args:
            tenant_id: Tenant identifier
            resource_type: Type of resource to filter
            start_time: Optional start of time range
            end_time: Optional end of time range
            limit: Maximum number of records to return
            
        Returns:
            List of usage records for the resource type
        """
        pass
    
    @abstractmethod
    async def bulk_create(self, usage_records: List[UsageRecord]) -> List[UsageRecord]:
        """
        Create multiple usage records in bulk.
        
        Args:
            usage_records: List of usage records to create
            
        Returns:
            List of created usage records
        """
        pass
    
    @abstractmethod
    async def delete_older_than(
        self,
        before: datetime
    ) -> int:
        """
        Delete usage records older than specified date.
        
        Args:
            before: Delete records before this date
            
        Returns:
            Number of records deleted
        """
        pass
