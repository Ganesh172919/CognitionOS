"""
Webhook Event Repository

Provides persistence for webhook events to enable idempotent processing and retry handling.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from uuid import UUID

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.persistence.webhook_event_models import (
    WebhookEventModel,
    WebhookEventStatus,
)

logger = logging.getLogger(__name__)


class WebhookEventRepository:
    """
    Repository for webhook event persistence and querying.
    
    Ensures idempotent webhook processing and provides retry management.
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session
    
    async def event_exists(self, event_id: str) -> bool:
        """
        Check if an event has already been processed.
        
        Args:
            event_id: Stripe event ID
            
        Returns:
            True if event exists in database
        """
        result = await self.session.execute(
            select(WebhookEventModel).where(
                WebhookEventModel.event_id == event_id
            )
        )
        return result.scalar_one_or_none() is not None
    
    async def store_event(
        self,
        event_id: str,
        event_type: str,
        result: Dict[str, Any],
        processed_at: Optional[datetime] = None,
    ) -> WebhookEventModel:
        """
        Store successfully processed webhook event.
        
        Args:
            event_id: Stripe event ID
            event_type: Type of event (e.g., payment_intent.succeeded)
            result: Processing result data
            processed_at: Processing timestamp
            
        Returns:
            Stored webhook event model
        """
        event = WebhookEventModel(
            event_id=event_id,
            event_type=event_type,
            status=WebhookEventStatus.PROCESSED,
            result=result,
            processed_at=processed_at or datetime.utcnow(),
            created_at=datetime.utcnow(),
        )
        
        self.session.add(event)
        await self.session.flush()
        
        logger.info(f"Stored webhook event: {event_id} ({event_type})")
        
        return event
    
    async def store_failed_event(
        self,
        event_id: str,
        event_type: str,
        error: str,
        failed_at: Optional[datetime] = None,
        retry_count: int = 0,
    ) -> WebhookEventModel:
        """
        Store failed webhook event for retry.
        
        Args:
            event_id: Stripe event ID
            event_type: Type of event
            error: Error message
            failed_at: Failure timestamp
            retry_count: Number of retry attempts
            
        Returns:
            Stored webhook event model
        """
        event = WebhookEventModel(
            event_id=event_id,
            event_type=event_type,
            status=WebhookEventStatus.FAILED,
            error_message=error,
            failed_at=failed_at or datetime.utcnow(),
            retry_count=retry_count,
            created_at=datetime.utcnow(),
        )
        
        self.session.add(event)
        await self.session.flush()
        
        logger.warning(f"Stored failed webhook event: {event_id} ({event_type}) - {error}")
        
        return event
    
    async def get_event(self, event_id: str) -> Optional[WebhookEventModel]:
        """
        Retrieve webhook event by ID.
        
        Args:
            event_id: Stripe event ID
            
        Returns:
            Webhook event model or None
        """
        result = await self.session.execute(
            select(WebhookEventModel).where(
                WebhookEventModel.event_id == event_id
            )
        )
        return result.scalar_one_or_none()
    
    async def get_failed_events(
        self,
        max_retry_count: int = 5,
        limit: int = 100,
    ) -> List[WebhookEventModel]:
        """
        Get failed events eligible for retry.
        
        Args:
            max_retry_count: Maximum number of retries before giving up
            limit: Maximum number of events to return
            
        Returns:
            List of failed webhook events
        """
        result = await self.session.execute(
            select(WebhookEventModel)
            .where(
                and_(
                    WebhookEventModel.status == WebhookEventStatus.FAILED,
                    WebhookEventModel.retry_count < max_retry_count,
                )
            )
            .order_by(WebhookEventModel.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def increment_retry_count(self, event_id: str) -> WebhookEventModel:
        """
        Increment retry count for a failed event.
        
        Args:
            event_id: Stripe event ID
            
        Returns:
            Updated webhook event model
        """
        event = await self.get_event(event_id)
        if event:
            event.retry_count += 1
            event.last_retry_at = datetime.utcnow()
            await self.session.flush()
            
            logger.info(f"Incremented retry count for event {event_id}: {event.retry_count}")
        
        return event
    
    async def mark_event_processed(
        self,
        event_id: str,
        result: Dict[str, Any],
    ) -> WebhookEventModel:
        """
        Mark a failed event as successfully processed after retry.
        
        Args:
            event_id: Stripe event ID
            result: Processing result data
            
        Returns:
            Updated webhook event model
        """
        event = await self.get_event(event_id)
        if event:
            event.status = WebhookEventStatus.PROCESSED
            event.result = result
            event.processed_at = datetime.utcnow()
            event.error_message = None
            await self.session.flush()
            
            logger.info(f"Marked event as processed after retry: {event_id}")
        
        return event
    
    async def cleanup_old_events(self, days: int = 90) -> int:
        """
        Clean up old processed webhook events.
        
        Args:
            days: Delete events older than this many days
            
        Returns:
            Number of events deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        result = await self.session.execute(
            select(WebhookEventModel).where(
                and_(
                    WebhookEventModel.status == WebhookEventStatus.PROCESSED,
                    WebhookEventModel.processed_at < cutoff_date,
                )
            )
        )
        events = result.scalars().all()
        
        for event in events:
            await self.session.delete(event)
        
        await self.session.flush()
        
        logger.info(f"Cleaned up {len(events)} old webhook events")
        
        return len(events)
    
    async def get_events_by_type(
        self,
        event_type: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[WebhookEventModel]:
        """
        Get webhook events by type.
        
        Args:
            event_type: Type of event to filter by
            limit: Maximum number of events to return
            offset: Number of events to skip
            
        Returns:
            List of webhook events
        """
        result = await self.session.execute(
            select(WebhookEventModel)
            .where(WebhookEventModel.event_type == event_type)
            .order_by(WebhookEventModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())
    
    async def get_event_statistics(
        self,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Get webhook event processing statistics.
        
        Args:
            hours: Calculate statistics for last N hours
            
        Returns:
            Dictionary with event statistics
        """
        since = datetime.utcnow() - timedelta(hours=hours)
        
        # Get all events in time window
        result = await self.session.execute(
            select(WebhookEventModel).where(
                WebhookEventModel.created_at >= since
            )
        )
        events = result.scalars().all()
        
        # Calculate statistics
        total_events = len(events)
        processed_events = sum(1 for e in events if e.status == WebhookEventStatus.PROCESSED)
        failed_events = sum(1 for e in events if e.status == WebhookEventStatus.FAILED)
        
        # Group by event type
        by_type = {}
        for event in events:
            if event.event_type not in by_type:
                by_type[event.event_type] = {"total": 0, "processed": 0, "failed": 0}
            by_type[event.event_type]["total"] += 1
            if event.status == WebhookEventStatus.PROCESSED:
                by_type[event.event_type]["processed"] += 1
            elif event.status == WebhookEventStatus.FAILED:
                by_type[event.event_type]["failed"] += 1
        
        return {
            "time_window_hours": hours,
            "total_events": total_events,
            "processed_events": processed_events,
            "failed_events": failed_events,
            "success_rate": (processed_events / total_events * 100) if total_events > 0 else 0,
            "by_type": by_type,
        }
