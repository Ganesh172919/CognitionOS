"""
Usage Tracker Infrastructure

Background usage tracking with async batching for performance.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from uuid import UUID

from core.domain.billing.entities import UsageRecord
from core.domain.billing.repositories import UsageRecordRepository

logger = logging.getLogger(__name__)


class UsageTracker:
    """
    Usage tracker with background aggregation and batching.
    
    Buffers usage records in memory and periodically flushes them to
    the database in batches for improved performance.
    """
    
    def __init__(
        self,
        usage_repository: UsageRecordRepository,
        batch_size: int = 100,
        flush_interval_seconds: float = 30.0,
        auto_start: bool = True,
    ):
        """
        Initialize usage tracker.
        
        Args:
            usage_repository: Repository for persisting usage records
            batch_size: Maximum number of records to batch before flushing
            flush_interval_seconds: Time interval for automatic flushing
            auto_start: Whether to start background task automatically
        """
        self.usage_repository = usage_repository
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        
        self._buffer: List[UsageRecord] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._flush_event = asyncio.Event()
        
        self._stats = {
            "tracked": 0,
            "flushed": 0,
            "errors": 0,
            "last_flush": None,
        }
        
        logger.info(
            f"UsageTracker initialized (batch_size={batch_size}, "
            f"flush_interval={flush_interval_seconds}s)"
        )
        
        if auto_start:
            self.start()
    
    def start(self) -> None:
        """Start background flush task."""
        if self._running:
            logger.warning("UsageTracker already running")
            return
        
        self._running = True
        self._flush_task = asyncio.create_task(self._background_flush())
        logger.info("UsageTracker background flush task started")
    
    async def stop(self) -> None:
        """Stop background flush task and flush remaining records."""
        if not self._running:
            return
        
        logger.info("Stopping UsageTracker...")
        self._running = False
        
        if self._flush_task:
            self._flush_event.set()
            await self._flush_task
        
        await self.flush()
        logger.info("UsageTracker stopped")
    
    async def track(
        self,
        tenant_id: UUID,
        resource_type: str,
        quantity: Decimal,
        unit: str = "count",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Track resource usage asynchronously.
        
        Adds usage to buffer and flushes when batch size is reached.
        
        Args:
            tenant_id: Tenant identifier
            resource_type: Type of resource consumed
            quantity: Quantity consumed
            unit: Unit of measurement
            metadata: Optional usage metadata
        """
        usage_record = UsageRecord.create(
            tenant_id=tenant_id,
            resource_type=resource_type,
            quantity=quantity,
            unit=unit,
            metadata=metadata,
        )
        
        async with self._buffer_lock:
            self._buffer.append(usage_record)
            self._stats["tracked"] += 1
            
            logger.debug(
                f"Tracked usage: tenant={tenant_id}, resource={resource_type}, "
                f"quantity={quantity}, buffer_size={len(self._buffer)}",
                extra={
                    "tenant_id": str(tenant_id),
                    "resource_type": resource_type,
                    "quantity": str(quantity),
                    "buffer_size": len(self._buffer),
                }
            )
            
            if len(self._buffer) >= self.batch_size:
                self._flush_event.set()
    
    async def track_execution(
        self,
        tenant_id: UUID,
        workflow_id: Optional[UUID] = None,
        agent_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Track workflow or agent execution.
        
        Args:
            tenant_id: Tenant identifier
            workflow_id: Optional workflow identifier
            agent_id: Optional agent identifier
            metadata: Optional metadata
        """
        exec_metadata = metadata or {}
        if workflow_id:
            exec_metadata["workflow_id"] = str(workflow_id)
        if agent_id:
            exec_metadata["agent_id"] = str(agent_id)
        
        await self.track(
            tenant_id=tenant_id,
            resource_type="executions",
            quantity=Decimal("1"),
            unit="count",
            metadata=exec_metadata,
        )
    
    async def track_api_call(
        self,
        tenant_id: UUID,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Track API call.
        
        Args:
            tenant_id: Tenant identifier
            endpoint: Optional API endpoint
            method: Optional HTTP method
            metadata: Optional metadata
        """
        api_metadata = metadata or {}
        if endpoint:
            api_metadata["endpoint"] = endpoint
        if method:
            api_metadata["method"] = method
        
        await self.track(
            tenant_id=tenant_id,
            resource_type="api_calls",
            quantity=Decimal("1"),
            unit="count",
            metadata=api_metadata,
        )
    
    async def track_tokens(
        self,
        tenant_id: UUID,
        tokens: int,
        model: Optional[str] = None,
        operation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Track LLM token usage.
        
        Args:
            tenant_id: Tenant identifier
            tokens: Number of tokens consumed
            model: Optional model name
            operation: Optional operation type
            metadata: Optional metadata
        """
        token_metadata = metadata or {}
        if model:
            token_metadata["model"] = model
        if operation:
            token_metadata["operation"] = operation
        
        await self.track(
            tenant_id=tenant_id,
            resource_type="tokens",
            quantity=Decimal(str(tokens)),
            unit="tokens",
            metadata=token_metadata,
        )
    
    async def track_storage(
        self,
        tenant_id: UUID,
        size_mb: Decimal,
        storage_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Track storage usage.
        
        Args:
            tenant_id: Tenant identifier
            size_mb: Size in megabytes
            storage_type: Optional storage type (e.g., "checkpoint", "artifact")
            metadata: Optional metadata
        """
        storage_metadata = metadata or {}
        if storage_type:
            storage_metadata["storage_type"] = storage_type
        
        await self.track(
            tenant_id=tenant_id,
            resource_type="storage_mb",
            quantity=size_mb,
            unit="megabytes",
            metadata=storage_metadata,
        )
    
    async def flush(self) -> int:
        """
        Flush all buffered usage records to database.
        
        Returns:
            Number of records flushed
        """
        async with self._buffer_lock:
            if not self._buffer:
                return 0
            
            records_to_flush = self._buffer.copy()
            self._buffer.clear()
        
        try:
            await self.usage_repository.bulk_create(records_to_flush)
            
            flushed_count = len(records_to_flush)
            self._stats["flushed"] += flushed_count
            self._stats["last_flush"] = datetime.utcnow()
            
            logger.info(
                f"Flushed {flushed_count} usage records to database",
                extra={"flushed_count": flushed_count}
            )
            
            return flushed_count
            
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(
                f"Error flushing usage records: {e}",
                exc_info=True,
                extra={"error": str(e), "record_count": len(records_to_flush)}
            )
            
            async with self._buffer_lock:
                self._buffer.extend(records_to_flush)
            
            raise
    
    async def _background_flush(self) -> None:
        """Background task that periodically flushes the buffer."""
        logger.info("Background flush task started")
        
        while self._running:
            try:
                await asyncio.wait_for(
                    self._flush_event.wait(),
                    timeout=self.flush_interval
                )
                self._flush_event.clear()
            except asyncio.TimeoutError:
                pass
            
            if not self._running:
                break
            
            try:
                buffer_size = len(self._buffer)
                if buffer_size > 0:
                    await self.flush()
            except Exception as e:
                logger.error(f"Error in background flush: {e}", exc_info=True)
                await asyncio.sleep(5.0)
        
        logger.info("Background flush task stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage tracker statistics.
        
        Returns:
            Dictionary with tracker statistics
        """
        return {
            **self._stats,
            "buffer_size": len(self._buffer),
            "batch_size": self.batch_size,
            "flush_interval": self.flush_interval,
            "running": self._running,
        }
    
    async def get_aggregated_stats(
        self,
        tenant_id: Optional[UUID] = None,
    ) -> Dict[str, Dict[str, Decimal]]:
        """
        Get aggregated usage statistics from buffer.
        
        Note: This only includes buffered (not yet flushed) records.
        For complete statistics, call flush() first.
        
        Args:
            tenant_id: Optional tenant filter
            
        Returns:
            Dictionary mapping tenant_id to resource usage totals
        """
        stats: Dict[UUID, Dict[str, Decimal]] = defaultdict(
            lambda: defaultdict(lambda: Decimal("0"))
        )
        
        async with self._buffer_lock:
            for record in self._buffer:
                if tenant_id and record.tenant_id != tenant_id:
                    continue
                
                stats[record.tenant_id][record.resource_type] += record.quantity
        
        return {str(tid): dict(resources) for tid, resources in stats.items()}
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        return False


class UsageAggregator:
    """
    Utility for aggregating historical usage data.
    
    Provides methods for analyzing usage patterns and generating reports.
    """
    
    def __init__(self, usage_repository: UsageRecordRepository):
        """
        Initialize usage aggregator.
        
        Args:
            usage_repository: Repository for querying usage records
        """
        self.usage_repository = usage_repository
    
    async def aggregate_by_tenant(
        self,
        start_time: datetime,
        end_time: datetime,
        resource_type: Optional[str] = None,
    ) -> Dict[UUID, Decimal]:
        """
        Aggregate usage by tenant for a time period.
        
        Args:
            start_time: Start of period
            end_time: End of period
            resource_type: Optional resource type filter
            
        Returns:
            Dictionary mapping tenant_id to total usage
        """
        logger.info(
            f"Aggregating usage from {start_time} to {end_time} "
            f"(resource_type={resource_type})"
        )
        
        aggregated: Dict[UUID, Decimal] = defaultdict(lambda: Decimal("0"))
        
        if resource_type:
            records = await self.usage_repository.get_by_resource_type(
                tenant_id=None,  # All tenants
                resource_type=resource_type,
                start_time=start_time,
                end_time=end_time,
            )
        else:
            # Would need a method to get all records for all tenants
            # This is a simplified implementation
            records = []
        
        for record in records:
            aggregated[record.tenant_id] += record.quantity
        
        return dict(aggregated)
    
    async def get_usage_trends(
        self,
        tenant_id: UUID,
        resource_type: str,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get daily usage trends for a tenant and resource type.
        
        Args:
            tenant_id: Tenant identifier
            resource_type: Resource type
            days: Number of days to analyze
            
        Returns:
            List of daily usage totals
        """
        from datetime import timedelta
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        records = await self.usage_repository.get_by_resource_type(
            tenant_id=tenant_id,
            resource_type=resource_type,
            start_time=start_time,
            end_time=end_time,
        )
        
        daily_usage: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        
        for record in records:
            day = record.timestamp.date().isoformat()
            daily_usage[day] += record.quantity
        
        trends = [
            {"date": day, "usage": str(usage)}
            for day, usage in sorted(daily_usage.items())
        ]
        
        logger.info(
            f"Retrieved {len(trends)} days of usage trends for tenant {tenant_id}",
            extra={
                "tenant_id": str(tenant_id),
                "resource_type": resource_type,
                "days": days,
            }
        )
        
        return trends
    
    async def get_top_consumers(
        self,
        resource_type: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get top resource consumers for a period.
        
        Args:
            resource_type: Resource type
            start_time: Start of period
            end_time: End of period
            limit: Maximum number of results
            
        Returns:
            List of top consumers with usage totals
        """
        aggregated = await self.aggregate_by_tenant(
            start_time=start_time,
            end_time=end_time,
            resource_type=resource_type,
        )
        
        sorted_consumers = sorted(
            aggregated.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        top_consumers = [
            {
                "tenant_id": str(tenant_id),
                "usage": str(usage),
                "resource_type": resource_type,
            }
            for tenant_id, usage in sorted_consumers
        ]
        
        logger.info(
            f"Retrieved top {len(top_consumers)} consumers for {resource_type}",
            extra={"resource_type": resource_type, "count": len(top_consumers)}
        )
        
        return top_consumers
