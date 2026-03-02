"""
Usage Meter Service - Token and execution tracking for billing.

Bridges infrastructure.saas.usage_metering_engine with persistence.
Wires to agent kernel and API for unified usage recording.
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional
from uuid import UUID

from core.domain.billing.entities import UsageRecord

logger = logging.getLogger(__name__)


class UsageMeterService:
    """
    Unified usage metering service.
    Records token and execution usage to usage_records (Postgres)
    and optionally to UsageMeteringEngine (in-memory/Redis).
    """

    def __init__(
        self,
        usage_record_repository,
        usage_metering_engine=None,
    ):
        self._usage_repo = usage_record_repository
        self._metering_engine = usage_metering_engine

    async def record_execution(
        self,
        tenant_id: UUID,
        event_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record an agent/workflow execution.
        Persists to usage_records for billing aggregation.
        """
        try:
            record = UsageRecord.create(
                tenant_id=tenant_id,
                resource_type="executions",
                quantity=Decimal("1"),
                unit="count",
                event_id=event_id,
                metadata=metadata or {},
            )
            await self._usage_repo.create(record)
        except Exception as e:
            logger.warning("Failed to record execution usage: %s", e, exc_info=True)

    async def record_tokens(
        self,
        tenant_id: UUID,
        total_tokens: int,
        event_id: str,
        model: Optional[str] = None,
        cost_usd: Optional[Decimal] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record token usage for billing.
        Persists to usage_records.
        """
        if total_tokens <= 0:
            return
        try:
            meta = metadata or {}
            if model is not None:
                meta["model"] = model
            if cost_usd is not None:
                meta["cost_usd"] = str(cost_usd)
            record = UsageRecord.create(
                tenant_id=tenant_id,
                resource_type="tokens",
                quantity=Decimal(str(total_tokens)),
                unit="tokens",
                event_id=event_id,
                metadata=meta,
            )
            await self._usage_repo.create(record)
        except Exception as e:
            logger.warning("Failed to record token usage: %s", e, exc_info=True)

    async def record_agent_run_usage(
        self,
        tenant_id: UUID,
        run_id: UUID,
        total_tokens: int,
        cost_usd: Decimal,
        model: Optional[str] = None,
    ) -> None:
        """
        Record usage for a completed agent run.
        Records both execution count and token usage.
        """
        exec_event_id = f"agent_run:{run_id}:executions"
        token_event_id = f"agent_run:{run_id}:tokens"
        await self.record_execution(
            tenant_id=tenant_id,
            event_id=exec_event_id,
            metadata={"run_id": str(run_id), "kind": "agent_run"},
        )
        if total_tokens > 0:
            await self.record_tokens(
                tenant_id=tenant_id,
                total_tokens=total_tokens,
                event_id=token_event_id,
                model=model,
                cost_usd=cost_usd,
                metadata={"run_id": str(run_id)},
            )
