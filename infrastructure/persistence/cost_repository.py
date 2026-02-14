"""
Cost Governance Infrastructure - PostgreSQL Repository Implementation

Concrete implementation of WorkflowBudgetRepository and CostTrackingRepository using PostgreSQL.
"""

from typing import List, Optional
from uuid import UUID
from decimal import Decimal

from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from core.domain.cost_governance.entities import (
    WorkflowBudget,
    CostEntry,
    BudgetStatus,
    OperationType
)
from core.domain.cost_governance.repositories import (
    WorkflowBudgetRepository,
    CostTrackingRepository
)

from infrastructure.persistence.cost_models import (
    WorkflowBudgetModel,
    CostTrackingModel,
    BudgetStatusEnum,
    OperationTypeEnum
)


class PostgreSQLWorkflowBudgetRepository(WorkflowBudgetRepository):
    """
    PostgreSQL implementation of WorkflowBudgetRepository.

    Maps between domain entities and SQLAlchemy models.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, budget: WorkflowBudget) -> None:
        """Persist budget to database"""
        model = self._to_model(budget)
        self.session.add(model)
        await self.session.flush()

    async def find_by_id(self, budget_id: UUID) -> Optional[WorkflowBudget]:
        """Retrieve budget by ID"""
        stmt = select(WorkflowBudgetModel).where(WorkflowBudgetModel.id == budget_id)
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._to_entity(model)

    async def find_by_workflow_execution(
        self,
        workflow_execution_id: UUID,
    ) -> Optional[WorkflowBudget]:
        """Find budget for a workflow execution"""
        stmt = select(WorkflowBudgetModel).where(
            WorkflowBudgetModel.workflow_execution_id == workflow_execution_id
        )
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._to_entity(model)

    async def update(self, budget: WorkflowBudget) -> None:
        """Update an existing budget"""
        model = self._to_model(budget)
        self.session.add(model)
        await self.session.flush()

    async def delete(self, budget_id: UUID) -> bool:
        """Delete a budget"""
        stmt = select(WorkflowBudgetModel).where(WorkflowBudgetModel.id == budget_id)
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return False
        
        await self.session.delete(model)
        await self.session.flush()
        return True

    async def exists(self, workflow_execution_id: UUID) -> bool:
        """Check if a budget exists for a workflow execution"""
        budget = await self.find_by_workflow_execution(workflow_execution_id)
        return budget is not None

    def _to_model(self, budget: WorkflowBudget) -> WorkflowBudgetModel:
        """Convert domain entity to ORM model"""
        # Calculate threshold values if percentages are provided
        warning_threshold_value = budget.allocated_budget * budget.warning_threshold
        critical_threshold_value = budget.allocated_budget * budget.critical_threshold
        
        return WorkflowBudgetModel(
            id=budget.id,
            workflow_execution_id=budget.workflow_execution_id,
            allocated_budget=Decimal(str(budget.allocated_budget)),
            consumed_budget=Decimal(str(budget.consumed_budget)),
            warning_threshold=Decimal(str(warning_threshold_value)),
            critical_threshold=Decimal(str(critical_threshold_value)),
            status=BudgetStatusEnum(budget.status.value),
            warnings_sent=budget.warnings_sent,
            halt_triggered_at=budget.halt_triggered_at,
            suspended_at=budget.suspended_at,
            created_at=budget.created_at,
            updated_at=budget.updated_at,
            metadata=budget.metadata
        )

    def _to_entity(self, model: WorkflowBudgetModel) -> WorkflowBudget:
        """Convert ORM model to domain entity"""
        allocated_budget = float(model.allocated_budget)
        consumed_budget = float(model.consumed_budget)
        
        # Convert absolute thresholds back to percentages
        warning_threshold = 0.8
        critical_threshold = 0.95
        
        if model.warning_threshold and allocated_budget > 0:
            warning_threshold = float(model.warning_threshold) / allocated_budget
        
        if model.critical_threshold and allocated_budget > 0:
            critical_threshold = float(model.critical_threshold) / allocated_budget
        
        return WorkflowBudget(
            id=model.id,
            workflow_execution_id=model.workflow_execution_id,
            allocated_budget=allocated_budget,
            consumed_budget=consumed_budget,
            currency="USD",
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            status=BudgetStatus(model.status.value),
            warnings_sent=model.warnings_sent or 0,
            halt_triggered_at=model.halt_triggered_at,
            suspended_at=model.suspended_at,
            metadata=model.metadata or {},
            created_at=model.created_at,
            updated_at=model.updated_at
        )


class PostgreSQLCostTrackingRepository(CostTrackingRepository):
    """
    PostgreSQL implementation of CostTrackingRepository.

    Maps between domain entities and SQLAlchemy models.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, cost_entry: CostEntry) -> None:
        """Persist cost entry to database"""
        model = self._to_model(cost_entry)
        self.session.add(model)
        await self.session.flush()

    async def find_by_id(self, cost_entry_id: UUID) -> Optional[CostEntry]:
        """Retrieve cost entry by ID"""
        stmt = select(CostTrackingModel).where(CostTrackingModel.id == cost_entry_id)
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._to_entity(model)

    async def find_by_workflow_execution(
        self,
        workflow_execution_id: UUID,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[CostEntry]:
        """Find cost entries for a workflow execution"""
        stmt = (
            select(CostTrackingModel)
            .where(CostTrackingModel.workflow_execution_id == workflow_execution_id)
            .order_by(CostTrackingModel.created_at.desc())
        )
        
        if offset:
            stmt = stmt.offset(offset)
        
        if limit:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]

    async def find_by_agent(
        self,
        agent_id: UUID,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[CostEntry]:
        """Find cost entries for a specific agent"""
        stmt = (
            select(CostTrackingModel)
            .where(CostTrackingModel.agent_id == agent_id)
            .order_by(CostTrackingModel.created_at.desc())
        )
        
        if offset:
            stmt = stmt.offset(offset)
        
        if limit:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]

    async def find_by_operation_type(
        self,
        workflow_execution_id: UUID,
        operation_type: OperationType,
        limit: Optional[int] = None,
    ) -> List[CostEntry]:
        """Find cost entries by operation type"""
        db_operation_type = OperationTypeEnum(operation_type.value)
        
        stmt = (
            select(CostTrackingModel)
            .where(
                CostTrackingModel.workflow_execution_id == workflow_execution_id,
                CostTrackingModel.operation_type == db_operation_type
            )
            .order_by(CostTrackingModel.created_at.desc())
        )
        
        if limit:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]

    async def get_total_cost(
        self,
        workflow_execution_id: UUID,
    ) -> float:
        """Get total cost for a workflow execution"""
        stmt = (
            select(func.sum(CostTrackingModel.cost))
            .where(CostTrackingModel.workflow_execution_id == workflow_execution_id)
        )
        
        result = await self.session.execute(stmt)
        total = result.scalar_one_or_none()
        
        return float(total) if total else 0.0

    async def get_cost_by_operation_type(
        self,
        workflow_execution_id: UUID,
    ) -> dict[OperationType, float]:
        """Get cost breakdown by operation type"""
        stmt = (
            select(
                CostTrackingModel.operation_type,
                func.sum(CostTrackingModel.cost)
            )
            .where(CostTrackingModel.workflow_execution_id == workflow_execution_id)
            .group_by(CostTrackingModel.operation_type)
        )
        
        result = await self.session.execute(stmt)
        rows = result.all()
        
        return {
            OperationType(row[0].value): float(row[1])
            for row in rows
        }

    async def get_cost_by_agent(
        self,
        workflow_execution_id: UUID,
    ) -> dict[UUID, float]:
        """Get cost breakdown by agent"""
        stmt = (
            select(
                CostTrackingModel.agent_id,
                func.sum(CostTrackingModel.cost)
            )
            .where(
                CostTrackingModel.workflow_execution_id == workflow_execution_id,
                CostTrackingModel.agent_id.isnot(None)
            )
            .group_by(CostTrackingModel.agent_id)
        )
        
        result = await self.session.execute(stmt)
        rows = result.all()
        
        return {
            row[0]: float(row[1])
            for row in rows
        }

    async def delete_by_workflow_execution(
        self,
        workflow_execution_id: UUID,
    ) -> int:
        """Delete all cost entries for a workflow execution"""
        stmt = delete(CostTrackingModel).where(
            CostTrackingModel.workflow_execution_id == workflow_execution_id
        )
        
        result = await self.session.execute(stmt)
        await self.session.flush()
        
        return result.rowcount

    def _to_model(self, cost_entry: CostEntry) -> CostTrackingModel:
        """Convert domain entity to ORM model"""
        return CostTrackingModel(
            id=cost_entry.id,
            workflow_execution_id=cost_entry.workflow_execution_id,
            agent_id=cost_entry.agent_id,
            operation_type=OperationTypeEnum(cost_entry.operation_type.value),
            provider=cost_entry.provider,
            model=cost_entry.model,
            tokens_used=cost_entry.tokens_used,
            execution_time_ms=cost_entry.execution_time_ms,
            memory_bytes=cost_entry.memory_bytes,
            cost=Decimal(str(cost_entry.cost)),
            currency=cost_entry.currency,
            task_id=cost_entry.task_id,
            step_name=cost_entry.step_name,
            created_at=cost_entry.created_at,
            request_payload=cost_entry.metadata.get("request_payload"),
            response_metadata=cost_entry.metadata.get("response_metadata"),
            metadata=cost_entry.metadata
        )

    def _to_entity(self, model: CostTrackingModel) -> CostEntry:
        """Convert ORM model to domain entity"""
        metadata = model.metadata or {}
        
        if model.request_payload:
            metadata["request_payload"] = model.request_payload
        
        if model.response_metadata:
            metadata["response_metadata"] = model.response_metadata
        
        return CostEntry(
            id=model.id,
            workflow_execution_id=model.workflow_execution_id,
            agent_id=model.agent_id,
            operation_type=OperationType(model.operation_type.value),
            provider=model.provider or "",
            model=model.model,
            tokens_used=model.tokens_used,
            execution_time_ms=model.execution_time_ms,
            memory_bytes=model.memory_bytes,
            cost=float(model.cost),
            currency=model.currency,
            task_id=model.task_id,
            step_name=model.step_name,
            metadata=metadata,
            created_at=model.created_at
        )
