"""
Cost Governance Application - Use Cases

Application layer use cases for Cost Governance bounded context.
Orchestrates domain entities and coordinates with infrastructure.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import UUID

from core.domain.cost_governance import (
    WorkflowBudget,
    CostEntry,
    BudgetStatus,
    OperationType,
    WorkflowBudgetRepository,
    CostTrackingRepository,
    CostGovernanceService,
    BudgetCreated,
    CostIncurred,
    BudgetWarningThresholdReached,
    BudgetCriticalThresholdReached,
    BudgetExhausted,
    BudgetSuspended,
)


# ==================== DTOs (Data Transfer Objects) ====================

@dataclass
class CreateBudgetCommand:
    """Command to create budget for workflow"""
    workflow_execution_id: UUID
    allocated_budget: float
    currency: str = "USD"
    warning_threshold: float = 0.8
    critical_threshold: float = 0.95


@dataclass
class RecordCostCommand:
    """Command to record cost entry with threshold checking"""
    workflow_execution_id: UUID
    operation_type: str
    provider: str
    cost: float
    agent_id: Optional[UUID] = None
    model: Optional[str] = None
    tokens_used: Optional[int] = None
    execution_time_ms: Optional[int] = None
    memory_bytes: Optional[int] = None
    task_id: Optional[UUID] = None
    step_name: Optional[str] = None
    currency: str = "USD"


@dataclass
class CostSummaryQuery:
    """Query to get cost summary for workflow"""
    workflow_execution_id: UUID


@dataclass
class EnforceBudgetLimitsCommand:
    """Command to enforce budget limits and suspend if needed"""
    workflow_execution_id: UUID


@dataclass
class BudgetResult:
    """Result of budget operation"""
    budget_id: UUID
    workflow_execution_id: UUID
    allocated_budget: float
    consumed_budget: float
    remaining_budget: float
    currency: str
    status: BudgetStatus
    usage_percentage: float
    warning_threshold: float
    critical_threshold: float
    is_exhausted: bool
    is_suspended: bool
    created_at: str


@dataclass
class CostEntryResult:
    """Result of cost entry operation"""
    cost_entry_id: UUID
    workflow_execution_id: UUID
    operation_type: OperationType
    provider: str
    cost: float
    agent_id: Optional[UUID] = None
    model: Optional[str] = None
    tokens_used: Optional[int] = None
    consumed_budget: float = 0.0
    remaining_budget: float = 0.0
    usage_percentage: float = 0.0
    created_at: str = None


@dataclass
class CostSummaryResult:
    """Result of cost summary query"""
    workflow_execution_id: UUID
    total_cost: float
    budget_exists: bool
    allocated_budget: Optional[float] = None
    consumed_budget: Optional[float] = None
    remaining_budget: Optional[float] = None
    usage_percentage: Optional[float] = None
    status: Optional[str] = None
    currency: Optional[str] = None
    is_exhausted: Optional[bool] = None
    is_suspended: Optional[bool] = None
    cost_by_operation_type: Optional[Dict[str, float]] = None
    cost_by_agent: Optional[Dict[str, float]] = None


# ==================== Use Cases ====================

class CreateWorkflowBudgetUseCase:
    """
    Use Case: Create budget for workflow.

    Orchestrates:
    1. Create budget via domain service
    2. Validate budget parameters
    3. Publish domain event
    """

    def __init__(
        self,
        cost_governance_service: CostGovernanceService,
        event_publisher: Optional[Any] = None
    ):
        self.cost_governance_service = cost_governance_service
        self.event_publisher = event_publisher

    async def execute(self, command: CreateBudgetCommand) -> BudgetResult:
        """
        Create workflow budget.

        Args:
            command: Create budget command

        Returns:
            BudgetResult with created budget details

        Raises:
            ValueError: If budget creation fails
        """
        # Create budget via domain service
        budget, event = await self.cost_governance_service.create_budget(
            workflow_execution_id=command.workflow_execution_id,
            allocated_budget=command.allocated_budget,
            currency=command.currency,
            warning_threshold=command.warning_threshold,
            critical_threshold=command.critical_threshold,
        )

        # Publish event
        if self.event_publisher:
            await self.event_publisher.publish(event)

        return BudgetResult(
            budget_id=budget.id,
            workflow_execution_id=budget.workflow_execution_id,
            allocated_budget=budget.allocated_budget,
            consumed_budget=budget.consumed_budget,
            remaining_budget=budget.get_remaining_budget(),
            currency=budget.currency,
            status=budget.status,
            usage_percentage=budget.get_usage_percentage(),
            warning_threshold=budget.warning_threshold,
            critical_threshold=budget.critical_threshold,
            is_exhausted=budget.is_exhausted(),
            is_suspended=budget.is_suspended(),
            created_at=budget.created_at.isoformat(),
        )


class RecordCostUseCase:
    """
    Use Case: Record cost entry with threshold checking.

    Orchestrates:
    1. Record cost via domain service
    2. Update budget consumption
    3. Check thresholds
    4. Publish domain events
    """

    def __init__(
        self,
        cost_governance_service: CostGovernanceService,
        event_publisher: Optional[Any] = None
    ):
        self.cost_governance_service = cost_governance_service
        self.event_publisher = event_publisher

    async def execute(self, command: RecordCostCommand) -> CostEntryResult:
        """
        Record cost entry.

        Args:
            command: Record cost command

        Returns:
            CostEntryResult with cost entry details

        Raises:
            ValueError: If cost recording fails
        """
        # Parse operation type
        operation_type = OperationType(command.operation_type)

        # Record cost via domain service
        cost_entry, event = await self.cost_governance_service.record_cost(
            workflow_execution_id=command.workflow_execution_id,
            operation_type=operation_type,
            provider=command.provider,
            cost=command.cost,
            agent_id=command.agent_id,
            model=command.model,
            tokens_used=command.tokens_used,
            execution_time_ms=command.execution_time_ms,
            memory_bytes=command.memory_bytes,
            task_id=command.task_id,
            step_name=command.step_name,
            currency=command.currency,
        )

        # Publish cost incurred event
        if self.event_publisher:
            await self.event_publisher.publish(event)

        # Check and enforce budget thresholds
        try:
            status, threshold_events = await self.cost_governance_service.check_and_enforce_budget(
                workflow_execution_id=command.workflow_execution_id
            )
            
            # Publish threshold events
            if self.event_publisher:
                for threshold_event in threshold_events:
                    await self.event_publisher.publish(threshold_event)
        except ValueError:
            # Budget doesn't exist - that's OK for cost tracking
            pass

        return CostEntryResult(
            cost_entry_id=cost_entry.id,
            workflow_execution_id=cost_entry.workflow_execution_id,
            operation_type=cost_entry.operation_type,
            provider=cost_entry.provider,
            cost=cost_entry.cost,
            agent_id=cost_entry.agent_id,
            model=cost_entry.model,
            tokens_used=cost_entry.tokens_used,
            consumed_budget=event.consumed_budget,
            remaining_budget=event.remaining_budget,
            usage_percentage=event.usage_percentage,
            created_at=cost_entry.created_at.isoformat(),
        )


class GetCostSummaryUseCase:
    """
    Use Case: Get cost summary for workflow.

    Retrieves comprehensive cost summary including budget status and breakdowns.
    """

    def __init__(
        self,
        cost_governance_service: CostGovernanceService
    ):
        self.cost_governance_service = cost_governance_service

    async def execute(self, query: CostSummaryQuery) -> CostSummaryResult:
        """
        Get cost summary.

        Args:
            query: Cost summary query

        Returns:
            CostSummaryResult with comprehensive cost data
        """
        # Get cost summary via domain service
        summary = await self.cost_governance_service.get_cost_summary(
            workflow_execution_id=query.workflow_execution_id
        )

        # Get cost breakdowns
        cost_by_operation_type = await self.cost_governance_service.get_cost_breakdown_by_operation_type(
            workflow_execution_id=query.workflow_execution_id
        )
        
        cost_by_agent = await self.cost_governance_service.get_cost_breakdown_by_agent(
            workflow_execution_id=query.workflow_execution_id
        )

        # Build result
        result = CostSummaryResult(
            workflow_execution_id=query.workflow_execution_id,
            total_cost=summary["total_cost"],
            budget_exists=summary["budget_exists"],
            cost_by_operation_type=cost_by_operation_type,
            cost_by_agent=cost_by_agent,
        )

        # Add budget details if exists
        if summary["budget_exists"]:
            result.allocated_budget = summary.get("allocated_budget")
            result.consumed_budget = summary.get("consumed_budget")
            result.remaining_budget = summary.get("remaining_budget")
            result.usage_percentage = summary.get("usage_percentage")
            result.status = summary.get("status")
            result.currency = summary.get("currency")
            result.is_exhausted = summary.get("is_exhausted")
            result.is_suspended = summary.get("is_suspended")

        return result


class EnforceBudgetLimitsUseCase:
    """
    Use Case: Enforce budget limits and suspend if needed.

    Orchestrates:
    1. Check budget thresholds
    2. Suspend workflow if budget exhausted
    3. Publish domain events
    """

    def __init__(
        self,
        cost_governance_service: CostGovernanceService,
        event_publisher: Optional[Any] = None
    ):
        self.cost_governance_service = cost_governance_service
        self.event_publisher = event_publisher

    async def execute(self, command: EnforceBudgetLimitsCommand) -> Dict[str, Any]:
        """
        Enforce budget limits.

        Args:
            command: Enforce budget limits command

        Returns:
            Dict with enforcement result details

        Raises:
            ValueError: If budget not found
        """
        # Check and enforce budget
        status, events = await self.cost_governance_service.check_and_enforce_budget(
            workflow_execution_id=command.workflow_execution_id
        )

        # Publish threshold events
        if self.event_publisher:
            for event in events:
                await self.event_publisher.publish(event)

        # Suspend if exhausted
        suspended = False
        if status == BudgetStatus.EXHAUSTED:
            try:
                budget, suspend_event = await self.cost_governance_service.suspend_budget(
                    workflow_execution_id=command.workflow_execution_id,
                    reason="Budget exhausted - automatic suspension"
                )
                suspended = True
                
                # Publish suspension event
                if self.event_publisher:
                    await self.event_publisher.publish(suspend_event)
            except ValueError:
                # Already suspended or error - that's OK
                pass

        return {
            "status": status.value,
            "events_raised": len(events),
            "suspended": suspended,
            "should_halt": status == BudgetStatus.EXHAUSTED,
        }
