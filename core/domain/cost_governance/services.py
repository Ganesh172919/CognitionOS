"""
Cost Governance Domain - Services

Domain services for cost governance orchestration and business logic.
"""

from typing import Dict, List, Optional
from uuid import UUID

from .entities import (
    BudgetStatus,
    CostEntry,
    OperationType,
    WorkflowBudget,
)
from .events import (
    BudgetCreated,
    BudgetCriticalThresholdReached,
    BudgetExhausted,
    BudgetSuspended,
    BudgetWarningThresholdReached,
    CostIncurred,
)
from .repositories import CostTrackingRepository, WorkflowBudgetRepository


class CostGovernanceService:
    """
    Domain service for cost governance operations.
    
    Orchestrates budget management, cost tracking, and enforcement.
    """

    def __init__(
        self,
        budget_repository: WorkflowBudgetRepository,
        cost_tracking_repository: CostTrackingRepository,
    ):
        """
        Initialize cost governance service.
        
        Args:
            budget_repository: Budget repository
            cost_tracking_repository: Cost tracking repository
        """
        self.budget_repository = budget_repository
        self.cost_tracking_repository = cost_tracking_repository

    async def create_budget(
        self,
        workflow_execution_id: UUID,
        allocated_budget: float,
        currency: str = "USD",
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95,
    ) -> tuple[WorkflowBudget, BudgetCreated]:
        """
        Create a new workflow budget.
        
        Args:
            workflow_execution_id: Workflow execution ID
            allocated_budget: Total budget allocation
            currency: Currency code (default: USD)
            warning_threshold: Warning threshold (0-1, default: 0.8)
            critical_threshold: Critical threshold (0-1, default: 0.95)
            
        Returns:
            Tuple of (created budget, budget created event)
            
        Raises:
            ValueError: If budget already exists for workflow execution
        """
        # Check if budget already exists
        existing = await self.budget_repository.find_by_workflow_execution(
            workflow_execution_id
        )
        if existing:
            raise ValueError(
                f"Budget already exists for workflow execution: {workflow_execution_id}"
            )

        # Create budget
        budget = WorkflowBudget.create(
            workflow_execution_id=workflow_execution_id,
            allocated_budget=allocated_budget,
            currency=currency,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
        )

        # Save budget
        await self.budget_repository.save(budget)

        # Create event
        event = BudgetCreated.create(
            budget_id=budget.id,
            workflow_execution_id=workflow_execution_id,
            allocated_budget=allocated_budget,
            currency=currency,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
        )

        return budget, event

    async def record_cost(
        self,
        workflow_execution_id: UUID,
        operation_type: OperationType,
        provider: str,
        cost: float,
        agent_id: Optional[UUID] = None,
        model: Optional[str] = None,
        tokens_used: Optional[int] = None,
        execution_time_ms: Optional[int] = None,
        memory_bytes: Optional[int] = None,
        task_id: Optional[UUID] = None,
        step_name: Optional[str] = None,
        currency: str = "USD",
    ) -> tuple[CostEntry, CostIncurred]:
        """
        Record a cost entry.
        
        Args:
            workflow_execution_id: Workflow execution ID
            operation_type: Type of operation
            provider: Service provider
            cost: Cost amount
            agent_id: Agent ID (optional)
            model: Model name (optional)
            tokens_used: Token count (optional)
            execution_time_ms: Execution time in ms (optional)
            memory_bytes: Memory usage in bytes (optional)
            task_id: Task ID (optional)
            step_name: Step name (optional)
            currency: Currency code (default: USD)
            
        Returns:
            Tuple of (cost entry, cost incurred event)
        """
        # Create cost entry
        cost_entry = CostEntry.create(
            workflow_execution_id=workflow_execution_id,
            operation_type=operation_type,
            provider=provider,
            cost=cost,
            agent_id=agent_id,
            model=model,
            tokens_used=tokens_used,
            execution_time_ms=execution_time_ms,
            memory_bytes=memory_bytes,
            task_id=task_id,
            step_name=step_name,
            currency=currency,
        )

        # Save cost entry
        await self.cost_tracking_repository.save(cost_entry)

        # Update budget if exists
        budget = await self.budget_repository.find_by_workflow_execution(
            workflow_execution_id
        )
        
        consumed_budget = cost
        remaining_budget = 0.0
        usage_percentage = 0.0
        
        if budget:
            budget.consume_budget(cost)
            await self.budget_repository.update(budget)
            consumed_budget = budget.consumed_budget
            remaining_budget = budget.get_remaining_budget()
            usage_percentage = budget.get_usage_percentage()

        # Create event
        event = CostIncurred.create(
            cost_entry_id=cost_entry.id,
            workflow_execution_id=workflow_execution_id,
            operation_type=operation_type,
            cost=cost,
            consumed_budget=consumed_budget,
            remaining_budget=remaining_budget,
            usage_percentage=usage_percentage,
        )

        return cost_entry, event

    async def check_and_enforce_budget(
        self,
        workflow_execution_id: UUID,
    ) -> tuple[BudgetStatus, List[object]]:
        """
        Check budget thresholds and enforce limits.
        
        Checks for 80%, 95%, and 100% thresholds and raises appropriate events.
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            Tuple of (budget status, list of events raised)
            
        Raises:
            ValueError: If budget not found
        """
        budget = await self.budget_repository.find_by_workflow_execution(
            workflow_execution_id
        )
        if not budget:
            raise ValueError(
                f"Budget not found for workflow execution: {workflow_execution_id}"
            )

        # Store previous status
        previous_status = budget.status

        # Check thresholds
        new_status = budget.check_thresholds()
        await self.budget_repository.update(budget)

        events = []

        # Generate events based on status transitions
        usage_percentage = budget.get_usage_percentage()

        if new_status == BudgetStatus.EXHAUSTED and previous_status != BudgetStatus.EXHAUSTED:
            event = BudgetExhausted.create(
                budget_id=budget.id,
                workflow_execution_id=workflow_execution_id,
                consumed_budget=budget.consumed_budget,
                allocated_budget=budget.allocated_budget,
                halt_triggered=True,
            )
            events.append(event)

        if new_status == BudgetStatus.CRITICAL and previous_status not in [
            BudgetStatus.CRITICAL,
            BudgetStatus.EXHAUSTED,
        ]:
            event = BudgetCriticalThresholdReached.create(
                budget_id=budget.id,
                workflow_execution_id=workflow_execution_id,
                threshold_percentage=budget.critical_threshold * 100,
                consumed_budget=budget.consumed_budget,
                allocated_budget=budget.allocated_budget,
                usage_percentage=usage_percentage,
            )
            events.append(event)

        if new_status == BudgetStatus.WARNING and previous_status == BudgetStatus.ACTIVE:
            event = BudgetWarningThresholdReached.create(
                budget_id=budget.id,
                workflow_execution_id=workflow_execution_id,
                threshold_percentage=budget.warning_threshold * 100,
                consumed_budget=budget.consumed_budget,
                allocated_budget=budget.allocated_budget,
                usage_percentage=usage_percentage,
            )
            budget.warnings_sent += 1
            await self.budget_repository.update(budget)
            events.append(event)

        return new_status, events

    async def suspend_budget(
        self,
        workflow_execution_id: UUID,
        reason: str,
    ) -> tuple[WorkflowBudget, BudgetSuspended]:
        """
        Suspend a budget.
        
        Args:
            workflow_execution_id: Workflow execution ID
            reason: Suspension reason
            
        Returns:
            Tuple of (suspended budget, suspension event)
            
        Raises:
            ValueError: If budget not found
        """
        budget = await self.budget_repository.find_by_workflow_execution(
            workflow_execution_id
        )
        if not budget:
            raise ValueError(
                f"Budget not found for workflow execution: {workflow_execution_id}"
            )

        budget.suspend()
        await self.budget_repository.update(budget)

        event = BudgetSuspended.create(
            budget_id=budget.id,
            workflow_execution_id=workflow_execution_id,
            suspension_reason=reason,
            consumed_budget=budget.consumed_budget,
            allocated_budget=budget.allocated_budget,
        )

        return budget, event

    async def get_cost_summary(
        self,
        workflow_execution_id: UUID,
    ) -> Dict[str, any]:
        """
        Get cost summary for a workflow execution.
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            Cost summary dictionary
        """
        budget = await self.budget_repository.find_by_workflow_execution(
            workflow_execution_id
        )
        total_cost = await self.cost_tracking_repository.get_total_cost(
            workflow_execution_id
        )

        summary = {
            "workflow_execution_id": str(workflow_execution_id),
            "total_cost": total_cost,
            "budget_exists": budget is not None,
        }

        if budget:
            summary.update({
                "allocated_budget": budget.allocated_budget,
                "consumed_budget": budget.consumed_budget,
                "remaining_budget": budget.get_remaining_budget(),
                "usage_percentage": budget.get_usage_percentage(),
                "status": budget.status.value,
                "currency": budget.currency,
                "warning_threshold": budget.warning_threshold,
                "critical_threshold": budget.critical_threshold,
                "is_exhausted": budget.is_exhausted(),
                "is_suspended": budget.is_suspended(),
            })

        return summary

    async def calculate_projected_cost(
        self,
        workflow_execution_id: UUID,
        completion_percentage: float,
    ) -> float:
        """
        Calculate projected total cost based on current spending rate.
        
        Args:
            workflow_execution_id: Workflow execution ID
            completion_percentage: Current completion percentage (0-100)
            
        Returns:
            Projected total cost
        """
        if completion_percentage <= 0 or completion_percentage > 100:
            raise ValueError("Completion percentage must be between 0 and 100")

        total_cost = await self.cost_tracking_repository.get_total_cost(
            workflow_execution_id
        )

        if completion_percentage == 0:
            return 0.0

        # Linear projection: (current_cost / completion%) * 100
        projected_cost = (total_cost / completion_percentage) * 100

        return projected_cost

    async def get_cost_breakdown_by_operation_type(
        self,
        workflow_execution_id: UUID,
    ) -> Dict[str, float]:
        """
        Get cost breakdown by operation type.
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            Dictionary mapping operation type to cost
        """
        breakdown = await self.cost_tracking_repository.get_cost_by_operation_type(
            workflow_execution_id
        )

        # Convert enum keys to strings for serialization
        return {op_type.value: cost for op_type, cost in breakdown.items()}

    async def get_cost_breakdown_by_agent(
        self,
        workflow_execution_id: UUID,
    ) -> Dict[str, float]:
        """
        Get cost breakdown by agent.
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            Dictionary mapping agent ID to cost
        """
        breakdown = await self.cost_tracking_repository.get_cost_by_agent(
            workflow_execution_id
        )

        # Convert UUID keys to strings for serialization
        return {str(agent_id): cost for agent_id, cost in breakdown.items()}

    async def get_budget_status(
        self,
        workflow_execution_id: UUID,
    ) -> Optional[BudgetStatus]:
        """
        Get current budget status.
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            Budget status if budget exists, None otherwise
        """
        budget = await self.budget_repository.find_by_workflow_execution(
            workflow_execution_id
        )
        return budget.status if budget else None
