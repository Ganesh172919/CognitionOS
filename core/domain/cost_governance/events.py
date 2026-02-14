"""
Cost Governance Domain - Events

Domain events for cost governance operations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from .entities import BudgetStatus, OperationType


@dataclass(frozen=True)
class CostGovernanceEvent:
    """Base cost governance domain event"""
    workflow_execution_id: UUID
    occurred_at: datetime
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class BudgetCreated(CostGovernanceEvent):
    """Event raised when a budget is created"""
    budget_id: UUID
    allocated_budget: float
    currency: str
    warning_threshold: float
    critical_threshold: float

    @classmethod
    def create(
        cls,
        budget_id: UUID,
        workflow_execution_id: UUID,
        allocated_budget: float,
        currency: str,
        warning_threshold: float,
        critical_threshold: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BudgetCreated":
        """Factory method to create event"""
        return cls(
            budget_id=budget_id,
            workflow_execution_id=workflow_execution_id,
            allocated_budget=allocated_budget,
            currency=currency,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            occurred_at=datetime.utcnow(),
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class CostIncurred(CostGovernanceEvent):
    """Event raised when a cost is incurred"""
    cost_entry_id: UUID
    operation_type: OperationType
    cost: float
    consumed_budget: float
    remaining_budget: float
    usage_percentage: float

    @classmethod
    def create(
        cls,
        cost_entry_id: UUID,
        workflow_execution_id: UUID,
        operation_type: OperationType,
        cost: float,
        consumed_budget: float,
        remaining_budget: float,
        usage_percentage: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CostIncurred":
        """Factory method to create event"""
        return cls(
            cost_entry_id=cost_entry_id,
            workflow_execution_id=workflow_execution_id,
            operation_type=operation_type,
            cost=cost,
            consumed_budget=consumed_budget,
            remaining_budget=remaining_budget,
            usage_percentage=usage_percentage,
            occurred_at=datetime.utcnow(),
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class BudgetWarningThresholdReached(CostGovernanceEvent):
    """Event raised when budget warning threshold is reached (default 80%)"""
    budget_id: UUID
    threshold_percentage: float
    consumed_budget: float
    allocated_budget: float
    usage_percentage: float

    @classmethod
    def create(
        cls,
        budget_id: UUID,
        workflow_execution_id: UUID,
        threshold_percentage: float,
        consumed_budget: float,
        allocated_budget: float,
        usage_percentage: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BudgetWarningThresholdReached":
        """Factory method to create event"""
        return cls(
            budget_id=budget_id,
            workflow_execution_id=workflow_execution_id,
            threshold_percentage=threshold_percentage,
            consumed_budget=consumed_budget,
            allocated_budget=allocated_budget,
            usage_percentage=usage_percentage,
            occurred_at=datetime.utcnow(),
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class BudgetCriticalThresholdReached(CostGovernanceEvent):
    """Event raised when budget critical threshold is reached (default 95%)"""
    budget_id: UUID
    threshold_percentage: float
    consumed_budget: float
    allocated_budget: float
    usage_percentage: float

    @classmethod
    def create(
        cls,
        budget_id: UUID,
        workflow_execution_id: UUID,
        threshold_percentage: float,
        consumed_budget: float,
        allocated_budget: float,
        usage_percentage: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BudgetCriticalThresholdReached":
        """Factory method to create event"""
        return cls(
            budget_id=budget_id,
            workflow_execution_id=workflow_execution_id,
            threshold_percentage=threshold_percentage,
            consumed_budget=consumed_budget,
            allocated_budget=allocated_budget,
            usage_percentage=usage_percentage,
            occurred_at=datetime.utcnow(),
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class BudgetExhausted(CostGovernanceEvent):
    """Event raised when budget is exhausted (100%)"""
    budget_id: UUID
    consumed_budget: float
    allocated_budget: float
    halt_triggered: bool

    @classmethod
    def create(
        cls,
        budget_id: UUID,
        workflow_execution_id: UUID,
        consumed_budget: float,
        allocated_budget: float,
        halt_triggered: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BudgetExhausted":
        """Factory method to create event"""
        return cls(
            budget_id=budget_id,
            workflow_execution_id=workflow_execution_id,
            consumed_budget=consumed_budget,
            allocated_budget=allocated_budget,
            halt_triggered=halt_triggered,
            occurred_at=datetime.utcnow(),
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class BudgetSuspended(CostGovernanceEvent):
    """Event raised when budget is suspended"""
    budget_id: UUID
    suspension_reason: str
    consumed_budget: float
    allocated_budget: float

    @classmethod
    def create(
        cls,
        budget_id: UUID,
        workflow_execution_id: UUID,
        suspension_reason: str,
        consumed_budget: float,
        allocated_budget: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BudgetSuspended":
        """Factory method to create event"""
        return cls(
            budget_id=budget_id,
            workflow_execution_id=workflow_execution_id,
            suspension_reason=suspension_reason,
            consumed_budget=consumed_budget,
            allocated_budget=allocated_budget,
            occurred_at=datetime.utcnow(),
            metadata=metadata or {},
        )
