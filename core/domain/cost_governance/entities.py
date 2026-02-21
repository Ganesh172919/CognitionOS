"""
Cost Governance Domain - Entities

Pure domain entities for cost governance functionality.
NO external dependencies except Python stdlib.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4


class BudgetStatus(str, Enum):
    """Budget status states"""
    ACTIVE = "active"
    WARNING = "warning"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"
    COMPLETED = "completed"
    SUSPENDED = "suspended"


class OperationType(str, Enum):
    """Types of operations that incur costs"""
    LLM_CALL = "llm_call"
    STORAGE = "storage"
    COMPUTE = "compute"
    MEMORY_OPERATION = "memory_operation"
    AGENT_EXECUTION = "agent_execution"
    API_CALL = "api_call"


# ==================== Entities ====================

@dataclass
class WorkflowBudget:
    """
    Workflow budget entity for cost governance.
    
    Tracks budget allocation and consumption for workflow executions.
    Design principles:
    - Real-time cost tracking
    - Multi-threshold warnings (80%, 95%, 100%)
    - Automated enforcement
    """
    id: UUID
    workflow_execution_id: UUID
    allocated_budget: float
    consumed_budget: float
    currency: str
    
    # Threshold configuration
    warning_threshold: float = 0.8
    critical_threshold: float = 0.95
    
    # Status
    status: BudgetStatus = BudgetStatus.ACTIVE
    
    # Tracking
    warnings_sent: int = 0
    halt_triggered_at: Optional[datetime] = None
    suspended_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate budget invariants"""
        if self.allocated_budget < 0:
            raise ValueError("Allocated budget cannot be negative")
        
        if self.consumed_budget < 0:
            raise ValueError("Consumed budget cannot be negative")
        
        if self.warning_threshold <= 0 or self.warning_threshold > 1:
            raise ValueError("Warning threshold must be between 0 and 1")
        
        if self.critical_threshold <= 0 or self.critical_threshold > 1:
            raise ValueError("Critical threshold must be between 0 and 1")
        
        if self.warning_threshold >= self.critical_threshold:
            raise ValueError("Warning threshold must be less than critical threshold")

    def consume_budget(self, amount: float) -> None:
        """
        Consume budget amount.
        
        Args:
            amount: Amount to consume
            
        Raises:
            ValueError: If amount is negative
        """
        if amount < 0:
            raise ValueError("Consumption amount cannot be negative")
        
        self.consumed_budget += amount
        self.updated_at = datetime.utcnow()

    def check_thresholds(self) -> BudgetStatus:
        """
        Check budget thresholds and update status.
        
        Returns:
            Current budget status
        """
        usage_percentage = self.get_usage_percentage()
        
        if usage_percentage >= 100:
            self.status = BudgetStatus.EXHAUSTED
            if not self.halt_triggered_at:
                self.halt_triggered_at = datetime.utcnow()
        elif usage_percentage >= self.critical_threshold * 100:
            self.status = BudgetStatus.CRITICAL
        elif usage_percentage >= self.warning_threshold * 100:
            self.status = BudgetStatus.WARNING
        else:
            self.status = BudgetStatus.ACTIVE
        
        self.updated_at = datetime.utcnow()
        return self.status

    def is_exhausted(self) -> bool:
        """
        Check if budget is exhausted.
        
        Returns:
            True if budget is exhausted
        """
        return self.consumed_budget >= self.allocated_budget

    def get_usage_percentage(self) -> float:
        """
        Get budget usage percentage.
        
        Returns:
            Usage percentage (0-100+)
        """
        if self.allocated_budget == 0:
            return 0.0
        return (self.consumed_budget / self.allocated_budget) * 100

    def get_remaining_budget(self) -> float:
        """
        Get remaining budget.
        
        Returns:
            Remaining budget amount
        """
        return max(0.0, self.allocated_budget - self.consumed_budget)

    def suspend(self) -> None:
        """Suspend the budget"""
        self.status = BudgetStatus.SUSPENDED
        self.suspended_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def is_suspended(self) -> bool:
        """Check if budget is suspended"""
        return self.suspended_at is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert budget to dictionary for persistence"""
        return {
            "id": str(self.id),
            "workflow_execution_id": str(self.workflow_execution_id),
            "allocated_budget": self.allocated_budget,
            "consumed_budget": self.consumed_budget,
            "currency": self.currency,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "status": self.status.value,
            "warnings_sent": self.warnings_sent,
            "halt_triggered_at": self.halt_triggered_at.isoformat() if self.halt_triggered_at else None,
            "suspended_at": self.suspended_at.isoformat() if self.suspended_at else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowBudget":
        """Create budget from dictionary"""
        return cls(
            id=UUID(data["id"]),
            workflow_execution_id=UUID(data["workflow_execution_id"]),
            allocated_budget=data["allocated_budget"],
            consumed_budget=data["consumed_budget"],
            currency=data.get("currency", "USD"),
            warning_threshold=data.get("warning_threshold", 0.8),
            critical_threshold=data.get("critical_threshold", 0.95),
            status=BudgetStatus(data.get("status", "active")),
            warnings_sent=data.get("warnings_sent", 0),
            halt_triggered_at=datetime.fromisoformat(data["halt_triggered_at"]) if data.get("halt_triggered_at") else None,
            suspended_at=datetime.fromisoformat(data["suspended_at"]) if data.get("suspended_at") else None,
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

    @classmethod
    def create(
        cls,
        workflow_execution_id: UUID,
        allocated_budget: float,
        currency: str = "USD",
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95,
    ) -> "WorkflowBudget":
        """Factory method to create a new workflow budget"""
        return cls(
            id=uuid4(),
            workflow_execution_id=workflow_execution_id,
            allocated_budget=allocated_budget,
            consumed_budget=0.0,
            currency=currency,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
        )


@dataclass
class CostEntry:
    """
    Cost entry entity for granular cost tracking.
    
    Records individual operations and their associated costs.
    """
    id: UUID
    workflow_execution_id: UUID
    agent_id: Optional[UUID]
    operation_type: OperationType
    provider: str
    model: Optional[str]
    
    # Usage metrics
    tokens_used: Optional[int] = None
    execution_time_ms: Optional[int] = None
    memory_bytes: Optional[int] = None
    
    # Cost
    cost: float = 0.0
    currency: str = "USD"
    
    # Task context
    task_id: Optional[UUID] = None
    step_name: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate cost entry invariants"""
        if self.cost < 0:
            raise ValueError("Cost cannot be negative")
        
        if self.tokens_used is not None and self.tokens_used < 0:
            raise ValueError("Tokens used cannot be negative")
        
        if self.execution_time_ms is not None and self.execution_time_ms < 0:
            raise ValueError("Execution time cannot be negative")
        
        if self.memory_bytes is not None and self.memory_bytes < 0:
            raise ValueError("Memory bytes cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        """Convert cost entry to dictionary for persistence"""
        return {
            "id": str(self.id),
            "workflow_execution_id": str(self.workflow_execution_id),
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "operation_type": self.operation_type.value,
            "provider": self.provider,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "execution_time_ms": self.execution_time_ms,
            "memory_bytes": self.memory_bytes,
            "cost": self.cost,
            "currency": self.currency,
            "task_id": str(self.task_id) if self.task_id else None,
            "step_name": self.step_name,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostEntry":
        """Create cost entry from dictionary"""
        return cls(
            id=UUID(data["id"]),
            workflow_execution_id=UUID(data["workflow_execution_id"]),
            agent_id=UUID(data["agent_id"]) if data.get("agent_id") else None,
            operation_type=OperationType(data["operation_type"]),
            provider=data["provider"],
            model=data.get("model"),
            tokens_used=data.get("tokens_used"),
            execution_time_ms=data.get("execution_time_ms"),
            memory_bytes=data.get("memory_bytes"),
            cost=data.get("cost", 0.0),
            currency=data.get("currency", "USD"),
            task_id=UUID(data["task_id"]) if data.get("task_id") else None,
            step_name=data.get("step_name"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
        )

    @classmethod
    def create(
        cls,
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
    ) -> "CostEntry":
        """Factory method to create a new cost entry"""
        return cls(
            id=uuid4(),
            workflow_execution_id=workflow_execution_id,
            agent_id=agent_id,
            operation_type=operation_type,
            provider=provider,
            model=model,
            tokens_used=tokens_used,
            execution_time_ms=execution_time_ms,
            memory_bytes=memory_bytes,
            cost=cost,
            currency=currency,
            task_id=task_id,
            step_name=step_name,
        )
