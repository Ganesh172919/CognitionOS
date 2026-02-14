# Cost Governance Domain

Phase 3 domain layer for budget management and cost governance.

## Overview

The cost governance domain provides real-time cost tracking with multi-threshold warnings and automated enforcement for workflow budget management.

## Components

### Entities

- **BudgetStatus** - Enum for budget states (active, warning, critical, exhausted, completed)
- **OperationType** - Enum for operation types (llm_call, storage, compute, memory_operation)
- **WorkflowBudget** - Budget tracking entity with configurable thresholds
- **CostEntry** - Granular cost entry for tracking individual operations

### Events

- **BudgetCreated** - Raised when a budget is initialized
- **CostIncurred** - Raised when a cost is recorded
- **BudgetWarningThresholdReached** - Raised at 80% threshold (configurable)
- **BudgetCriticalThresholdReached** - Raised at 95% threshold (configurable)
- **BudgetExhausted** - Raised at 100% threshold
- **BudgetSuspended** - Raised when budget is manually suspended

### Repositories

- **WorkflowBudgetRepository** - Interface for budget persistence
- **CostTrackingRepository** - Interface for cost entry persistence

### Services

- **CostGovernanceService** - Orchestrates budget management and cost tracking

## Usage

```python
from uuid import uuid4
from core.domain.cost_governance import (
    WorkflowBudget,
    CostEntry,
    OperationType,
    CostGovernanceService,
)

# Create a workflow budget
budget = WorkflowBudget.create(
    workflow_execution_id=uuid4(),
    allocated_budget=100.0,
    warning_threshold=0.8,   # 80%
    critical_threshold=0.95,  # 95%
)

# Consume budget
budget.consume_budget(50.0)
status = budget.check_thresholds()  # Returns BudgetStatus.ACTIVE

# Record a cost entry
cost_entry = CostEntry.create(
    workflow_execution_id=budget.workflow_execution_id,
    operation_type=OperationType.LLM_CALL,
    provider="openai",
    model="gpt-4",
    cost=0.05,
    tokens_used=1500,
)

# Using the service (async)
service = CostGovernanceService(budget_repo, cost_repo)
budget, event = await service.create_budget(workflow_id, 100.0)
cost_entry, event = await service.record_cost(
    workflow_id,
    OperationType.LLM_CALL,
    "openai",
    0.05,
    model="gpt-4",
    tokens_used=1500,
)
status, events = await service.check_and_enforce_budget(workflow_id)
```

## Key Features

- **Multi-threshold warnings** - 80%, 95%, 100% thresholds with event-based notifications
- **Real-time tracking** - Immediate budget consumption updates
- **Granular cost entries** - Track costs by operation type, agent, provider, and model
- **Cost breakdown** - Analyze costs by operation type or agent
- **Projected costs** - Calculate projected total cost based on completion percentage
- **Budget suspension** - Manual budget suspension capability
- **Automatic enforcement** - Halt triggering at budget exhaustion

## Design Principles

- Pure domain logic (no infrastructure dependencies)
- Immutable events (frozen dataclasses)
- Repository pattern for persistence abstraction
- Factory methods for entity creation
- Comprehensive validation in `__post_init__`
- Type annotations throughout

## Pattern Compliance

This domain follows the exact same pattern as:
- `core/domain/checkpoint`
- `core/domain/health_monitoring`

All domains use:
- Dataclasses with field defaults
- Factory methods (`.create()`)
- Frozen dataclasses for events
- ABC pattern for repository interfaces
- No external dependencies except Python stdlib
