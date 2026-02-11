# CognitionOS V3 Core Domain

## Clean Architecture - Domain Layer

This directory contains the **core business logic** of CognitionOS, organized into **bounded contexts** following Domain-Driven Design principles.

### Architecture Principles

1. **Zero External Dependencies**: Domain layer depends ONLY on Python standard library
2. **Dependency Rule**: All dependencies point inward toward domain
3. **Rich Domain Model**: Business logic lives in entities, not services
4. **Repository Pattern**: Interfaces defined here, implementations in infrastructure
5. **Domain Events**: Immutable facts representing state changes

### Bounded Contexts

#### 1. Workflow Context (`core/domain/workflow/`)
Manages workflow definitions, executions, and DAG orchestration.

**Key Entities:**
- `Workflow`: Aggregate root for workflow definitions
- `WorkflowExecution`: Aggregate root for execution instances
- `WorkflowStep`, `StepExecution`: Step definitions and executions

**Domain Services:**
- `WorkflowValidator`: Validates DAG structure and templates
- `WorkflowExecutionOrchestrator`: Determines ready steps
- `WorkflowDagAnalyzer`: Analyzes critical path and parallelism

#### 2. Agent Context (`core/domain/agent/`)
Manages AI agents, capabilities, tools, and lifecycle.

**Key Entities:**
- `Agent`: Aggregate root for agent instances
- `AgentDefinition`: Agent blueprints/templates
- `Tool`: Agent capabilities

**Value Objects:**
- `BudgetLimits`, `BudgetUsage`: Resource constraints
- `Capability`, `ModelConfig`: Agent configuration

**Domain Services:**
- `AgentMatcher`: Matches agents to tasks
- `AgentLoadBalancer`: Distributes load across agents
- `AgentHealthMonitor`: Detects stuck/unhealthy agents

#### 3. Memory Context (`core/domain/memory/`)
Manages long-term memory storage, retrieval, and lifecycle.

**Key Entities:**
- `Memory`: Aggregate root for memory entries
- `MemoryCollection`: Groups related memories
- `MemoryLifecyclePolicy`: Defines compression/archival rules

**Value Objects:**
- `Embedding`: Vector embeddings for semantic search
- `MemoryNamespace`: Hierarchical organization

**Domain Services:**
- `MemoryIndexer`: Manages embeddings
- `MemoryDeduplicator`: Detects duplicates
- `MemoryGarbageCollector`: Applies lifecycle policies
- `MemoryRetrieval`: Intelligent retrieval strategies

#### 4. Task Context (`core/domain/task/`)
Manages task planning, assignment, and execution.

**Key Entities:**
- `Task`: Aggregate root for work units

#### 5. Execution Context (`core/domain/execution/`)
Tracks execution history for observability.

**Key Entities:**
- `ExecutionTrace`: Detailed execution tracking

### Usage Example

```python
from core.domain.workflow import (
    Workflow,
    WorkflowId,
    Version,
    WorkflowValidator
)

# Create workflow
workflow = Workflow(
    id=WorkflowId("my-workflow"),
    version=Version.from_string("1.0.0"),
    name="Deploy Application",
    description="CI/CD pipeline",
    steps=[...]
)

# Validate
is_valid, error = WorkflowValidator.validate_dag(workflow)

# Activate
if is_valid:
    workflow.activate()
```

### Repository Pattern

Domain defines **interfaces**, infrastructure provides **implementations**:

```python
# Domain layer (this package)
class WorkflowRepository(ABC):
    @abstractmethod
    async def save(self, workflow: Workflow) -> None:
        pass

# Infrastructure layer (services/workflow-engine/infrastructure/)
class PostgreSQLWorkflowRepository(WorkflowRepository):
    async def save(self, workflow: Workflow) -> None:
        # Implementation using SQLAlchemy
        ...
```

### Domain Events

Events are **immutable facts** about state changes:

```python
from core.domain.workflow.events import WorkflowExecutionStarted

event = WorkflowExecutionStarted(
    occurred_at=datetime.utcnow(),
    event_id=uuid4(),
    execution_id=execution.id,
    workflow_id=workflow.id,
    workflow_version=workflow.version,
    inputs=execution.inputs,
    user_id=user_id
)

# Events are published by application layer
await event_bus.publish(event)
```

### Testing Domain Logic

Domain entities are **pure Python** with no external dependencies, making them trivial to test:

```python
def test_workflow_dag_validation():
    # No mocks needed!
    workflow = Workflow(...)
    assert workflow.has_valid_dag() == True

def test_agent_budget_limits():
    agent = Agent(...)
    agent.record_token_usage(1000, 0.01)
    assert agent.budget_usage.within_limits(agent.budget_limits)
```

### Migration from V2

V2 models (in `services/*/src/models/`) are being migrated to this domain layer:

1. **Extract entities**: Move from Pydantic models to domain entities
2. **Remove dependencies**: Strip out FastAPI, SQLAlchemy dependencies
3. **Add business logic**: Move validation/rules into entities
4. **Define repositories**: Create repository interfaces
5. **Preserve behavior**: Ensure no functional changes

### Next Steps

1. Create **application layer** (`core/application/`) with use cases
2. Implement **infrastructure layer** repositories (PostgreSQL, Redis)
3. Refactor **existing services** to use clean architecture
4. Add **domain tests** for complete coverage

---

**V3 Transformation:** Phase 1B Complete ✅
