# CognitionOS V3: Dependency Graph & Architecture Layers

## Overview

This document defines the dependency relationships between architecture layers and components in CognitionOS V3. Following Clean Architecture principles, all dependencies point **inward** toward the domain layer.

## Dependency Rule

> **The Dependency Rule**: Source code dependencies must only point inward, toward higher-level policies.

- **Domain Layer** has ZERO dependencies on other layers
- **Application Layer** depends ONLY on Domain Layer
- **Infrastructure Layer** depends on Domain + Application Layers
- **Interface Layer** depends on Application Layer (through use cases)

## Layer Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                     Interface Layer                          │
│  (Controllers, CLI, gRPC, GraphQL, WebSocket handlers)       │
│                           │                                   │
│                           ▼                                   │
├─────────────────────────────────────────────────────────────┤
│                    Application Layer                         │
│         (Use Cases, DTOs, Application Services)              │
│                           │                                   │
│                           ▼                                   │
├─────────────────────────────────────────────────────────────┤
│                      Domain Layer                            │
│    (Entities, Value Objects, Domain Services, Events)        │
│                     ZERO DEPENDENCIES                        │
│                           ▲                                   │
│                           │                                   │
├─────────────────────────────────────────────────────────────┤
│                  Infrastructure Layer                        │
│  (DB, Redis, Message Queue, External APIs, File System)      │
└─────────────────────────────────────────────────────────────┘
```

## Bounded Context Dependencies

### 1. Workflow Context

```
workflow-api (Interface)
    │
    ├──> ExecuteWorkflowUseCase (Application)
    │       │
    │       ├──> Workflow (Domain Entity)
    │       ├──> WorkflowRepository (Domain Interface)
    │       └──> WorkflowExecutor (Domain Service)
    │
    └──> PostgreSQLWorkflowRepository (Infrastructure)
            └──> Implements WorkflowRepository
```

**Dependencies**:
- Workflow Context → Task Context (creates tasks)
- Workflow Context → Agent Context (assigns agents)
- Workflow Context → Execution Context (tracks executions)

### 2. Agent Context

```
agent-api (Interface)
    │
    ├──> AssignAgentUseCase (Application)
    │       │
    │       ├──> Agent (Domain Entity)
    │       ├──> AgentRepository (Domain Interface)
    │       └──> AgentMatcher (Domain Service)
    │
    └──> PostgreSQLAgentRepository (Infrastructure)
            └──> Implements AgentRepository
```

**Dependencies**:
- Agent Context → Task Context (accepts tasks)
- Agent Context → Memory Context (reads/writes memories)
- Agent Context is INDEPENDENT of Workflow Context

### 3. Memory Context

```
memory-api (Interface)
    │
    ├──> StoreMemoryUseCase (Application)
    │       │
    │       ├──> Memory (Domain Entity)
    │       ├──> MemoryRepository (Domain Interface)
    │       └──> MemoryIndexer (Domain Service)
    │
    └──> PgVectorMemoryRepository (Infrastructure)
            └──> Implements MemoryRepository
```

**Dependencies**:
- Memory Context is COMPLETELY INDEPENDENT
- Other contexts use it via application layer

### 4. Task Context

```
task-api (Interface)
    │
    ├──> CreateTaskUseCase (Application)
    │       │
    │       ├──> Task (Domain Entity)
    │       ├──> TaskRepository (Domain Interface)
    │       └──> TaskValidator (Domain Service)
    │
    └──> PostgreSQLTaskRepository (Infrastructure)
            └──> Implements TaskRepository
```

**Dependencies**:
- Task Context → Agent Context (for assignment)
- Task Context is INDEPENDENT of Workflow/Execution

### 5. Execution Context

```
execution-api (Interface)
    │
    ├──> TrackExecutionUseCase (Application)
    │       │
    │       ├──> Execution (Domain Entity)
    │       ├──> ExecutionRepository (Domain Interface)
    │       └──> ExecutionTracker (Domain Service)
    │
    └──> PostgreSQLExecutionRepository (Infrastructure)
            └──> Implements ExecutionRepository
```

**Dependencies**:
- Execution Context → Workflow Context (tracks workflow runs)
- Execution Context → Task Context (tracks task executions)
- Execution Context → Agent Context (tracks agent actions)

## Cross-Cutting Concerns

### Dependency Injection Flow

```
main.py (Composition Root)
    │
    ├──> Infrastructure Layer Instances
    │       ├──> PostgreSQLWorkflowRepository
    │       ├──> RedisCache
    │       └──> RabbitMQEventBus
    │
    ├──> Application Layer Instances
    │       ├──> ExecuteWorkflowUseCase(workflow_repo, executor)
    │       ├──> CreateTaskUseCase(task_repo, validator)
    │       └──> StoreMemoryUseCase(memory_repo, indexer)
    │
    └──> Interface Layer Instances
            ├──> WorkflowController(execute_workflow_use_case)
            └──> TaskController(create_task_use_case)
```

### Event Flow

```
Domain Event → Domain Event Bus (Interface) → Application Event Handlers
    │
    ├──> WorkflowExecutedEvent
    │       └──> UpdateMetricsHandler (Application)
    │
    ├──> TaskCompletedEvent
    │       └──> NotifySubscribersHandler (Application)
    │
    └──> MemoryStoredEvent
            └──> IndexMemoryHandler (Application)
```

**Key Principle**: Events are defined in domain, handlers in application layer.

## Package Dependencies

### Core Domain (Zero Dependencies)

```python
# core/domain/workflow/entities.py
from dataclasses import dataclass
from typing import List
# NO IMPORTS from application, infrastructure, or interface layers

@dataclass
class Workflow:
    id: str
    version: str
    steps: List[WorkflowStep]
```

### Application Layer

```python
# core/application/workflow/use_cases.py
from core.domain.workflow.entities import Workflow
from core.domain.workflow.repositories import WorkflowRepository
# ONLY imports from domain layer

class ExecuteWorkflowUseCase:
    def __init__(self, repo: WorkflowRepository):
        self.repo = repo  # Depends on interface, not implementation
```

### Infrastructure Layer

```python
# infrastructure/persistence/workflow_repository.py
from core.domain.workflow.entities import Workflow
from core.domain.workflow.repositories import WorkflowRepository
from sqlalchemy import select
# Imports from domain + external libraries

class PostgreSQLWorkflowRepository(WorkflowRepository):
    async def get_by_id(self, id: str) -> Workflow:
        # Implementation using SQLAlchemy
        ...
```

### Interface Layer

```python
# services/workflow-engine/src/api/controllers.py
from core.application.workflow.use_cases import ExecuteWorkflowUseCase
from fastapi import APIRouter
# Imports from application layer + web framework

router = APIRouter()

@router.post("/workflows/{id}/execute")
async def execute_workflow(id: str, use_case: ExecuteWorkflowUseCase):
    return await use_case.execute(id)
```

## Forbidden Dependencies

### ❌ Domain → Application

```python
# WRONG: Domain importing from application layer
from core.application.workflow.use_cases import ExecuteWorkflowUseCase

class Workflow:
    def execute_via_use_case(self):  # ❌ Domain shouldn't know about use cases
        use_case = ExecuteWorkflowUseCase()
        use_case.execute(self.id)
```

### ❌ Domain → Infrastructure

```python
# WRONG: Domain importing infrastructure
from infrastructure.persistence.models import WorkflowModel

class Workflow:
    def save_to_db(self):  # ❌ Domain shouldn't know about database
        model = WorkflowModel.from_entity(self)
        session.add(model)
```

### ❌ Application → Interface

```python
# WRONG: Use case importing controller
from services.workflow_engine.src.api.controllers import WorkflowController

class ExecuteWorkflowUseCase:
    def execute(self):
        controller = WorkflowController()  # ❌ Application shouldn't know about HTTP
        ...
```

### ❌ Infrastructure → Interface

```python
# WRONG: Repository importing from API layer
from services.workflow_engine.src.api.schemas import WorkflowResponse

class PostgreSQLWorkflowRepository:
    async def get(self) -> WorkflowResponse:  # ❌ Repo shouldn't return API schemas
        ...
```

## Dependency Inversion Examples

### Example 1: Workflow Execution

**Without DI (Bad)**:
```python
class ExecuteWorkflowUseCase:
    def __init__(self):
        # Hardcoded dependency on infrastructure
        self.repo = PostgreSQLWorkflowRepository()  # ❌
```

**With DI (Good)**:
```python
class ExecuteWorkflowUseCase:
    def __init__(self, repo: WorkflowRepository):  # Interface from domain
        self.repo = repo  # ✅ Depends on abstraction
```

### Example 2: Event Publishing

**Without DI (Bad)**:
```python
class Workflow:
    def complete(self):
        self.status = WorkflowStatus.COMPLETED
        # Direct dependency on infrastructure
        rabbitmq_client.publish("workflow.completed", self.id)  # ❌
```

**With DI (Good)**:
```python
class Workflow:
    def complete(self) -> WorkflowCompletedEvent:
        self.status = WorkflowStatus.COMPLETED
        return WorkflowCompletedEvent(workflow_id=self.id)  # ✅ Returns event

# Application layer handles publishing
class ExecuteWorkflowUseCase:
    def __init__(self, event_bus: EventBus):  # Interface
        self.event_bus = event_bus

    async def execute(self):
        workflow.complete()
        event = workflow.complete()
        await self.event_bus.publish(event)  # ✅
```

## Module Import Rules

### Allowed Imports by Layer

| Layer          | Can Import From                          |
|----------------|------------------------------------------|
| Domain         | Python stdlib only, typing, dataclasses  |
| Application    | Domain + Python stdlib                   |
| Infrastructure | Domain + Application + External libs     |
| Interface      | Application + Web frameworks             |

### Example Import Map

```python
# core/domain/workflow/entities.py
from dataclasses import dataclass      # ✅ stdlib
from typing import List, Optional      # ✅ stdlib
from enum import Enum                  # ✅ stdlib

# core/application/workflow/use_cases.py
from core.domain.workflow.entities import Workflow  # ✅ domain
from core.domain.workflow.repositories import WorkflowRepository  # ✅ domain
from typing import List  # ✅ stdlib

# infrastructure/persistence/workflow_repository.py
from core.domain.workflow.entities import Workflow  # ✅ domain
from core.domain.workflow.repositories import WorkflowRepository  # ✅ domain
from sqlalchemy import select  # ✅ external lib for infrastructure
from redis import Redis  # ✅ external lib for infrastructure

# services/workflow-engine/src/api/controllers.py
from core.application.workflow.use_cases import ExecuteWorkflowUseCase  # ✅ application
from fastapi import APIRouter, Depends  # ✅ web framework
```

## Testing Dependencies

### Unit Testing (Domain Layer)

```python
# tests/unit/domain/test_workflow.py
from core.domain.workflow.entities import Workflow
# NO mocking needed - pure domain logic

def test_workflow_can_execute():
    workflow = Workflow(...)
    assert workflow.can_execute() == True
```

### Integration Testing (Application Layer)

```python
# tests/integration/application/test_execute_workflow.py
from core.application.workflow.use_cases import ExecuteWorkflowUseCase
from core.domain.workflow.repositories import WorkflowRepository

class InMemoryWorkflowRepository(WorkflowRepository):
    # Fake implementation for testing
    ...

def test_execute_workflow_use_case():
    repo = InMemoryWorkflowRepository()
    use_case = ExecuteWorkflowUseCase(repo)
    result = await use_case.execute(...)
```

### E2E Testing (Full Stack)

```python
# tests/e2e/test_workflow_api.py
from fastapi.testclient import TestClient
# Tests full dependency chain

def test_execute_workflow_endpoint():
    response = client.post("/workflows/123/execute")
    assert response.status_code == 200
```

## Verification Tools

### 1. Import Linter

```bash
# Check for forbidden imports
python scripts/check_dependencies.py

# Example violations:
# ❌ core/domain/workflow/entities.py imports from core.application
# ❌ core/application/workflow/use_cases.py imports from fastapi
```

### 2. Dependency Graph Generator

```bash
# Generate visual dependency graph
python scripts/generate_dependency_graph.py --output docs/v3/graphs/

# Output:
# - dependency_graph.png
# - circular_dependencies.txt
# - layer_violations.txt
```

### 3. Architecture Tests

```python
# tests/architecture/test_dependency_rules.py
import pytest
from arch_test import check_dependencies

def test_domain_has_no_external_dependencies():
    violations = check_dependencies("core/domain")
    assert len(violations) == 0, f"Domain has forbidden imports: {violations}"

def test_application_only_depends_on_domain():
    violations = check_dependencies("core/application", allowed=["core.domain"])
    assert len(violations) == 0
```

## Migration Strategy

### Phase 1: Current State Analysis

```bash
# Generate current dependency graph
python scripts/analyze_current_dependencies.py

# Output: Shows existing circular dependencies and violations
```

### Phase 2: Gradual Extraction

```
Week 1: Extract domain entities (no dependencies)
Week 2: Create application use cases (depend on domain)
Week 3: Implement infrastructure (depend on domain interfaces)
Week 4: Refactor interfaces (depend on application)
```

### Phase 3: Validation

```bash
# Run dependency checks in CI/CD
pre-commit hook: python scripts/check_dependencies.py
CI pipeline: pytest tests/architecture/
```

## Conclusion

The dependency graph ensures:
- **Testability**: Domain logic has no external dependencies
- **Flexibility**: Infrastructure can be swapped without changing domain
- **Clarity**: Each layer has well-defined responsibilities
- **Maintainability**: Changes propagate in predictable directions

Next: Begin extracting domain entities following this dependency structure.
