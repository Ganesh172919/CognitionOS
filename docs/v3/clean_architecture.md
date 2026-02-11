# CognitionOS V3 - Clean Architecture Specification

**Version**: 3.0
**Date**: 2026-02-11
**Purpose**: Define clean architecture principles for CognitionOS V3

---

## Overview

CognitionOS V3 adopts **Clean Architecture** (also known as Hexagonal Architecture or Ports & Adapters) to achieve:

1. **Independence from frameworks** - Core logic doesn't depend on FastAPI, SQLAlchemy, etc.
2. **Testability** - Domain logic can be tested without external dependencies
3. **Independence from UI** - Business rules don't know about the web interface
4. **Independence from Database** - Domain doesn't depend on PostgreSQL specifics
5. **Independence from external agencies** - Business rules don't know about LLM APIs

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     INTERFACE LAYER                         │
│  (HTTP APIs, CLI, Event Handlers)                           │
│  • FastAPI routes                                           │
│  • WebSocket handlers                                       │
│  • CLI commands                                             │
└────────────────────────┬────────────────────────────────────┘
                         │ Depends on ↓
┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                         │
│  (Use Cases, Application Services)                          │
│  • ExecuteWorkflowUseCase                                   │
│  • CreateAgentUseCase                                       │
│  • StoreMemoryUseCase                                       │
└────────────────────────┬────────────────────────────────────┘
                         │ Depends on ↓
┌─────────────────────────────────────────────────────────────┐
│                     DOMAIN LAYER                            │
│  (Entities, Value Objects, Domain Services)                 │
│  • Workflow                                                 │
│  • Agent                                                    │
│  • Memory                                                   │
│  • Task                                                     │
└────────────────────────┬────────────────────────────────────┘
                         │ Uses ↑ (via interfaces)
┌─────────────────────────────────────────────────────────────┐
│                 INFRASTRUCTURE LAYER                        │
│  (External Dependencies, Implementations)                   │
│  • PostgreSQL repositories                                  │
│  • Redis cache                                              │
│  • OpenAI/Anthropic clients                                 │
│  • File storage                                             │
└─────────────────────────────────────────────────────────────┘
```

**Dependency Rule**: Source code dependencies must only point **inward**. Nothing in an inner circle can know anything about something in an outer circle.

---

## Layer Definitions

### 1. Domain Layer (Core)

**Purpose**: Contains enterprise-wide business rules

**Location**: `/core/domain/`

**Contents**:
- **Entities**: Objects with identity that persist over time
- **Value Objects**: Immutable objects defined by their attributes
- **Domain Services**: Operations that don't naturally fit in entities
- **Repository Interfaces**: Contracts for data access (no implementations)
- **Domain Events**: Events that domain experts care about

**Rules**:
- No dependencies on outer layers
- No framework dependencies
- Pure Python (no FastAPI, SQLAlchemy, etc.)
- No I/O operations
- 100% testable without mocks

**Example Structure**:
```python
# core/domain/entities/workflow.py
from dataclasses import dataclass
from uuid import UUID
from typing import List
from .value_objects import WorkflowStatus, WorkflowStep

@dataclass
class Workflow:
    """Domain entity representing a workflow."""

    id: str
    version: str
    name: str
    steps: List[WorkflowStep]
    status: WorkflowStatus

    def can_execute(self) -> bool:
        """Domain logic: Check if workflow can be executed."""
        return (
            self.status == WorkflowStatus.ACTIVE and
            len(self.steps) > 0 and
            self._has_valid_dag()
        )

    def _has_valid_dag(self) -> bool:
        """Validate DAG has no cycles."""
        # Pure domain logic
        pass
```

### 2. Application Layer

**Purpose**: Contains application-specific business rules (use cases)

**Location**: `/application/`

**Contents**:
- **Use Cases**: Application-specific operations
- **Application Services**: Coordinate use cases
- **DTOs (Data Transfer Objects)**: Data structures for layer boundaries
- **Ports**: Interfaces that outer layers must implement

**Rules**:
- Depends only on domain layer
- Orchestrates domain objects
- No framework-specific code
- No database queries (uses repository interfaces)
- No HTTP knowledge

**Example Structure**:
```python
# application/use_cases/workflows/execute_workflow.py
from dataclasses import dataclass
from uuid import UUID
from core.domain.entities import Workflow
from core.domain.repositories import WorkflowRepository, ExecutionRepository
from core.domain.services import WorkflowExecutor

@dataclass
class ExecuteWorkflowCommand:
    """Input DTO for use case."""
    workflow_id: str
    workflow_version: str
    inputs: dict
    user_id: UUID

@dataclass
class ExecuteWorkflowResult:
    """Output DTO from use case."""
    execution_id: UUID
    status: str

class ExecuteWorkflowUseCase:
    """
    Use case: Execute a workflow.

    This is application logic that coordinates domain entities
    and services to accomplish a specific task.
    """

    def __init__(
        self,
        workflow_repo: WorkflowRepository,
        execution_repo: ExecutionRepository,
        executor: WorkflowExecutor
    ):
        self.workflow_repo = workflow_repo
        self.execution_repo = execution_repo
        self.executor = executor

    async def execute(self, command: ExecuteWorkflowCommand) -> ExecuteWorkflowResult:
        # Get workflow (domain entity)
        workflow = await self.workflow_repo.get_by_version(
            command.workflow_id,
            command.workflow_version
        )

        if not workflow:
            raise WorkflowNotFound()

        # Check domain rules
        if not workflow.can_execute():
            raise WorkflowNotExecutable()

        # Execute (domain service)
        execution = await self.executor.execute(
            workflow=workflow,
            inputs=command.inputs,
            user_id=command.user_id
        )

        # Persist
        await self.execution_repo.save(execution)

        # Return DTO
        return ExecuteWorkflowResult(
            execution_id=execution.id,
            status=execution.status.value
        )
```

### 3. Infrastructure Layer

**Purpose**: Contains all external dependencies and their implementations

**Location**: `/infrastructure/`

**Contents**:
- **Persistence**: Database implementations
- **External Services**: LLM clients, external APIs
- **File System**: File operations
- **Messaging**: Message queues, event buses
- **Caching**: Redis, memcached

**Rules**:
- Implements interfaces defined in domain layer
- Contains all framework-specific code
- All I/O operations happen here
- Can depend on any layer

**Example Structure**:
```python
# infrastructure/persistence/postgresql/workflow_repository.py
from sqlalchemy.ext.asyncio import AsyncSession
from core.domain.entities import Workflow
from core.domain.repositories import WorkflowRepository
from .models import WorkflowModel

class PostgreSQLWorkflowRepository(WorkflowRepository):
    """
    PostgreSQL implementation of WorkflowRepository interface.

    This is infrastructure code that knows about SQLAlchemy,
    but the domain doesn't know about this.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_version(self, id: str, version: str) -> Workflow:
        # SQLAlchemy-specific code
        model = await self.session.get(
            WorkflowModel,
            {"id": id, "version": version}
        )

        if not model:
            return None

        # Convert from ORM model to domain entity
        return self._to_domain_entity(model)

    def _to_domain_entity(self, model: WorkflowModel) -> Workflow:
        """Convert ORM model to domain entity."""
        return Workflow(
            id=model.id,
            version=model.version,
            name=model.name,
            steps=[...],  # Convert steps
            status=WorkflowStatus(model.status)
        )
```

### 4. Interface Layer

**Purpose**: Handles external communication (HTTP, CLI, events)

**Location**: `/interfaces/`

**Contents**:
- **API**: HTTP routes (FastAPI)
- **CLI**: Command-line interfaces
- **Events**: Event handlers
- **Presenters**: Format output for different interfaces

**Rules**:
- Depends on application layer
- Translates external requests to use case commands
- Translates use case results to external responses
- Framework-specific (FastAPI, Click, etc.)

**Example Structure**:
```python
# interfaces/api/workflows/routes.py
from fastapi import APIRouter, Depends
from application.use_cases.workflows import (
    ExecuteWorkflowUseCase,
    ExecuteWorkflowCommand
)

router = APIRouter()

@router.post("/workflows/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    request: ExecuteWorkflowRequest,
    use_case: ExecuteWorkflowUseCase = Depends(get_execute_workflow_use_case)
):
    """
    HTTP endpoint that translates HTTP request to use case command.

    This is interface code - it knows about FastAPI and HTTP,
    but the use case doesn't.
    """

    # Translate HTTP request to use case command
    command = ExecuteWorkflowCommand(
        workflow_id=workflow_id,
        workflow_version=request.version,
        inputs=request.inputs,
        user_id=request.user_id
    )

    # Execute use case
    result = await use_case.execute(command)

    # Translate use case result to HTTP response
    return ExecuteWorkflowResponse(
        execution_id=str(result.execution_id),
        status=result.status
    )
```

---

## Dependency Injection

**Problem**: How do use cases get repository implementations without depending on infrastructure?

**Solution**: Dependency injection at application startup

```python
# interfaces/api/dependencies.py
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from application.use_cases.workflows import ExecuteWorkflowUseCase
from infrastructure.persistence.postgresql import PostgreSQLWorkflowRepository
from core.domain.services import WorkflowExecutor

async def get_db_session() -> AsyncSession:
    """Provide database session."""
    async with async_session_maker() as session:
        yield session

async def get_workflow_repository(
    session: AsyncSession = Depends(get_db_session)
) -> WorkflowRepository:
    """Provide workflow repository implementation."""
    return PostgreSQLWorkflowRepository(session)

async def get_execute_workflow_use_case(
    workflow_repo: WorkflowRepository = Depends(get_workflow_repository),
    execution_repo: ExecutionRepository = Depends(get_execution_repository),
    executor: WorkflowExecutor = Depends(get_workflow_executor)
) -> ExecuteWorkflowUseCase:
    """Provide use case with all dependencies."""
    return ExecuteWorkflowUseCase(
        workflow_repo=workflow_repo,
        execution_repo=execution_repo,
        executor=executor
    )
```

---

## Bounded Contexts

CognitionOS has several **bounded contexts** (areas of the domain with clear boundaries):

### 1. Workflow Context
**Responsibility**: Workflow definition, execution, and lifecycle

**Entities**:
- Workflow
- WorkflowExecution
- WorkflowStep

**Value Objects**:
- WorkflowStatus
- ExecutionStatus
- StepType

**Services**:
- WorkflowValidator
- WorkflowExecutor
- WorkflowVersionManager

### 2. Agent Context
**Responsibility**: Agent lifecycle, assignment, and performance

**Entities**:
- Agent
- AgentAssignment
- AgentMetrics

**Value Objects**:
- AgentRole
- AgentStatus
- FailureStrategy

**Services**:
- AgentOrchestrator
- AgentPerformanceTracker
- AgentReplacementService

### 3. Memory Context
**Responsibility**: Memory storage, retrieval, and lifecycle

**Entities**:
- Memory
- MemoryNamespace
- MemoryCluster

**Value Objects**:
- MemoryType
- MemoryScope
- EmbeddingVector

**Services**:
- MemoryRetriever
- MemoryGarbageCollector
- SemanticSearchService

### 4. Task Context
**Responsibility**: Task planning and execution

**Entities**:
- Task
- TaskPlan
- TaskDependency

**Value Objects**:
- TaskStatus
- TaskPriority
- Complexity

**Services**:
- TaskPlanner
- DAGBuilder
- DependencyResolver

### 5. Execution Context
**Responsibility**: Tool execution and sandboxing

**Entities**:
- ToolExecution
- ExecutionEnvironment

**Value Objects**:
- ToolType
- ExecutionResult
- ResourceLimits

**Services**:
- ToolExecutor
- SandboxManager
- PermissionChecker

---

## Migration Strategy

**Phase 1: Extract Domain (Week 1)**
1. Identify core domain entities in existing services
2. Create domain models in `/core/domain/`
3. Keep existing code working alongside new structure

**Phase 2: Create Application Layer (Week 1-2)**
1. Extract use cases from service endpoints
2. Create use case classes in `/application/`
3. Define repository interfaces in domain
4. Use cases call domain logic

**Phase 3: Implement Infrastructure (Week 2)**
1. Move SQLAlchemy models to `/infrastructure/`
2. Implement repository interfaces
3. Move LLM clients to `/infrastructure/`
4. Connect via dependency injection

**Phase 4: Refactor Interfaces (Week 2)**
1. Update FastAPI routes to call use cases
2. Move routes to `/interfaces/api/`
3. Create DTOs for layer boundaries
4. Remove business logic from routes

**Phase 5: Validate (Week 2)**
1. Ensure all tests still pass
2. Verify no business logic in infrastructure
3. Confirm dependency direction is correct
4. Document any exceptions

---

## Benefits

**Testability**:
```python
# Test domain logic without database
def test_workflow_validation():
    workflow = Workflow(...)
    assert workflow.can_execute() == True

# Test use case with mock repository
async def test_execute_workflow_use_case():
    mock_repo = MockWorkflowRepository()
    use_case = ExecuteWorkflowUseCase(mock_repo, ...)

    result = await use_case.execute(command)
    assert result.status == "running"
```

**Flexibility**:
```python
# Easy to switch from PostgreSQL to MongoDB
# Just implement MongoDBWorkflowRepository
# Domain and application layers don't change

# Easy to switch from FastAPI to GraphQL
# Just create new interface layer
# Use cases stay the same
```

**Maintainability**:
- Clear separation of concerns
- Each layer has single responsibility
- Easy to find code (domain logic is in domain, not scattered)
- Changes isolated to specific layers

---

## Examples

### Complete Flow Example

**User Request**: Execute workflow via HTTP

**Flow**:
```
1. HTTP Request
   ↓
2. FastAPI Route (Interface Layer)
   - Parse HTTP request
   - Create ExecuteWorkflowCommand
   ↓
3. ExecuteWorkflowUseCase (Application Layer)
   - Get workflow from repository
   - Validate domain rules
   - Call domain service
   ↓
4. WorkflowExecutor (Domain Service)
   - Execute workflow logic
   - Create WorkflowExecution entity
   ↓
5. ExecutionRepository (Infrastructure)
   - Save to PostgreSQL
   ↓
6. Return through layers
   - Result → DTO → HTTP Response
```

**Code Flow**:
```python
# 1. Interface Layer
@router.post("/workflows/{id}/execute")
async def execute_workflow(
    id: str,
    request: Request,
    use_case: ExecuteWorkflowUseCase = Depends()
):
    command = ExecuteWorkflowCommand(...)
    result = await use_case.execute(command)
    return result

# 2. Application Layer
class ExecuteWorkflowUseCase:
    async def execute(self, command):
        workflow = await self.repo.get(command.workflow_id)
        if not workflow.can_execute():  # Domain logic
            raise WorkflowNotExecutable()
        execution = await self.executor.execute(workflow)
        return ExecutionResult(...)

# 3. Domain Layer
class Workflow:
    def can_execute(self) -> bool:
        return self.status == WorkflowStatus.ACTIVE

# 4. Infrastructure Layer
class PostgreSQLWorkflowRepository:
    async def get(self, id):
        model = await self.session.get(WorkflowModel, id)
        return self._to_domain_entity(model)
```

---

## Rules & Principles

**1. The Dependency Rule**
- Dependencies point inward only
- Inner layers don't know about outer layers
- Domain is the center, most stable

**2. Stable Dependencies Principle**
- Depend on stable abstractions
- Domain is most stable (rarely changes)
- Infrastructure is least stable (frequently changes)

**3. Interface Segregation**
- Repository interfaces are specific to use cases
- Don't create generic CRUD interfaces
- Design interfaces for domain needs

**4. Don't Repeat Yourself (DRY)**
- But don't couple layers to avoid duplication
- DTOs may duplicate domain entities (acceptable)
- Transformation code is necessary

**5. Single Responsibility**
- Each layer has one reason to change
- Domain changes for business rule changes
- Infrastructure changes for technology changes

---

## Anti-Patterns to Avoid

**❌ Don't**: Put business logic in HTTP routes
```python
# BAD
@router.post("/workflows")
async def create_workflow(request):
    if len(request.steps) == 0:  # Business logic in route
        raise ValueError("Workflow needs steps")
    ...
```

**✅ Do**: Put business logic in domain
```python
# GOOD
class Workflow:
    def validate(self):
        if len(self.steps) == 0:
            raise InvalidWorkflow("Needs steps")
```

**❌ Don't**: Let domain depend on infrastructure
```python
# BAD - Domain depends on SQLAlchemy
class Workflow:
    def save(self):
        session.add(self)  # Domain shouldn't know about database
```

**✅ Do**: Use repository interface
```python
# GOOD - Domain defines interface, infrastructure implements
class WorkflowRepository(ABC):
    @abstractmethod
    async def save(self, workflow: Workflow): pass
```

**❌ Don't**: Mix concerns in use cases
```python
# BAD - Use case does HTTP and database
class ExecuteWorkflowUseCase:
    async def execute(self, request: Request):  # HTTP in use case
        result = session.query(...)  # Database in use case
```

**✅ Do**: Separate concerns
```python
# GOOD - Use case is pure application logic
class ExecuteWorkflowUseCase:
    async def execute(self, command: ExecuteWorkflowCommand):
        workflow = await self.repo.get(...)  # Interface
        return ExecuteWorkflowResult(...)  # DTO
```

---

## Conclusion

Clean architecture transforms CognitionOS into a maintainable, testable, flexible platform by:

1. **Separating concerns** into clear layers
2. **Protecting domain logic** from external changes
3. **Making dependencies explicit** through interfaces
4. **Enabling testing** without external dependencies
5. **Future-proofing** the architecture

The investment in clean architecture pays off through:
- Easier onboarding for new developers
- Faster development cycles
- Lower maintenance costs
- Higher code quality
- Better testability

---

**Document Version**: 1.0
**Status**: Active
**Last Updated**: 2026-02-11
