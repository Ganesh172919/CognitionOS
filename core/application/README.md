# CognitionOS V3 Application Layer

## Overview

The **Application Layer** contains use cases that orchestrate domain entities and coordinate business workflows. This layer implements application-specific business rules.

### Clean Architecture Position

```
Interface Layer (Controllers, CLI)
        ↓
Application Layer (Use Cases) ← YOU ARE HERE
        ↓
Domain Layer (Entities, Services)
        ↑
Infrastructure Layer (Repositories, External Services)
```

### Key Principles

1. **Depends only on Domain**: No dependencies on infrastructure or interface layers
2. **Use Case Pattern**: Each use case represents one application operation
3. **DTOs (Data Transfer Objects)**: Commands and results for decoupling
4. **Event Publishing**: Publishes domain events for side effects
5. **Thin Layer**: Orchestrates domain logic, doesn't contain it

### Structure

```
core/application/
├── workflow/
│   ├── __init__.py
│   └── use_cases.py          # Workflow use cases
├── agent/
│   ├── __init__.py
│   └── use_cases.py          # Agent use cases
└── README.md                  # This file
```

## Use Cases

### Workflow Use Cases

#### CreateWorkflowUseCase
Creates a new workflow definition.

**Dependencies:**
- `WorkflowRepository` (domain interface)
- `EventPublisher` (optional)

**Flow:**
1. Parse workflow definition from command
2. Validate DAG using `WorkflowValidator`
3. Activate workflow if valid
4. Persist via repository
5. Publish `WorkflowCreated` event

**Example:**
```python
from core.application.workflow import (
    CreateWorkflowUseCase,
    CreateWorkflowCommand
)

# Setup (dependency injection)
use_case = CreateWorkflowUseCase(
    workflow_repository=postgres_workflow_repo,
    event_publisher=event_bus
)

# Execute
command = CreateWorkflowCommand(
    workflow_id="deploy-app",
    version="1.0.0",
    name="Deploy Application",
    description="CI/CD pipeline",
    steps=[
        {
            "id": "build",
            "type": "execute_task",
            "params": {"command": "npm run build"}
        },
        {
            "id": "deploy",
            "type": "kubernetes_apply",
            "depends_on": ["build"],
            "params": {"manifest": "k8s/deployment.yaml"}
        }
    ]
)

workflow_id = await use_case.execute(command)
```

#### ExecuteWorkflowUseCase
Starts execution of a workflow.

**Flow:**
1. Load workflow definition
2. Validate workflow can execute
3. Create `WorkflowExecution` entity
4. Create `StepExecution` entities for each step
5. Persist execution
6. Publish `WorkflowExecutionStarted` event

#### ProcessWorkflowStepUseCase
Processes ready workflow steps.

**Flow:**
1. Load workflow and execution
2. Get ready steps using `WorkflowExecutionOrchestrator`
3. Execute ready steps
4. Check if workflow complete
5. Update execution status
6. Publish completion events

### Agent Use Cases

#### RegisterAgentDefinitionUseCase
Registers a new agent blueprint.

**Example:**
```python
from core.application.agent import (
    RegisterAgentDefinitionUseCase,
    RegisterAgentDefinitionCommand
)
from core.domain.agent import AgentRole

command = RegisterAgentDefinitionCommand(
    name="code-executor",
    role=AgentRole.EXECUTOR,
    version="1.0.0",
    description="Executes code tasks",
    capabilities=["python", "javascript", "bash"],
    tools=[
        {
            "name": "execute_code",
            "description": "Run code",
            "parameters": {"language": "str", "code": "str"}
        }
    ],
    model_config={
        "provider": "anthropic",
        "model_name": "claude-3-opus",
        "temperature": 0.7
    },
    default_budget={
        "max_tokens": 32000,
        "max_cost_usd": 1.0
    },
    system_prompt="You are a code execution agent..."
)

definition_id = await use_case.execute(command)
```

#### CreateAgentUseCase
Creates agent instance from definition.

#### AssignTaskToAgentUseCase
Assigns task to best available agent.

**Flow:**
1. Get idle agents with required capabilities
2. Use `AgentMatcher` to find best agent
3. Assign task to agent
4. Publish `AgentAssigned` event

## Dependency Injection

Use cases receive dependencies via constructor injection:

```python
# BAD: Hardcoded dependencies
class CreateWorkflowUseCase:
    def __init__(self):
        self.repo = PostgreSQLWorkflowRepository()  # ❌ Coupled to infrastructure

# GOOD: Dependency injection
class CreateWorkflowUseCase:
    def __init__(self, workflow_repository: WorkflowRepository):  # ✅ Depends on interface
        self.workflow_repository = workflow_repository
```

### Composition Root Example

```python
# services/workflow-engine/src/main.py
from core.application.workflow import CreateWorkflowUseCase
from infrastructure.persistence import PostgreSQLWorkflowRepository
from infrastructure.events import RabbitMQEventBus

# Composition root - wire dependencies
workflow_repo = PostgreSQLWorkflowRepository(db_session)
event_bus = RabbitMQEventBus(connection)

create_workflow_use_case = CreateWorkflowUseCase(
    workflow_repository=workflow_repo,
    event_publisher=event_bus
)

# Controller uses use case
@app.post("/workflows")
async def create_workflow(request: WorkflowRequest):
    command = CreateWorkflowCommand(**request.dict())
    workflow_id = await create_workflow_use_case.execute(command)
    return {"workflow_id": str(workflow_id)}
```

## Testing Use Cases

Use cases are easy to test with fake repositories:

```python
# tests/application/test_create_workflow.py
from core.application.workflow import CreateWorkflowUseCase, CreateWorkflowCommand
from tests.fakes import InMemoryWorkflowRepository

async def test_create_workflow():
    # Arrange
    repo = InMemoryWorkflowRepository()
    use_case = CreateWorkflowUseCase(workflow_repository=repo)

    command = CreateWorkflowCommand(
        workflow_id="test-workflow",
        version="1.0.0",
        name="Test",
        description="Test workflow",
        steps=[
            {"id": "step1", "type": "execute_task", "params": {}}
        ]
    )

    # Act
    workflow_id = await use_case.execute(command)

    # Assert
    assert workflow_id.value == "test-workflow"
    workflows = await repo.get_by_status(WorkflowStatus.ACTIVE)
    assert len(workflows) == 1
```

## Next Steps

1. Implement remaining use cases for Memory, Task, Execution contexts
2. Add integration tests
3. Create fake repositories for testing
4. Wire use cases in existing services

---

**V3 Phase 1C:** Application Layer ✅
