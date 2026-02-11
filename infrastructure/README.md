# CognitionOS V3 Infrastructure Layer

## Overview

The **Infrastructure Layer** provides concrete implementations of domain repository interfaces and external service adapters. This layer handles all technical concerns like database access, message queues, external APIs, etc.

### Clean Architecture Position

```
Interface Layer (Controllers, CLI)
        ↓
Application Layer (Use Cases)
        ↓
Domain Layer (Entities, Services)
        ↑
Infrastructure Layer ← YOU ARE HERE
```

## Structure

```
infrastructure/
├── persistence/           # Database implementations
│   ├── base.py           # Database connection & session
│   ├── workflow_models.py    # SQLAlchemy ORM models
│   ├── workflow_repository.py # PostgreSQL repository impl
│   └── __init__.py
├── events/               # Event bus implementations
│   ├── event_bus.py     # In-memory event bus
│   └── __init__.py
├── config/               # Configuration management
└── __init__.py
```

## Components

### 1. Persistence (`infrastructure/persistence/`)

#### Database Connection

```python
from infrastructure.persistence import (
    DatabaseConfig,
    init_database,
    get_database
)

# Initialize database
config = DatabaseConfig(
    host="localhost",
    port=5432,
    database="cognition_os",
    user="postgres",
    password="postgres",
    pool_size=20
)

db = init_database(config)

# Get session
async with db.get_session() as session:
    # Use session
    repo = PostgreSQLWorkflowRepository(session)
    await repo.save(workflow)
```

#### Repository Implementation

**PostgreSQL Repositories:**
- `PostgreSQLWorkflowRepository` - Implements `WorkflowRepository` interface
- `PostgreSQLWorkflowExecutionRepository` - Implements `WorkflowExecutionRepository` interface

**Mapping Pattern:**
```python
class PostgreSQLWorkflowRepository(WorkflowRepository):
    def _to_model(self, entity: Workflow) -> WorkflowModel:
        """Domain entity → ORM model"""
        ...

    def _to_entity(self, model: WorkflowModel) -> Workflow:
        """ORM model → Domain entity"""
        ...
```

### 2. Event Bus (`infrastructure/events/`)

Simple in-memory event bus for domain events:

```python
from infrastructure.events import init_event_bus, get_event_bus

# Initialize
event_bus = init_event_bus()

# Subscribe to events
async def handle_workflow_created(event):
    print(f"Workflow created: {event.workflow_id}")

event_bus.subscribe("WorkflowCreated", handle_workflow_created)

# Publish events
from core.domain.workflow.events import WorkflowCreated

event = WorkflowCreated(...)
await event_bus.publish(event)
```

**Production Ready:**
For production, replace with RabbitMQ, Kafka, or AWS EventBridge.

## Database Schema

### Workflow Tables

**workflows**
- Stores workflow definitions
- Composite key: (id, version_major, version_minor, version_patch)
- JSON column for steps

**workflow_executions**
- Tracks workflow execution instances
- Foreign key to workflows
- JSON columns for inputs/outputs

**step_executions**
- Individual step executions
- Foreign key to workflow_executions

### Agent Tables

**agent_definitions**
- Agent blueprints/templates
- JSON columns for tools, config, budget

**agents**
- Agent instances
- Foreign key to agent_definitions
- Tracks status, budget usage

### Memory Tables

**memories_v3**
- Memory entries with vector embeddings
- IVFFlat index on embedding column for fast similarity search
- Namespace for hierarchical organization

**memory_collections**
- Groups of related memories

**memory_lifecycle_policies**
- Rules for compression/archival/deletion

### Task & Execution Tables

**tasks_v3** - Task instances
**execution_traces** - Detailed execution tracking

## Usage Examples

### Complete Setup

```python
# 1. Initialize infrastructure
from infrastructure.persistence import init_database, DatabaseConfig
from infrastructure.events import init_event_bus

db_config = DatabaseConfig(
    host="localhost",
    database="cognition_os"
)

db = init_database(db_config)
event_bus = init_event_bus()

# 2. Create repositories
from infrastructure.persistence import (
    PostgreSQLWorkflowRepository,
    PostgreSQLWorkflowExecutionRepository
)

async with db.get_session() as session:
    workflow_repo = PostgreSQLWorkflowRepository(session)
    execution_repo = PostgreSQLWorkflowExecutionRepository(session)

    # 3. Create use cases
    from core.application.workflow import CreateWorkflowUseCase

    use_case = CreateWorkflowUseCase(
        workflow_repository=workflow_repo,
        event_publisher=event_bus
    )

    # 4. Execute
    from core.application.workflow import CreateWorkflowCommand

    command = CreateWorkflowCommand(
        workflow_id="deploy-app",
        version="1.0.0",
        name="Deploy Application",
        steps=[...]
    )

    workflow_id = await use_case.execute(command)
```

### Testing with Fake Repositories

```python
# tests/fakes/workflow_repository.py
from core.domain.workflow import WorkflowRepository

class InMemoryWorkflowRepository(WorkflowRepository):
    def __init__(self):
        self._workflows = {}

    async def save(self, workflow):
        key = (workflow.id.value, str(workflow.version))
        self._workflows[key] = workflow

    async def get_by_id(self, workflow_id, version):
        key = (workflow_id.value, str(version))
        return self._workflows.get(key)
```

## Database Migrations

### Running Migrations

```bash
# Apply V3 migration
psql -U postgres -d cognition_os -f database/migrations/002_v3_clean_architecture.sql

# Or use migration script
python database/run_migrations.py
```

### Migration Files

- `001_initial_schema.sql` - V1/V2 schema
- `002_v3_clean_architecture.sql` - V3 clean architecture tables

## Configuration

### Environment Variables

```bash
# Database
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=cognition_os
export DB_USER=postgres
export DB_PASSWORD=postgres
export DB_POOL_SIZE=20

# Redis (future)
export REDIS_URL=redis://localhost:6379

# Message Queue (future)
export RABBITMQ_URL=amqp://guest:guest@localhost:5672
```

### Config Class

```python
from dataclasses import dataclass
import os

@dataclass
class InfrastructureConfig:
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "cognition_os")
    db_user: str = os.getenv("DB_USER", "postgres")
    db_password: str = os.getenv("DB_PASSWORD", "postgres")
```

## Next Steps

1. **Implement remaining repositories:**
   - Agent repositories
   - Memory repositories
   - Task repositories
   - Execution trace repositories

2. **Add external service adapters:**
   - LLM clients (OpenAI, Anthropic)
   - Vector database (Pinecone, Weaviate)
   - Message queue (RabbitMQ, Kafka)

3. **Enhance event bus:**
   - Persistent event store
   - Event replay
   - Dead letter queue

4. **Add monitoring:**
   - Database connection pool metrics
   - Repository operation metrics
   - Event publishing metrics

---

**V3 Phase 1D:** Infrastructure Layer ✅
