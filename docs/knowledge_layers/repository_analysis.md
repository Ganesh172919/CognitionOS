# Repository Analysis (Code-Anchored)

This document is a structured walkthrough of CognitionOS with a bias toward **“where does this behavior live?”** and **dependency boundaries**.

## Repo walkthrough (top-level)

| Directory | Primary responsibility | Notes / coupling points |
|---|---|---|
| `core/` | Domain + application (Clean Architecture / DDD) | Domain should be stdlib-only; application orchestrates repositories/services |
| `services/` | Runtime services (FastAPI API, worker services, etc.) | `services/api/` is the primary async API surface |
| `infrastructure/` | Concrete integrations (DB repos, Redis pool, RabbitMQ bus, OTel/Prometheus) | Keep infra from leaking into `core/domain/*` |
| `database/` | Schema + migrations + DB utilities | pgvector + migrations define tables used by infra repositories |
| `frontend/` | Next.js/TypeScript UI | Treat as consumer of API contracts |
| `kubernetes/`, Compose files | Deployment descriptors | Production posture decisions live here |
| `tests/` | Test suites | Prefer testing domain/application without infra where possible |

### Dependency direction (what we want)
```text
            +-------------------+
            |   services/*      |  (controllers, HTTP, ASGI)
            +---------+---------+
                      |
                      v
            +-------------------+
            | core/application  |  (use cases, orchestration)
            +---------+---------+
                      |
                      v
            +-------------------+
            |   core/domain     |  (entities, value objects, domain services)
            +---------+---------+
                      ^
                      |
            +-------------------+
            | infrastructure/*  |  (DB/Redis/RabbitMQ implementations)
            +-------------------+
```

## Where to look for X (index)

| You need… | Start here | Then follow… |
|---|---|---|
| FastAPI app lifecycle, middleware, health | `services/api/src/main.py` | `services/api/src/middleware/*`, `services/api/src/dependencies/injection.py` |
| Workflow create/execute HTTP routes | `services/api/src/routes/workflows.py` | `core/application/workflow/use_cases.py` → `core/domain/workflow/*` |
| DAG validation / topo-sort | `core/domain/workflow/entities.py` | `core/domain/workflow/services.py` |
| Step readiness/orchestration | `core/domain/workflow/services.py` | (look for “ready steps”, “can continue”) |
| Memory tiering (L1/L2/L3) | `core/domain/memory_hierarchy/entities.py` | `core/domain/memory_hierarchy/services.py` |
| Memory vector search (pgvector) | `infrastructure/persistence/memory_hierarchy_repository.py` | `infrastructure/persistence/memory_hierarchy_models.py`, `database/migrations/003_phase3_extended_operation.sql` |
| DB migrations & schema | `database/migrations/` | `database/run_migrations.py`, `database/README.md` |
| Observability (logs/metrics/tracing) | `infrastructure/observability/` | `services/api/src/main.py`, `services/api/src/middleware/request_id.py` |
| RabbitMQ health/eventing | `services/api/src/dependencies/injection.py` | `infrastructure/message_broker/rabbitmq_event_bus.py`, `infrastructure/health/checks.py` |
| Celery tasks | `infrastructure/tasks/` | `core/config.py` (Celery config), worker entrypoints in services |

## Cross-module linkage table (route → use case → domain → repo → DB)

This table is the “golden path” for tracing runtime behavior from HTTP to storage.

| API surface | Use case (application) | Domain entities/services | Infra repo(s) | DB tables / migrations |
|---|---|---|---|---|
| `POST /api/v3/workflows` | `CreateWorkflowUseCase` (`core/application/workflow/use_cases.py`) | `Workflow`, `WorkflowStep`, `WorkflowValidator` (`core/domain/workflow/*`) | `PostgreSQLWorkflowRepository` (`infrastructure/persistence/workflow_repository.py`) | `workflows` (`database/migrations/002_v3_clean_architecture.sql`) |
| `POST /api/v3/workflows/execute` | `ExecuteWorkflowUseCase` (`core/application/workflow/use_cases.py`) | `WorkflowExecution`, `StepExecution`, `WorkflowExecutionOrchestrator` | `PostgreSQLWorkflowExecutionRepository` | `workflow_executions`, `step_executions` (`002_v3_clean_architecture.sql`) + attempt/idempotency (`008_execution_persistence.sql`) |
| `POST /api/v3/memory/working` | `StoreWorkingMemoryUseCase` (`core/application/memory_hierarchy/use_cases.py`) | `WorkingMemory`, `MemoryEmbedding` | `PostgreSQLWorkingMemoryRepository` (`infrastructure/persistence/memory_hierarchy_repository.py`) | `working_memory` (`003_phase3_extended_operation.sql`) |
| `GET /api/v3/memory/working/{agent_id}` | `RetrieveWorkingMemoryUseCase` | `WorkingMemory.update_access()` semantics | same as above | `working_memory` + indexes (`003_phase3_extended_operation.sql`) |
| `POST /api/v3/memory/search` | `SearchMemoriesAcrossTiersUseCase` | tiered search semantics | tier repositories (L1/L2/L3) | `working_memory`, `episodic_memory`, `longterm_memory` + pgvector indexes (`003_phase3_extended_operation.sql`) |
| `GET /health` | health handlers in `services/api/src/main.py` | (N/A) | `check_database_health`, Redis pool, `aio_pika` robust connect | DB/Redis/RabbitMQ connectivity checks |

## Coupling points & boundary checks

Common “boundary crossing” locations (important during refactors):
- `services/api/src/dependencies/injection.py`: composition root for repositories + use cases; also owns a singleton SQLAlchemy engine.
- `infrastructure/persistence/*`: concrete DB adapters (SQLAlchemy models + mapping).
- `database/migrations/*`: schema is the runtime contract for persistence adapters.
- `infrastructure/observability/*`: tracing/metrics/logging wiring.

## Architecture smells to watch

- **Multiple DB/session patterns**: `database/connection.py`, `infrastructure/persistence/base.py`, and `services/api/src/dependencies/injection.py` each manage engines/sessions. This increases the risk of inconsistent pooling, instrumentation, and transaction semantics.
- **Schema/ORM drift**: Migrations in `database/migrations/*` and SQLAlchemy models in both `database/models.py` and `infrastructure/persistence/*_models.py` can diverge. Treat migrations as the “source of truth” for production schema.
- **Domain dependency leakage**: any import from `infrastructure/*` inside `core/domain/*` is a red flag.
- **High-cardinality metrics labels**: Prometheus labels on raw paths (with IDs) can blow up series count (see `infrastructure/observability/metrics.py`).

## Advanced Engineering Notes

### Failure Scenarios: “Boundary erosion”
Symptoms
- Domain code starts requiring SQLAlchemy types, FastAPI objects, or network clients.
- Tests for domain entities require database/mocks to run.

Root cause
- “Quick fix” changes that add infrastructure dependencies inside domain/application.

Mitigation
- Enforce dependency rules in code review: domain is stdlib-only; infra implements domain interfaces.
- Keep adapters in `infrastructure/` and expose only interfaces to `core/application/*`.

Observability signals (logs/metrics/traces)
- Increased DB calls inside request handlers that should be pure orchestration.
- Traces showing unexpected DB spans during “validation-only” operations.

### Complexity & Performance
- Building a trace from route → use case → repo is **O(H)** in number of hops; codify hops in this table to keep it small.
- The real performance risks are typically at coupling points (DB pools, queues, vector indexes), not inside domain object creation.

## Research Extensions

- Automate extraction of this linkage table by parsing FastAPI routes + injection wiring (static analysis).
- Add a “concept-to-code” linter that fails CI if a new route lacks a knowledge-layer mapping entry.

## System Design Deep Dive

See `system_design_deep_dive.md` for the execution sequences that traverse these boundaries.

## Future Evolution Strategy

See `future_evolution_strategy.md` for work items that reduce coupling (outbox, orchestration upgrades, unified DB session management).

