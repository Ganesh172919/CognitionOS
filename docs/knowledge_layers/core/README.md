# Core Knowledge Layer (Domain + Application)

This folder focuses on CognitionOS **core logic**: domain entities/services (pure Python) and application use cases (orchestration).

Primary anchors:
- Workflow domain: `core/domain/workflow/`
- Memory hierarchy domain: `core/domain/memory_hierarchy/`
- Use cases: `core/application/*/use_cases.py`

## Core layer map (bounded contexts)

| Context | Domain package | Application package | Key responsibilities |
|---|---|---|---|
| Workflow | `core/domain/workflow/` | `core/application/workflow/` | DAG validation, versioning, execution records |
| Memory hierarchy | `core/domain/memory_hierarchy/` | `core/application/memory_hierarchy/` | L1/L2/L3 tiering, promotion/eviction, vector search orchestration |
| Agent (overview) | `core/domain/agent/` | `core/application/agent/` | agent definition, capabilities, task assignment |
| Checkpoint (overview) | `core/domain/checkpoint/` | `core/application/checkpoint/` | checkpoint/resume semantics |

## Invariants and purity rules

**Domain (“purity”) rules**
- `core/domain/*` is **stdlib-only** (see headers in domain entity modules).
- Domain services are **stateless** and operate on domain entities (example: `WorkflowValidator`).
- Domain entities enforce invariants in constructors/`__post_init__` methods.

**Application rules**
- `core/application/*` orchestrates domain and repositories.
- Use cases accept **commands/queries** and return IDs/results; they do not do HTTP/ASGI concerns.
- Event publishing is optional and should be behind an interface (example: `InMemoryEventBus` in DI wiring).

### Clean architecture “quick check”
```text
Domain:   core/domain/*                 (no FastAPI/SQLAlchemy/Redis/RabbitMQ)
App:      core/application/*            (orchestrates domain + repos)
Adapters: infrastructure/persistence/*  (SQLAlchemy, pgvector, asyncpg)
Delivery: services/api/*                (FastAPI routes + DI)
```

## Advanced Engineering Notes

### Failure Scenarios: “Domain impurity”
Symptoms
- Domain imports from `sqlalchemy`, `fastapi`, `redis`, `aio_pika`, etc.
- Pure-domain tests now require mocks or DB connections.

Root cause
- Logic accidentally implemented in the wrong layer (common under delivery pressure).

Mitigation
- Move side effects to adapters; expose interfaces in domain/application.
- Enforce dependency direction in review: domain must remain stdlib-only.

Observability signals (logs/metrics/traces)
- Unexpected DB spans during “validation-only” operations.
- Increased startup import time due to heavy dependencies pulled into domain.

### Complexity & Performance
- Domain operations should be cheap and deterministic; typical hotspots are graph algorithms (workflow DAG) and ranking/search (memory retrieval).
- Prefer pushing bulk operations to the database (filters, limits) while keeping domain semantics explicit.

## Research Extensions

- Property-based tests for domain invariants (DAG validity, tier promotion thresholds).
- Deterministic replay harnesses for workflow execution semantics.

## System Design Deep Dive

- See `../system_design_deep_dive.md` for how core logic is exercised end-to-end through API + persistence.

## Future Evolution Strategy

- See `../future_evolution_strategy.md` for roadmap items that affect core semantics (determinism, orchestration, idempotency).

