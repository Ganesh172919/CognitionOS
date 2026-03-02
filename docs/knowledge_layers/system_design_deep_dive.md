# System Design Deep Dive (End-to-End Flows)

This doc describes key CognitionOS flows using **ASCII diagrams**, plus distributed-systems trade-offs (idempotency, retries, backpressure, timeouts, load shedding).

## End-to-end flows (ASCII)

### 1) Workflow creation flow
```text
Client
  |
  | POST /api/v3/workflows
  v
FastAPI route: services/api/src/routes/workflows.py
  |
  | CreateWorkflowCommand
  v
Use case: core/application/workflow/use_cases.py::CreateWorkflowUseCase
  |
  | WorkflowStep + Workflow aggregate
  | WorkflowValidator.validate_dag()
  v
Repo (adapter): infrastructure/persistence/workflow_repository.py::PostgreSQLWorkflowRepository
  |
  v
DB: workflows (database/migrations/002_v3_clean_architecture.sql)
```

### 2) Workflow execution flow (step readiness/orchestration)
```text
Client
  |
  | POST /api/v3/workflows/execute
  v
API: ExecuteWorkflowUseCase
  |
  | creates WorkflowExecution + StepExecution rows
  v
DB: workflow_executions, step_executions
  |
  | "driver loop" / workers (documented patterns)
  v
Domain: WorkflowExecutionOrchestrator.get_ready_steps()
  |
  | schedule ready steps (Celery/RabbitMQ/etc.)
  v
Workers
  |
  | execute_step_async(...) (see infrastructure/tasks/workflow_tasks.py)
  v
DB: step_executions + step_execution_attempts (migration 008)
  |
  v
Completion: mark execution terminal + publish events
```

### 3) Memory write/read flow (tiers + vector search)
```text
Write (L1)
Client/API -> StoreWorkingMemoryUseCase
  -> WorkingMemory entity
  -> PostgreSQLWorkingMemoryRepository
  -> DB: working_memory (vector(1536))

Read (tiered)
Client/API -> SearchMemoriesAcrossTiersUseCase
  -> query embedding
  -> L1: working_memory cosine_distance()
  -> L2: episodic_memory cosine_distance()
  -> L3: longterm_memory cosine_distance()
  -> merge + rank + threshold (documented pattern)
```

## Distributed systems trade-offs (trade-off table)

| Concern | “Simple default” | Production-aware trade-off | Recommended direction |
|---|---|---|---|
| Delivery semantics | at-least-once (Celery/RabbitMQ) | duplicates cause double-writes / double tool execution | add idempotency keys + dedupe at DB boundary (migration `008_execution_persistence.sql`) |
| Retries | naive retries everywhere | retry storms amplify outages | bounded retries + jitter + circuit breakers + backpressure |
| Timeouts | long timeouts | resource starvation and poor tail latency | tiered timeouts (client → API → downstream) + cancellation |
| Consistency | “eventual” by accident | user sees stale reads or missing outputs | define RYW expectations per endpoint; implement read models where needed |
| Backpressure | none | queue explosion, OOM, DB overload | concurrency limits + queue thresholds + load shedding |

## Alternative architectures (when scaling pressure shows up)

### Orchestration: Celery vs “Temporal-like” engines
| Option | Pros | Cons | When to choose |
|---|---|---|---|
| Celery + RabbitMQ | simple, familiar, cheap | limited workflow state machine semantics; idempotency is on you | early stage; moderate complexity workflows |
| Temporal-like orchestration | durable execution semantics; retries/timers/state built-in | new operational surface area | long-running workflows, replay/determinism requirements |

### Persistence posture: DB-centric vs event-sourced
| Option | Pros | Cons |
|---|---|---|
| DB-centric state (current) | easy querying; operationally familiar | hard to reproduce past states without explicit snapshots |
| Event sourcing + projections | time travel, replay, audit trails | complex; projection lag; schema evolution in events |

## Advanced Engineering Notes

### Failure Scenarios: “Duplicate step execution”
Symptoms
- A step’s side effects happen twice (double charge, double email, duplicate tool output).
- Workflow appears to “skip ahead” or emits conflicting outputs.

Root cause
- At-least-once delivery + retry without idempotency at the side-effect boundary.

Mitigation
- Define and persist an idempotency key per step attempt (see `database/migrations/008_execution_persistence.sql`).
- Make side-effecting adapters idempotent (e.g., “upsert by idempotency_key”).

Observability signals (logs/metrics/traces)
- Metrics: spikes in `workflow_steps_executed_total{status="completed"}` for same `(workflow_id, step_id)`.
- Logs: repeated “starting step” messages with same logical step but different attempt numbers.
- Traces: two spans with identical semantic attributes (execution_id/step_id) overlapping in time.

### Complexity & Performance
- Step readiness computation is typically **O(V + E)** per “driver loop” iteration (V steps, E dependencies).
- If readiness is recomputed too frequently, overall scheduling can become **O(K·(V+E))** where K is number of scheduling ticks; batch updates and event-driven readiness reduce K.

## Research Extensions

- Formalize execution semantics as a small state machine and property-test it (see `research_extensions.md`).
- Add a replay harness backed by `step_execution_attempts` to detect nondeterminism regressions.

## System Design Deep Dive

(This document is the deep dive; other docs link back here for system-level flow context.)

## Future Evolution Strategy

See `future_evolution_strategy.md` for the concrete steps to reach deterministic replay, outbox-based eventing, and mature orchestration.

