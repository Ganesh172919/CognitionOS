# Glossary (Canonical Terms)

This glossary provides a **single source of truth** for key CognitionOS terms and how they map to code.

## Concept-to-code map (table)

| Term | Meaning (short) | Primary code anchors | Related knowledge-layer docs |
|---|---|---|---|
| Workflow | Versioned DAG of steps | `core/domain/workflow/entities.py` | `core/workflow_dag_execution.md`, `system_design_deep_dive.md` |
| Step | Node in a workflow DAG | `core/domain/workflow/entities.py` (`WorkflowStep`) | `core/workflow_dag_execution.md` |
| Execution | Runtime instance of a workflow | `core/application/workflow/use_cases.py` (`WorkflowExecution`) | `system_design_deep_dive.md` |
| Step execution attempt | Idempotent attempt record for replay | `database/migrations/008_execution_persistence.sql` | `services_api/api_contracts_versioning.md` (idempotency), `future_evolution_strategy.md` |
| L1/L2/L3 memory | Working/Episodic/Long-term tiers | `core/domain/memory_hierarchy/*` | `core/memory_hierarchy_vector_search.md` |
| pgvector | Vector type + ANN indexes in Postgres | `database/migrations/*` + `infrastructure/persistence/memory_hierarchy_models.py` | `database/schema_migrations_concurrency.md` |
| Request ID | Correlation ID for logs | `services/api/src/middleware/request_id.py` | `services_api/request_lifecycle_middleware.md` |
| Trace | Distributed span graph | `infrastructure/observability/tracing.py` | `cross_cutting/observability.md` |

## Terms (expanded)

- **Clean Architecture**: dependency direction where domain is independent, application orchestrates, infrastructure adapts.
- **DDD bounded context**: cohesive domain area (example: Workflow, Memory Hierarchy).
- **Idempotency key**: a key that makes repeated requests safe (same effect once).
- **Backpressure**: slowing producers when consumers are saturated to prevent overload cascades.

## Advanced Engineering Notes

### Failure Scenarios: “Term overload”
Symptoms
- “Execution”, “task”, and “step” are used interchangeably in code and docs.
- API responses contain ambiguous status fields.

Root cause
- No canonical glossary; terms evolve faster than documentation.

Mitigation
- Treat this glossary as the canonical reference; update it when introducing a new term.
- Prefer explicit naming in code (`workflow_execution_id` vs `execution_id`).

Observability signals (logs/metrics/traces)
- Metrics labels inconsistent across services (same concept, different name).
- Traces missing attributes because services disagree on attribute keys.

### Complexity & Performance
- Glossary usage is intended to make communication **O(1)**: lookup the term instead of debating it.
- Maintaining a concept-to-code map is **O(T)** where T is number of terms; keep it small and high-value.

## Research Extensions

- Auto-generate glossary candidates by mining docstrings, route tags, and database schema comments.

## System Design Deep Dive

- See `system_design_deep_dive.md` for how glossary terms compose into end-to-end flows.

## Future Evolution Strategy

- See `future_evolution_strategy.md` for how terms (and their code anchors) are expected to evolve over time.

