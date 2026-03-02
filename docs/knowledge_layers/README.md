# Knowledge Layers (CognitionOS)

This folder is a **research-grade, production-aware** documentation suite that maps CognitionOS concepts to the **actual repository structure** (Clean Architecture/DDD in `core/`, async FastAPI in `services/api/`, Postgres+pgvector in `database/` + `infrastructure/persistence/`, plus Redis/RabbitMQ/Celery/OTel/Prometheus).

**Rules of engagement**
- **Docs-only**: This suite documents patterns, failure modes, and future paths **without changing runtime behavior**.
- **Code-anchored**: Prefer linking to real modules (example: `core/domain/workflow/entities.py`) over generic theory.
- **Production-aware**: Every deep-dive includes performance notes, failure scenarios, and observability signals.

## Navigation Map

| Area | Doc | When to use it |
|---|---|---|
| Repository map | `repository_analysis.md` | “Where does X live?” and dependency-boundary reviews |
| System flows | `system_design_deep_dive.md` | End-to-end workflow/memory execution flows; distributed trade-offs |
| Core overview | `core/README.md` | Domain/application layering rules; bounded contexts overview |
| Workflow DAG | `core/workflow_dag_execution.md` | DAG validation/scheduling, determinism, failure modes, benchmarking |
| Memory + vectors | `core/memory_hierarchy_vector_search.md` | L1/L2/L3 tiering + pgvector index trade-offs + concurrency |
| API overview | `services_api/README.md` | FastAPI lifecycle, DI boundaries, routers, middleware stack |
| Request lifecycle | `services_api/request_lifecycle_middleware.md` | Cancellation/timeouts, request IDs, tracing/logging correlation |
| API contracts | `services_api/api_contracts_versioning.md` | `/api/v3/*` versioning, compat rules, error envelope, idempotency |
| DB overview | `database/README.md` | Migrations, connection/session patterns, operational basics |
| Migrations + locking | `database/schema_migrations_concurrency.md` | Migration safety playbooks + isolation/locking pitfalls |
| Observability | `cross_cutting/observability.md` | Logs/metrics/traces correlation; “alert → trace → logs → DB” workflow |
| Security | `cross_cutting/security_hardening.md` | Threat model extensions; hardening checklists; multi-tenant isolation |
| Performance | `cross_cutting/performance_benchmarking_capacity.md` | Benchmarking methods + capacity planning templates |
| Resilience | `cross_cutting/failure_modes_resilience.md` | Failure catalog + mitigations for DB/Redis/RabbitMQ/async execution |
| Roadmap | `future_evolution_strategy.md` | Concrete near/mid/long-term evolution plan |
| Research | `research_extensions.md` | Experiments/hypotheses: CRDT memory merges, evaluation harness, etc. |
| Glossary | `glossary.md` | Canonical term definitions + concept-to-code mapping |

## How to use during design reviews

1. Start at `repository_analysis.md` to confirm dependency direction and coupling points.
2. For any new flow, sketch it using the sequence templates in `system_design_deep_dive.md`.
3. Validate algorithmic assumptions against:
   - `core/workflow_dag_execution.md` for workflow scheduling/execution semantics
   - `core/memory_hierarchy_vector_search.md` for memory/storage/retrieval semantics
4. Confirm operational guardrails:
   - `cross_cutting/observability.md` (signals + runbooks)
   - `cross_cutting/security_hardening.md` (threats + controls)
   - `cross_cutting/failure_modes_resilience.md` (failure containment)

## How to use during incident response

Suggested workflow (copy/paste into an incident doc):
```text
1) Identify the failing user-facing contract (API route / job / worker)
2) Pull request_id / trace_id from client and search logs (structured logging)
3) Jump from logs → trace → DB queries / queue events
4) Classify failure mode:
   - retry storm? backpressure? lock contention? vector index regression? partial outage?
5) Apply mitigations:
   - shed load, reduce concurrency, increase timeouts (carefully), disable costly features,
     roll back migrations/index builds, pause consumers, drain queues
6) Capture: timeline, blast radius, root cause, permanent fixes (in roadmap doc)
```

## Advanced Engineering Notes

### Conventions (links, code, diagrams)
- **Code references** are written as inline paths (example: `services/api/src/main.py`).
- **Diagrams** are ASCII in fenced blocks.
- **Pseudocode** is illustrative; treat it as a design pattern, not current behavior.

### Failure Scenarios: “Knowledge drift”
Symptoms
- Engineers follow docs but observe different runtime behavior.
- New endpoints exist without documentation; old endpoints behave differently.

Root cause
- Docs updated without being anchored to code changes, or code changed without updating docs.

Mitigation
- Gate major architecture changes on updating the relevant knowledge-layer doc(s).
- Add “doc anchors” (paths and symbols) in PR descriptions when behavior changes.

Observability signals (logs/metrics/traces)
- Increased “unknown”/unexpected error types in `services/api/src/error_handlers.py` logs.
- New route paths appearing in access logs without corresponding doc updates.

### Complexity & Performance
- Using knowledge layers during a review is **O(D)** in number of docs consulted; keep it small by starting with `repository_analysis.md`.
- Repository searching via `rg` is typically **O(N)** in repository size; prefer targeted searches (routes/use_cases/models) over broad scans.

## Research Extensions

See `research_extensions.md` for experimental ideas and evaluation harness concepts.

## System Design Deep Dive

See `system_design_deep_dive.md` for end-to-end sequences and distributed trade-offs.

## Future Evolution Strategy

See `future_evolution_strategy.md` for a concrete roadmap.

