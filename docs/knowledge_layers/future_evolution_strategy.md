# Future Evolution Strategy (Roadmap)

This roadmap is intentionally concrete: it assumes the current repository structure (Clean Architecture in `core/`, FastAPI in `services/api/`, persistence in `infrastructure/` + `database/`) and proposes evolution steps without requiring a big-bang rewrite.

## Roadmap (table)

| Horizon | Theme | Concrete deliverables | Primary anchors |
|---|---|---|---|
| Near-term (weeks) | Contract rigor + determinism | canonical error envelope, idempotency keys for side effects, request/trace correlation | `services/api/src/error_handlers.py`, `database/migrations/008_execution_persistence.sql`, `services/api/src/middleware/request_id.py` |
| Mid-term (months) | Orchestration maturity | outbox pattern, durable workflow driver loop, worker autoscaling/backpressure controls | `infrastructure/tasks/*`, `infrastructure/message_broker/*`, DB schema additions |
| Long-term (quarters) | Multi-region & data evolution | multi-region posture, tenant isolation hardening, analytics lakehouse integration | `kubernetes/`, `database/migrations/*`, `frontend/` analytics surfaces |

## Near-term (next 1–3 iterations)

- **API contract discipline**
  - Document and standardize `/api/v3/*` versioning + deprecation policy (see `services_api/api_contracts_versioning.md`).
  - Make error responses consistent with `services/api/src/error_handlers.py` output across routes.
- **Deterministic replay spec (P0)**
  - Treat `step_execution_attempts` (migration `008_execution_persistence.sql`) as the backbone:
    - define idempotency key derivation
    - define what “deterministic” means per step type
    - define “nondeterminism flags” taxonomy (external API, time, random, model variance)
- **Unified DB connection/session strategy**
  - Pick one session pattern to be the “golden path” (today: `services/api/src/dependencies/injection.py`).
  - Document a migration plan away from duplicated engines/sessions.

## Mid-term (orchestration upgrades)

- **Temporal-like semantics (without necessarily adopting Temporal)**
  - Implement a durable driver loop that:
    - reads step states
    - computes ready steps
    - writes “lease/lock” records (see `execution_locks` in migration `008_execution_persistence.sql`)
    - schedules work with bounded concurrency
- **Outbox pattern for eventing**
  - Persist domain events in a DB outbox table inside the same transaction as state updates.
  - Relay to RabbitMQ (or equivalent) asynchronously with dedupe.
- **Backpressure and load shedding**
  - Add admission control at API and worker layers (rate limiting, queue depth limits, concurrency caps).

## Long-term (multi-region posture + data mesh)

- **Multi-region**
  - Define which data is regional (latency-sensitive) vs global (shared config/billing).
  - Decide on replication strategy (logical replication, managed Postgres multi-region, etc.).
- **Analytics / lakehouse integration**
  - Stream execution events and memory lifecycle events to an analytical store.
  - Build retrieval quality dashboards and drift detection.

## Advanced Engineering Notes

### Failure Scenarios: “Big-bang rewrite”
Symptoms
- Long-running branch, frequent merge conflicts, “we’ll fix it after the rewrite” culture.
- Production incidents increase due to partially migrated behavior.

Root cause
- Scope not constrained; contract discipline and safety rails missing.

Mitigation
- Ship evolution behind stable contracts (API versioning, idempotency, deterministic replay).
- Prefer “expand/contract” schema patterns and incremental adapter migrations.

Observability signals (logs/metrics/traces)
- Divergent error envelopes across routes.
- Increased 5xx rates during refactors; trace graphs with missing spans due to new code paths.

### Complexity & Performance
- Incremental evolution reduces organizational “coordination complexity” from ~**O(Teams²)** to ~**O(Teams)** by decoupling via contracts.
- Outbox/eventing adds write amplification (extra inserts) but reduces coupling and enables replay/auditing.

## Research Extensions

- Evaluate deterministic replay under LLM nondeterminism: define acceptable divergence thresholds.
- Explore CRDT-style merges for memory updates across regions (see `research_extensions.md`).

## System Design Deep Dive

- Use `system_design_deep_dive.md` sequences as the baseline. Any roadmap item should specify which sequence changes, and how.

## Future Evolution Strategy

(This document is the future evolution strategy; link to it from addenda and other docs.)

