# Database Knowledge Layer (Postgres + pgvector)

This document focuses on the **database subsystem**: schema/migrations, connection/session management patterns, ORM boundaries, and operational posture.

Primary anchors:
- Migrations: `database/migrations/*.sql`
- DB utilities: `database/connection.py`, `database/run_migrations.py`, `database/README.md`
- Persistence adapters: `infrastructure/persistence/*_repository.py`, `infrastructure/persistence/*_models.py`
- API DB session wiring: `services/api/src/dependencies/injection.py`

## DB subsystem overview

```text
FastAPI routes
  -> DI: AsyncSession (services/api/src/dependencies/injection.py)
    -> Repositories (infrastructure/persistence/*)
      -> ORM models (infrastructure/persistence/*_models.py)
        -> Postgres schema (database/migrations/*.sql)
```

### “Source of truth” hierarchy (pragmatic)
1. **Production schema** is defined by `database/migrations/*.sql` (what actually exists in Postgres).
2. ORM models must match the schema (or explicitly document why they don’t yet).
3. Domain entities are independent of the DB.

## Migrations

Key migrations (non-exhaustive, but high-signal):
- `database/migrations/002_v3_clean_architecture.sql`: workflows, executions, step executions, agents, memories_v3, tasks_v3, …
- `database/migrations/003_phase3_extended_operation.sql`: checkpoints + hierarchical memory tables (L1/L2/L3) + ivfflat indexes
- `database/migrations/007_auth_users.sql`: auth tables + audit log
- `database/migrations/008_execution_persistence.sql`: step attempts + snapshots + replay sessions + correlation IDs + locks
- `database/migrations/009_multi_tenancy_billing.sql`: tenants, subscriptions, usage, api_keys, tenant_id columns
- `database/migrations/005_phase5_v4_evolution.sql`: semantic cache + HNSW/IVFFlat index experiments

## Connection and session management (what exists)

There are multiple patterns in-repo:
- `services/api/src/dependencies/injection.py`: creates a singleton async engine and yields `AsyncSession` per request.
- `infrastructure/persistence/base.py`: defines `DatabaseSession` and global init/get helpers.
- `database/connection.py`: defines another async engine/session factory and helpers (plus `init_db()`).

This duplication is an **architecture smell**: it increases the probability of mismatched pool sizing, timeouts, instrumentation, and transactional semantics.

## Operational notes (basics)

Document-only checklist:
- Backups: daily full + WAL archiving (PITR).
- Statement timeouts for endpoints with variable complexity (vector search, analytics).
- Lock timeouts during migrations to avoid production freeze.
- Monitor vacuum/analyze, especially for high-churn memory tables with vector indexes.

## Advanced Engineering Notes

### Failure Scenarios: “Connection pool exhaustion”
Symptoms
- Requests hang, then time out.
- DB shows many idle-in-transaction sessions or max connections reached.

Root cause
- Leaked sessions (not closed), too much concurrency, slow queries, or pool sizing mismatch across services.

Mitigation
- Ensure sessions are always closed (request-scoped async generators).
- Add statement timeouts and reduce concurrency on expensive endpoints.
- Harmonize pool settings across services and remove duplicated engine managers.

Observability signals (logs/metrics/traces)
- Metrics: `db_connections_active` high; `http_requests_in_progress` sustained.
- Traces: long DB spans; queueing before DB execution.
- DB: `pg_stat_activity` shows many waiting sessions / locks.

### Complexity & Performance
- Most OLTP queries should be **O(log N)** with proper indexes.
- Vector search cost is dominated by candidate evaluation: **O(k·d)** for top-k with d-dimensional vectors (plus ANN index overhead).

## Research Extensions

- Automated schema/ORM drift detector: parse migrations + reflect ORM metadata; fail CI on mismatch.
- Experiment with partitioning strategies for `working_memory` and other high-churn tables.

## System Design Deep Dive

- See `../system_design_deep_dive.md` for sequences that exercise the DB (workflow execute, memory search, replay).

## Future Evolution Strategy

- Near-term: unify DB session management; add migration safety guardrails.
- Mid-term: outbox pattern for eventing; reduce cross-service coupling via durable DB contracts.

