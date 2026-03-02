# Schema, Migrations, and Concurrency (Playbooks + Pitfalls)

This is a production-aware deep dive anchored to the migrations and the main persistence adapters.

## Logical model (high-signal entities)

| Area | Tables (examples) | Primary migrations |
|---|---|---|
| Workflows | `workflows` | `002_v3_clean_architecture.sql` |
| Executions | `workflow_executions`, `step_executions` | `002_v3_clean_architecture.sql` |
| Determinism/replay | `step_execution_attempts`, `execution_snapshots`, `replay_sessions`, `execution_locks` | `008_execution_persistence.sql` |
| Memory tiers | `working_memory`, `episodic_memory`, `longterm_memory` | `003_phase3_extended_operation.sql` |
| Auth | `users`, `user_sessions`, `auth_audit_log` | `007_auth_users.sql` |
| Tenancy/billing | `tenants`, `subscriptions`, `usage_records`, `api_keys`, plus `tenant_id` columns | `009_multi_tenancy_billing.sql` |

## Indexing strategy (what to watch)

| Pattern | Index type | Where it appears | Notes |
|---|---|---|---|
| Primary lookup | btree | most tables | keep PK/UK stable; avoid string PK hot-spots |
| Filtering JSON metadata | GIN | `memories` in `005_phase5_v4_evolution.sql` | be careful: GIN can bloat under high churn |
| Vector similarity | IVFFlat / HNSW | memory tables (`003_*`), semantic cache (`005_*`) | tune parameters; monitor recall/latency |
| Time ordering | btree on `created_at` | many tables | consistent ordering requires explicit tie-breakers |

## Concurrency control (Postgres)

### Isolation level guidance (documented)
- Default `READ COMMITTED` is usually fine for OLTP.
- For “exactly once” semantics, rely on **unique constraints + idempotency keys** rather than SERIALIZABLE everywhere.

### Locking pitfalls
Common causes of deadlocks and lock contention:
- Long transactions that mix reads and writes across many rows.
- Schema migrations that take exclusive locks on hot tables.
- Concurrent index builds without planning (`CREATE INDEX` vs `CREATE INDEX CONCURRENTLY`).

Document-only best practices:
- Keep transactions small; avoid user-driven “interactive” transactions.
- Use statement timeouts on expensive queries.
- Prefer expand/contract migrations that avoid table rewrites.

## Migration safety playbook (expand/contract)

### Pre-flight (before running migration)
- Set `statement_timeout` and `lock_timeout` for the migration session.
- Identify hot tables and peak traffic windows.
- Dry-run on a production-like dataset (size matters for index builds).

### Expand (safe changes)
- Add new nullable columns with defaults handled in application code (avoid table rewrite).
- Add new indexes **concurrently** when possible.
- Add new tables without touching hot tables.

### Contract (cleanup)
- Backfill in batches (id-range or time-range).
- Flip reads/writes to the new column/table.
- Drop old columns/indexes after verifying no reads depend on them.

## Failure Scenarios

### Failure Scenarios: “Long-running migration blocks traffic”
Symptoms
- API latency spikes; requests queue waiting on DB locks.
- `ALTER TABLE` or index builds appear in `pg_stat_activity` for long durations.

Root cause
- Migration takes exclusive locks or rewrites a large table during peak traffic.

Mitigation
- Use `CREATE INDEX CONCURRENTLY` and expand/contract patterns.
- Schedule migration windows; split migrations into small steps.

Observability signals (logs/metrics/traces)
- DB: lock wait events; increased deadlocks.
- API: elevated p99 latency; timeouts.
- Traces: DB spans waiting on locks.

### Failure Scenarios: “Deadlocks under concurrent writes”
Symptoms
- Sporadic transaction failures with “deadlock detected”.

Root cause
- Two code paths update rows in different orders; cyclic waits form under load.

Mitigation
- Standardize update order (by table/PK) in repositories.
- Reduce transaction scope; avoid “select then update many tables” patterns.

Observability signals (logs/metrics/traces)
- DB logs: deadlock reports.
- Metrics: increased 5xx + retry counts at application layer.

## Complexity & Performance (query shapes)

Examples (mental models):
- `workflows` by `(id, version)` lookup: **O(log N)** with PK index.
- `workflow_executions` by workflow_id + created_at: **O(log N + K)** with composite indexes.
- Vector search across tiers: ANN lookup + candidate scoring ~**O(k·d)** where k is top-k, d is dimensions.

Hotspot map (where accidental **O(N²)** appears):
- Recomputing readiness across all steps too frequently (driver loop tick rate too high).
- Joining large JSON payload tables without filtering indexes.
- High-cardinality GIN/vector indexes on hot tables without vacuum/analyze discipline.

## Advanced Engineering Notes

### Trade-offs: “Uniqueness constraints vs serializable transactions”
| Approach | Pros | Cons |
|---|---|---|
| Uniqueness + idempotency keys | fast, scalable | requires careful key design |
| SERIALIZABLE everywhere | simpler mental model | high contention, retries, poor tail latency |

## Research Extensions

- Evaluate different IVFFlat/HNSW tuning parameters on production-like datasets and measure recall/latency.
- Build a migration simulator that estimates lock impact from table size and DDL type.

## System Design Deep Dive

- See `../system_design_deep_dive.md` for where concurrency semantics matter (execute workflow, step attempts, memory promotions).

## Future Evolution Strategy

- Near-term: migration guardrails + unify session management.
- Mid-term: event outbox + replay-driven validation of schema evolution.

