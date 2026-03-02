# Failure Modes & Resilience (Catalog + Mitigations)

This document catalogs realistic failure modes for the current architecture (FastAPI + Postgres/pgvector + Redis + RabbitMQ/Celery + OTel/Prometheus).

## Failure catalog (table)

| Failure mode | Symptoms | Likely root cause | Mitigations |
|---|---|---|---|
| DB failover / restart | bursts of 5xx, retries | connection drops; pool invalid | pool_pre_ping; retry with jitter; fast fail for write-heavy endpoints |
| Redis eviction | cache misses spike; latency up | maxmemory policy, hot keys | set maxmemory + policy; fallback to DB; monitor evictions |
| RabbitMQ backlog | delayed executions | consumers slow/crashed; producer overload | backpressure; consumer autoscale; DLQ for poison messages |
| Retry storm | cascading latency + errors | retries without jitter; no circuit breaking | bounded retries + jitter + bulkheads + load shedding |
| Clock skew | token/session issues; ordering bugs | NTP drift; multi-node skew | enforce NTP; avoid time-based ordering without tie-breakers |
| Vector index regression | p95 search latency rises | bloat; parameter mismatch | vacuum/analyze; reindex; tune HNSW/IVFFlat |

## Recovery patterns (documented)

### Bulkheads
- Separate pools/queues for expensive operations (vector search, long workflow steps).
- Enforce per-tenant concurrency limits.

### Circuit breakers
- Persist breaker state if needed (migration `005_phase5_v4_evolution.sql` includes circuit breaker tables).
- Open breakers on repeated downstream failures to prevent stampedes.

### Backpressure and load shedding
- Reject early when saturated rather than queue unbounded.
- Return retryable errors with backoff guidance.

## Advanced Engineering Notes

### Failure Scenarios: “Poison messages and duplicate deliveries”
Symptoms
- Same step fails repeatedly; queue backlog grows; duplicates appear.

Root cause
- A message always fails due to data/contract bug; at-least-once delivery retries without quarantine.

Mitigation
- Add DLQ / quarantine queue after N failures.
- Persist attempt records with idempotency keys (migration `008_execution_persistence.sql`) to avoid duplicate side effects.

Observability signals (logs/metrics/traces)
- Metrics: `events_failed_total` spikes; increasing queue depth.
- Logs: repeated failures with same payload signature.

### Complexity & Performance
- Retries can amplify load exponentially if unbounded; treat retry cost as **O(r)** where r is retries per request, but r can explode under storms.
- Backpressure reduces effective work and improves tail latency by controlling concurrency.

## Research Extensions

- Chaos testing: inject DB restarts, Redis failures, queue pauses; verify system degrades gracefully.
- “Failure fingerprints”: cluster incidents by signal patterns (metrics + logs) to speed up diagnosis.

## System Design Deep Dive

- See `../system_design_deep_dive.md` for end-to-end flows where resilience patterns must be applied.

## Future Evolution Strategy

- Near-term: idempotency + bounded retries + normalized error contracts.
- Mid-term: outbox + durable orchestration + DLQ patterns.
- Long-term: multi-region failover posture and consistent tenancy isolation under partial outages.

