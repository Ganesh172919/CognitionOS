# Observability (Logs, Metrics, Traces)

Anchors:
- Structured logging: `infrastructure/observability/logging.py`
- Prometheus metrics: `infrastructure/observability/metrics.py`
- OpenTelemetry tracing: `infrastructure/observability/tracing.py`
- Request correlation: `services/api/src/middleware/request_id.py`
- FastAPI wiring: `services/api/src/main.py`

## Correlation strategy (logs ↔ traces ↔ metrics)

Target (documented):
- Every request has a **request_id** (`X-Request-ID`) that appears in logs.
- Every trace has a trace/span context; request_id should also appear as a span attribute.
- DB/queue operations should be linked via shared attributes: `workflow_id`, `execution_id`, `step_id`, `tenant_id`.

## Golden signals (RED/USE)

| Signal | What to measure | Where |
|---|---|---|
| Rate | RPS per endpoint | `http_requests_total` |
| Errors | 4xx/5xx rate | `http_requests_total{status=...}` + error logs |
| Duration | p50/p95/p99 latency | `http_request_duration_seconds` histogram |
| Utilization | DB connections, queue backlog | `db_connections_active`, RabbitMQ exporter, worker concurrency |

## What spans/metrics should exist (workflow + memory)

Table (documented “must haves”):
| Flow | Spans (examples) | Metrics (examples) |
|---|---|---|
| Create workflow | `workflow.create` | `workflows_created_total{workflow_id=...}` |
| Execute workflow | `workflow.execute`, `step.schedule`, `step.execute` | `workflow_executions_total{status=...}`, `workflow_steps_executed_total{...}` |
| Memory store | `memory.store` | counters by tier + size |
| Memory search | `memory.search` + DB vector span | latency histogram by tier; “no results” rate |

## Incident debugging workflow (alert → trace → logs → DB)

```text
Alert fires (latency/error/budget)
  -> identify endpoint/flow from metric labels
  -> grab exemplar request_id/trace_id (if available)
  -> open trace, find slow span (DB? Redis? queue?)
  -> pivot to structured logs using request_id
  -> inspect DB queries / locks / vector index usage
  -> apply mitigation (shed load, reduce concurrency, rollback index build, etc.)
```

## Advanced Engineering Notes

### Failure Scenarios: “High-cardinality metrics explode Prometheus”
Symptoms
- Prometheus memory usage spikes; scraping slows or fails.

Root cause
- Labels include high-cardinality values (raw paths with IDs, user IDs, etc.).

Mitigation
- Normalize endpoint labels (route templates, not raw paths).
- Avoid labeling metrics with unbounded identifiers (request IDs, UUIDs).

Observability signals (logs/metrics/traces)
- Prometheus scrape errors/timeouts; rapidly increasing time series count.
- Metrics server CPU/memory spikes after deployments adding new labels.

### Failure Scenarios: “Missing trace propagation across async boundaries”
Symptoms
- HTTP traces end at the API; worker activity is invisible.

Root cause
- Trace context not injected into queue messages; workers don’t extract context.

Mitigation
- Propagate traceparent/request_id in message headers; store correlation_id in DB (migration `008_execution_persistence.sql`).

Observability signals (logs/metrics/traces)
- Logs show worker activity but traces have gaps; no parent/child relationships.

### Complexity & Performance
- Instrumentation overhead is **O(1)** per operation but can be a significant constant factor at high throughput.
- Prefer sampling for tracing at high volume; keep logs structured but bounded in size.

## Research Extensions

- Add exemplars linking histograms to traces (Prometheus exemplar support).
- Add automated “trace gap” detection: alert when worker spans are missing for executions.

## System Design Deep Dive

- See `../system_design_deep_dive.md` for sequences where correlation must survive across boundaries.

## Future Evolution Strategy

- Near-term: normalize endpoint metric labels; enforce correlation IDs in all critical flows.
- Mid-term: end-to-end tracing across RabbitMQ/Celery and DB outbox relay.

