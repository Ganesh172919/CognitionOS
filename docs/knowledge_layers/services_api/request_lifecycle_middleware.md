# Request Lifecycle & Middleware (Correlation, Cancellation, Shutdown)

Anchors:
- Middleware: `services/api/src/middleware/request_id.py`
- Logging context: `infrastructure/observability/logging.py`
- Tracing + instrumentation: `infrastructure/observability/tracing.py`
- Metrics middleware: `infrastructure/observability/metrics.py`
- Lifespan + shutdown: `services/api/src/main.py`

## Request ID propagation and logging correlation

`RequestIDMiddleware`:
- Accepts inbound `X-Request-ID` or generates a UUID.
- Stores it on `request.state.request_id`.
- Sets it into a `contextvars` slot via `set_trace_id(...)` (structured logs pick it up).
- Echoes it back on the response header.

Diagram
```text
X-Request-ID (client) ─┐
                        v
RequestIDMiddleware -> contextvar(trace_id) -> JSON logs include trace_id
                        |
                        └-> response header X-Request-ID
```

Production-aware notes (document-only):
- Consider explicitly mapping request_id into OpenTelemetry span attributes (and vice versa) so logs ↔ traces correlate reliably.
- For async tasks (Celery), propagate correlation IDs explicitly via message headers and persist them (see `correlation_id` additions in `database/migrations/008_execution_persistence.sql`).

## Timeouts, cancellation, and asyncio/ASGI realities

Key ASGI constraints:
- Client disconnect does not automatically cancel all downstream work unless you check disconnect/cancellation and propagate it.
- Long-running handlers should:
  - bound work via timeouts
  - avoid blocking calls
  - externalize work to background workers where appropriate

Documented patterns:
- Wrap downstream calls with `asyncio.timeout(...)` (Python 3.11+) or equivalent.
- Ensure DB operations respect statement timeouts (DB-level) to prevent “stuck” requests.

## Graceful shutdown semantics

Current behavior:
- `services/api/src/main.py` sleeps for `shutdown_timeout_seconds` and then closes DB/Redis resources.

Document-only improvements:
- Stop accepting new requests (readiness endpoint fails) before shutdown.
- Drain in-flight requests with a bounded timeout.
- Coordinate with workers (pause consumers) for safe deploys.

## Rate limiting and payload size (documented patterns)

Even if not implemented in code today, document as “recommended defaults”:
- Per-tenant and per-api-key request quotas.
- Max payload sizes for endpoints that accept embeddings or large documents.
- Separate “expensive” endpoints (vector search) into lower rate limits.

Trade-off table
| Control | Pro | Con | Where to enforce |
|---|---|---|---|
| API gateway rate limits | protects upstream quickly | needs shared state | edge/gateway |
| App-level rate limits | contextual (tenant-aware) | adds per-request overhead | FastAPI middleware |
| DB statement timeouts | prevents runaway queries | can surface as 5xx if too strict | Postgres session/config |

## Advanced Engineering Notes

### Failure Scenarios: “Thundering herd on hot endpoints”
Symptoms
- Sudden spike in concurrent requests; DB pool saturates.
- p99 latency explodes; timeouts cascade.

Root cause
- No admission control; clients retry simultaneously; expensive vector searches hit at once.

Mitigation
- Enforce concurrency limits and rate limits at the edge and/or app.
- Add caching for repeated queries where safe.
- Use backpressure (reject fast) instead of queueing unboundedly.

Observability signals (logs/metrics/traces)
- Metrics: `http_requests_in_progress` sustained high; DB connection gauges maxed.
- Traces: long DB spans; many requests waiting before query execution.

### Failure Scenarios: “Slow client / large payload”
Symptoms
- Many connections stay open; worker threads/event loop starve.

Root cause
- Large request bodies or slow uploads; no payload limits; no timeouts.

Mitigation
- Enforce max body size; set server/read timeouts; stream uploads if needed.

Observability signals (logs/metrics/traces)
- High open connections; request durations dominated by receive time.

### Complexity & Performance
- Request ID middleware is **O(1)** per request (UUID generation + header copy).
- Structured logging overhead is constant factor; ensure JSON logs are not excessively large.

## Research Extensions

- Experiment: use traceparent propagation + request_id bridging to unify correlation across HTTP + async workers.

## System Design Deep Dive

- See `../system_design_deep_dive.md` for sequences where request IDs should flow across boundaries (HTTP → queue → workers → DB).

## Future Evolution Strategy

- Near-term: unify correlation_id semantics (request_id vs trace_id vs DB correlation_id).
- Mid-term: standardize timeouts and cancellation propagation across all downstream adapters.

