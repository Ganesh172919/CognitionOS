# Services/API Knowledge Layer (FastAPI)

This folder documents the async API service in `services/api/` with a focus on lifecycle, middleware, dependency injection, and contracts.

Primary anchors:
- App lifecycle: `services/api/src/main.py`
- Routers: `services/api/src/routes/`
- Middleware: `services/api/src/middleware/`
- Dependency injection: `services/api/src/dependencies/injection.py`
- Error handling: `services/api/src/error_handlers.py`
- Observability wiring: `infrastructure/observability/*`

## API service map

### Lifecycle (startup/shutdown)
`services/api/src/main.py` uses a FastAPI lifespan manager:
- Initializes tracing (optional) via `infrastructure/observability/tracing.py`
- Initializes metrics (optional) via `infrastructure/observability/metrics.py`
- Initializes Redis pool via `infrastructure/persistence/redis_pool.py`
- On shutdown: waits `config.api.shutdown_timeout_seconds`, closes DB engine and Redis pool

### Middleware stack (as configured)
```text
RequestIDMiddleware (services/api/src/middleware/request_id.py)
  -> CORS (FastAPI)
  -> GZip (FastAPI)
  -> PrometheusMiddleware (infrastructure/observability/metrics.py) [optional]
  -> Routes + exception handlers
```

### Router overview (examples)

| Router file | Prefix | Notes |
|---|---:|---|
| `services/api/src/routes/workflows.py` | `/api/v3/workflows` | create/get/execute workflow |
| `services/api/src/routes/memory.py` | `/api/v3/memory` | L1/L2/L3 memory operations |
| `services/api/src/routes/auth.py` | `/api/v3/auth` | authentication endpoints |
| `services/api/src/routes/execution_persistence.py` | `/api/v3/executions` | persistence + replay-related endpoints |
| `services/api/src/routes/health.py` | `/api/v3/health` | richer health monitoring (in addition to `/health`) |
| `services/api/src/routes/webhooks.py` | `/webhooks` | inbound integrations (treat as public-facing) |
| `services/api/src/routes/websocket.py` | `/ws` | websocket surface |

## Request lifecycle (high-level)
```text
Client
  -> ASGI server (uvicorn)
  -> Middleware (request_id, cors, gzip, metrics)
  -> Route handler (FastAPI)
  -> Dependency injection (db session, repositories, use cases)
  -> Use case (core/application/*)
  -> Repositories (infrastructure/persistence/*)
  -> Postgres/Redis/RabbitMQ
  -> Response (with X-Request-ID)
```

## Advanced Engineering Notes

### Failure Scenarios: “Graceful shutdown drops in-flight work”
Symptoms
- Requests fail with 5xx during deploys.
- In-flight workflows/steps appear partially persisted.

Root cause
- Shutdown waits a fixed sleep; it doesn’t coordinate draining with worker queues or DB transactions.

Mitigation
- Document and implement (later) coordinated draining:
  - stop accepting new requests
  - wait for in-flight handlers to finish up to a timeout
  - close resources only after draining

Observability signals (logs/metrics/traces)
- Spike in 5xx around deploy timestamps.
- Increased request latency and aborted connections during shutdown.

### Complexity & Performance
- Middleware adds mostly **O(1)** overhead per request; the dominant costs are DB queries, queue operations, and vector searches.
- Beware metrics label cardinality: endpoint labels on raw paths can blow up time series.

## Research Extensions

- Add synthetic “request lifecycle” tracing tests to ensure request_id/trace propagation stays intact.

## System Design Deep Dive

- See `../system_design_deep_dive.md` for end-to-end sequences that traverse API → use cases → persistence.

## Future Evolution Strategy

- See `../future_evolution_strategy.md` for API-facing roadmap: contract discipline, idempotency, backpressure, and mature orchestration.

