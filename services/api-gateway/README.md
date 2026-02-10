# API Gateway

Single entry point for all CognitionOS client requests.

## Features

- **Request Routing**: Routes requests to appropriate microservices
- **Authentication**: Validates JWT tokens with auth service
- **Rate Limiting**: Per-user and per-IP rate limiting
- **Load Balancing**: Distributes load across service instances
- **Circuit Breaker**: Protects against cascading failures
- **Request Tracing**: Distributed tracing with trace IDs
- **WebSocket Support**: Bidirectional communication for real-time updates

## Architecture

```
Client → API Gateway → [Auth Service | Task Service | Agent Orchestrator | ...]
```

## Routes

- `/auth/*` → Auth Service
- `/tasks/*` → Task Planner
- `/agents/*` → Agent Orchestrator
- `/memory/*` → Memory Service
- `/ws` → WebSocket for real-time updates

## Environment Variables

```
API_GATEWAY_PORT=8000
AUTH_SERVICE_URL=http://localhost:8001
TASK_SERVICE_URL=http://localhost:8002
AGENT_SERVICE_URL=http://localhost:8003
MEMORY_SERVICE_URL=http://localhost:8004
RATE_LIMIT_PER_MINUTE=60
CIRCUIT_BREAKER_THRESHOLD=5
REQUEST_TIMEOUT=30
```

## Circuit Breaker

If a downstream service fails 5 times in a row, the circuit opens and requests fast-fail for 60 seconds before attempting recovery.

## Rate Limiting

- Default: 60 requests per minute per IP
- Authenticated users: 120 requests per minute
- Burst allowance: 10 additional requests

## Tech Stack

- FastAPI for routing
- httpx for async HTTP client
- WebSockets for real-time communication
- Redis for rate limiting state
