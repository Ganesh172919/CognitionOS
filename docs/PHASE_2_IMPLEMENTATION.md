# CognitionOS Phase 2 Implementation Guide

This document provides a comprehensive guide to the Phase 2 implementation of CognitionOS, focusing on the API layer, LLM integration, async task queue, and observability features.

## Overview

Phase 2 transforms the V3 clean architecture into a production-ready system by adding:

1. **FastAPI REST API** - HTTP endpoints for workflow and agent management
2. **LLM Provider Abstraction** - Multi-provider LLM support with fallback
3. **Async Task Queue** - Celery-based async workflow execution
4. **Structured Logging** - JSON logging with correlation IDs
5. **Configuration Management** - Centralized config with Pydantic v2

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client Applications                   │
│              Web UI │ CLI │ API Clients                  │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────────────────────────────────┐
│              FastAPI V3 API (Port 8100)                │
│    • Workflow Endpoints    • Agent Endpoints           │
│    • Health Checks         • Error Handling            │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────────────────────────────────┐
│            Application Layer (Use Cases)               │
│  • CreateWorkflow    • ExecuteWorkflow                 │
│  • RegisterAgent     • AssignTask                      │
└────────────────────────────┬────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Domain    │    │Infrastructure│    │  Celery     │
│   Entities  │    │  Repositories│    │  Workers    │
│   Services  │    │  Event Bus   │    │  (Async)    │
└─────────────┘    └─────────────┘    └─────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ PostgreSQL  │    │   Redis     │    │  RabbitMQ   │
│  (Data)     │    │  (Cache)    │    │  (Events)   │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Components

### 1. FastAPI V3 API Service

**Location**: `services/api/`

**Features**:
- RESTful endpoints following OpenAPI specification
- Pydantic v2 schema validation
- Dependency injection for use cases
- Health check endpoints
- CORS and compression middleware
- Auto-generated Swagger/ReDoc documentation

**Endpoints**:
```
POST   /api/v3/workflows                    - Create workflow
GET    /api/v3/workflows/{id}/{version}     - Get workflow
GET    /api/v3/workflows                    - List workflows
POST   /api/v3/workflows/execute            - Execute workflow
GET    /api/v3/workflows/executions/{id}    - Get execution status
GET    /health                              - Health check
```

**Configuration**:
```bash
API_HOST=0.0.0.0
API_PORT=8100
API_DEBUG=false
API_LOG_LEVEL=info
API_CORS_ORIGINS=http://localhost:3000,http://localhost:8000
```

### 2. Configuration Management

**Location**: `core/config.py`

**Features**:
- Pydantic v2 Settings with validation
- Environment variable support
- Structured configuration for all components
- Type safety and auto-completion

**Configuration Sections**:
```python
- DatabaseConfig      # PostgreSQL settings
- RedisConfig         # Redis cache/session settings
- RabbitMQConfig      # Message broker settings
- LLMConfig           # LLM provider settings
- CeleryConfig        # Task queue settings
- APIConfig           # API server settings
- SecurityConfig      # Auth/encryption settings
- ObservabilityConfig # Logging/tracing settings
```

**Usage**:
```python
from core.config import get_config

config = get_config()
print(config.database.url)
print(config.llm.default_provider)
```

### 3. LLM Provider Abstraction

**Location**: `infrastructure/llm/provider.py`

**Features**:
- Unified interface for multiple LLM providers
- Automatic fallback on provider failure
- Cost tracking per request
- Health checks for all providers
- Async support

**Supported Providers**:
- OpenAI (GPT-4, GPT-3.5-turbo)
- Anthropic (Claude 3 Opus, Sonnet, Haiku)
- Extensible for local/custom models

**Usage**:
```python
from infrastructure.llm import create_llm_router, LLMRequest

# Create router with fallback
router = create_llm_router(
    openai_api_key="sk-...",
    anthropic_api_key="sk-ant-...",
)

# Generate response
request = LLMRequest(
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"}
    ],
    model="gpt-4",
    temperature=0.7,
)

response = await router.generate(request)
print(response.content)
print(f"Cost: ${response.cost_usd:.4f}")
print(f"Provider: {response.provider}")
```

### 4. Async Task Queue (Celery)

**Location**: `infrastructure/tasks/`

**Features**:
- Redis-backed task queue
- Async workflow execution
- Retry logic with exponential backoff
- Task routing by queue
- Task monitoring and management

**Tasks**:
```python
- execute_workflow_async    # Async workflow execution
- execute_step_async        # Async step execution
- process_workflow_completion  # Post-completion processing
```

**Configuration**:
```bash
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

**Running Workers**:
```bash
# Start workflow worker
celery -A infrastructure.tasks.celery_config worker \
  --queue=workflows \
  --concurrency=4 \
  --loglevel=info

# Start agent worker
celery -A infrastructure.tasks.celery_config worker \
  --queue=agents \
  --concurrency=8 \
  --loglevel=info
```

### 5. Structured Logging

**Location**: `infrastructure/observability/logging.py`

**Features**:
- JSON-formatted logs
- Correlation IDs (trace_id)
- Context-aware logging
- Service metadata in all logs
- Configurable log levels

**Usage**:
```python
from infrastructure.observability import get_logger, set_trace_id

# Set trace ID for request
set_trace_id("req-abc-123")

# Get logger
logger = get_logger(__name__)

# Log with context
logger.info("Workflow created", extra={
    "workflow_id": "wf-123",
    "version": "1.0.0",
})

# Output:
# {
#   "timestamp": "2026-02-14T05:00:00.000Z",
#   "level": "INFO",
#   "service": "cognitionos-v3",
#   "environment": "production",
#   "trace_id": "req-abc-123",
#   "message": "Workflow created",
#   "workflow_id": "wf-123",
#   "version": "1.0.0"
# }
```

## Deployment

### Docker Compose

The simplest way to run the complete stack:

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api-v3

# Stop services
docker-compose down
```

### Manual Deployment

1. **Install Dependencies**:
```bash
cd services/api
pip install -r requirements.txt
```

2. **Set Environment Variables**:
```bash
export DB_HOST=localhost
export DB_PASSWORD=changeme
export REDIS_HOST=localhost
export RABBITMQ_HOST=localhost
export LLM_OPENAI_API_KEY=sk-...
```

3. **Run Database Migrations**:
```bash
cd database
psql -U cognition -d cognitionos -f migrations/002_v3_clean_architecture.sql
```

4. **Start API Server**:
```bash
cd services/api/src
python main.py
```

5. **Start Celery Workers**:
```bash
celery -A infrastructure.tasks.celery_config worker --loglevel=info
```

## API Usage Examples

### Create a Workflow

```bash
curl -X POST http://localhost:8100/api/v3/workflows \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "data-pipeline",
    "version": "1.0.0",
    "name": "Data Processing Pipeline",
    "description": "Extract, transform, load data",
    "steps": [
      {
        "step_id": "extract",
        "name": "Extract Data",
        "agent_capability": "data_extraction",
        "inputs": {"source": "s3://bucket/data.csv"},
        "depends_on": []
      },
      {
        "step_id": "transform",
        "name": "Transform Data",
        "agent_capability": "data_transformation",
        "inputs": {},
        "depends_on": ["extract"]
      }
    ],
    "tags": ["data", "etl"]
  }'
```

### Execute a Workflow

```bash
curl -X POST http://localhost:8100/api/v3/workflows/execute \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "data-pipeline",
    "workflow_version": "1.0.0",
    "inputs": {
      "data_source": "production",
      "batch_size": 1000
    }
  }'
```

### Get Execution Status

```bash
curl http://localhost:8100/api/v3/workflows/executions/{execution_id}
```

## Testing

### Unit Tests

```bash
pytest services/api/tests/unit/ -v
```

### Integration Tests

```bash
pytest services/api/tests/integration/ -v
```

### API Testing with Swagger

1. Navigate to http://localhost:8100/docs
2. Use the interactive Swagger UI to test endpoints
3. All schemas and validation are auto-documented

## Monitoring

### Health Check

```bash
curl http://localhost:8100/health
```

Response:
```json
{
  "status": "healthy",
  "version": "3.0.0",
  "timestamp": "2026-02-14T05:00:00.000Z",
  "database": "healthy",
  "redis": "healthy",
  "rabbitmq": "healthy"
}
```

### Logs

View structured logs:
```bash
# Docker
docker-compose logs -f api-v3

# Local
tail -f /var/log/cognitionos/api.log | jq
```

## Next Steps

### Phase 2B: Event-Driven Architecture
- [ ] Replace in-memory event bus with RabbitMQ
- [ ] Implement event persistence and replay
- [ ] Add event handlers for workflow lifecycle

### Phase 2C: Security
- [ ] JWT authentication middleware
- [ ] RBAC implementation
- [ ] API key management
- [ ] Secrets encryption

### Phase 2D: Observability
- [ ] OpenTelemetry tracing
- [ ] Prometheus metrics
- [ ] Grafana dashboards
- [ ] Alert rules

### Phase 2E: Testing
- [ ] Unit test coverage >95%
- [ ] Integration tests for all endpoints
- [ ] E2E workflow tests
- [ ] Performance benchmarks

## Troubleshooting

### API Service Won't Start

Check database connection:
```bash
psql -U cognition -d cognitionos -h localhost
```

Check environment variables:
```bash
env | grep DB_
env | grep API_
```

### Celery Workers Not Processing Tasks

Check Redis connection:
```bash
redis-cli ping
```

Check worker logs:
```bash
celery -A infrastructure.tasks.celery_config inspect active
celery -A infrastructure.tasks.celery_config inspect stats
```

### LLM Provider Errors

Check API keys:
```bash
env | grep LLM_
```

Test provider health:
```python
from infrastructure.llm import create_llm_router

router = create_llm_router(openai_api_key="sk-...")
health = await router.health_check_all()
print(health)
```

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

MIT
