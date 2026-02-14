# CognitionOS V3 API Service

FastAPI-based REST API for the CognitionOS V3 clean architecture.

## Features

- **Clean Architecture**: Domain-Driven Design with dependency inversion
- **Async Support**: Fully asynchronous with SQLAlchemy async
- **OpenAPI Documentation**: Auto-generated Swagger/ReDoc docs
- **Type Safety**: Pydantic v2 for request/response validation
- **Observability**: Structured logging and distributed tracing
- **Production-Ready**: Health checks, error handling, CORS support

## Endpoints

### Workflow Management
- `POST /api/v3/workflows` - Create a new workflow
- `GET /api/v3/workflows/{workflow_id}/{version}` - Get workflow details
- `GET /api/v3/workflows` - List workflows (with pagination)
- `POST /api/v3/workflows/execute` - Execute a workflow
- `GET /api/v3/workflows/executions/{execution_id}` - Get execution status

### Health Check
- `GET /health` - Health check endpoint

## Running Locally

### With Docker Compose (Recommended)
```bash
# From repository root
docker-compose up api-v3
```

The API will be available at http://localhost:8100

### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DB_HOST=localhost
export DB_PORT=5432
export DB_DATABASE=cognitionos
export DB_USERNAME=cognition
export DB_PASSWORD=changeme
export REDIS_HOST=localhost
export REDIS_PORT=6379

# Run the application
cd src
python main.py
```

## API Documentation

Once running, access:
- Swagger UI: http://localhost:8100/docs
- ReDoc: http://localhost:8100/redoc
- OpenAPI Schema: http://localhost:8100/openapi.json

## Configuration

Configuration is managed via environment variables with the `core.config` module.

### Database
- `DB_HOST` - Database host (default: localhost)
- `DB_PORT` - Database port (default: 5432)
- `DB_DATABASE` - Database name (default: cognitionos)
- `DB_USERNAME` - Database username (default: cognition)
- `DB_PASSWORD` - Database password (default: changeme)

### Redis
- `REDIS_HOST` - Redis host (default: localhost)
- `REDIS_PORT` - Redis port (default: 6379)

### RabbitMQ
- `RABBITMQ_HOST` - RabbitMQ host (default: localhost)
- `RABBITMQ_PORT` - RabbitMQ port (default: 5672)

### API
- `API_HOST` - API host (default: 0.0.0.0)
- `API_PORT` - API port (default: 8100)
- `API_DEBUG` - Debug mode (default: false)
- `API_LOG_LEVEL` - Log level (default: info)

### LLM
- `LLM_OPENAI_API_KEY` - OpenAI API key
- `LLM_ANTHROPIC_API_KEY` - Anthropic API key

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Style
```bash
# Format code
black src/

# Lint
pylint src/
```

## Architecture

The API service follows clean architecture principles:

```
services/api/
├── src/
│   ├── main.py              # FastAPI application
│   ├── routes/              # API endpoints
│   │   └── workflows.py
│   ├── schemas/             # Pydantic models
│   │   ├── workflows.py
│   │   └── agents.py
│   ├── dependencies/        # Dependency injection
│   │   └── injection.py
│   ├── middleware/          # Custom middleware
│   └── auth/                # Authentication
├── requirements.txt
├── Dockerfile
└── README.md
```

Dependencies point inward:
- Routes depend on schemas and use cases
- Use cases depend on domain interfaces
- Infrastructure implements domain interfaces
- Domain has zero external dependencies

## Error Handling

The API uses standard HTTP status codes:

- `200 OK` - Successful GET request
- `201 Created` - Successful POST request
- `202 Accepted` - Async operation started
- `400 Bad Request` - Invalid request data
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

All errors return a consistent JSON format:
```json
{
  "error": "ErrorType",
  "message": "Human-readable message",
  "detail": {},
  "trace_id": "abc-123"
}
```

## License

MIT
