# Integration Guide

This guide explains how to integrate and run all CognitionOS services together.

## Architecture Overview

```
┌─────────────────┐
│   API Gateway   │  Port 8000 - Main entry point
└────────┬────────┘
         │
    ┌────┴─────┬────────┬─────────┬──────────┬─────────┐
    ▼          ▼        ▼         ▼          ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│  Auth  │ │ Planner│ │ Agent  │ │   AI   │ │  Tool  │ │ Memory │
│ Service│ │ Service│ │  Orch. │ │Runtime │ │ Runner │ │ Service│
│  8001  │ │  8002  │ │  8003  │ │  8005  │ │  8006  │ │  8004  │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘
    │          │          │          │          │          │
    └──────────┴──────────┴──────────┴──────────┴──────────┘
                           │
                   ┌───────┴───────┐
                   ▼               ▼
              ┌──────────┐   ┌─────────┐
              │PostgreSQL│   │  Redis  │
              │+ pgvector│   │  Cache  │
              └──────────┘   └─────────┘
```

## Quick Start (Docker)

### Prerequisites

- Docker 24.0+
- Docker Compose 2.0+
- 4GB+ RAM available

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/CognitionOS.git
   cd CognitionOS
   ```

2. **Set environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start all services**
   ```bash
   docker-compose up -d
   ```

4. **Initialize database**
   ```bash
   docker-compose exec api-gateway python /app/scripts/init_database.py
   ```

5. **Verify services**
   ```bash
   curl http://localhost:8000/health
   ```

## Development Setup (Local)

### Prerequisites

- Python 3.11+
- PostgreSQL 14+ with pgvector
- Redis 7+
- Node.js 18+ (for frontend, optional)

### 1. Database Setup

```bash
# Install PostgreSQL and pgvector
sudo apt-get install postgresql postgresql-contrib postgresql-14-pgvector

# Create database
sudo -u postgres psql
CREATE DATABASE cognitionos;
CREATE USER cognition WITH PASSWORD 'cognition';
GRANT ALL PRIVILEGES ON DATABASE cognitionos TO cognition;
\q

# Set environment variable
export DATABASE_URL="postgresql://cognition:cognition@localhost:5432/cognitionos"

# Install Python dependencies
pip install -r database/requirements.txt

# Run migrations
python database/run_migrations.py init

# Initialize default data
python scripts/init_database.py
```

### 2. Redis Setup

```bash
# Install Redis
sudo apt-get install redis-server

# Start Redis
sudo systemctl start redis
sudo systemctl enable redis

# Verify
redis-cli ping  # Should return "PONG"
```

### 3. Service Setup

Each service can be run independently:

```bash
# API Gateway (Port 8000)
cd services/api-gateway
pip install -r requirements.txt
python src/main.py

# Auth Service (Port 8001)
cd services/auth-service
pip install -r requirements.txt
python src/main.py

# Task Planner (Port 8002)
cd services/task-planner
pip install -r requirements.txt
python src/main.py

# Agent Orchestrator (Port 8003)
cd services/agent-orchestrator
pip install -r requirements.txt
python src/main.py

# Memory Service (Port 8004)
cd services/memory-service
pip install -r requirements.txt
python src/main.py

# AI Runtime (Port 8005)
cd services/ai-runtime
pip install -r requirements.txt
python src/main.py

# Tool Runner (Port 8006)
cd services/tool-runner
pip install -r requirements.txt
python src/main.py
```

## Environment Variables

Create `.env` file in the project root:

```bash
# Database
DATABASE_URL=postgresql://cognition:cognition@localhost:5432/cognitionos

# Redis
REDIS_URL=redis://localhost:6379/0

# LLM APIs
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Service Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
SERVICE_VERSION=0.1.0

# Security
JWT_SECRET=your-secret-key-change-in-production
API_KEY_SALT=your-salt-change-in-production

# Service Ports (optional, defaults shown)
API_GATEWAY_PORT=8000
AUTH_SERVICE_PORT=8001
TASK_PLANNER_PORT=8002
AGENT_ORCHESTRATOR_PORT=8003
MEMORY_SERVICE_PORT=8004
AI_RUNTIME_PORT=8005
TOOL_RUNNER_PORT=8006
```

## Service Integration

### API Gateway Configuration

The API Gateway routes requests to appropriate services:

```python
# services/api-gateway/src/routes.py
SERVICE_URLS = {
    "auth": "http://auth-service:8001",
    "planner": "http://task-planner:8002",
    "orchestrator": "http://agent-orchestrator:8003",
    "memory": "http://memory-service:8004",
    "ai": "http://ai-runtime:8005",
    "tools": "http://tool-runner:8006",
}
```

For local development, change to localhost:

```python
SERVICE_URLS = {
    "auth": "http://localhost:8001",
    "planner": "http://localhost:8002",
    "orchestrator": "http://localhost:8003",
    "memory": "http://localhost:8004",
    "ai": "http://localhost:8005",
    "tools": "http://localhost:8006",
}
```

### Database Integration

All services can access the database using the shared models:

```python
from database import get_db, User, Task, Memory

@app.get("/tasks/{task_id}")
async def get_task(task_id: str, db: AsyncSession = Depends(get_db)):
    task = await db.get(Task, task_id)
    return task
```

### Memory Service Integration

Other services can store and retrieve memories:

```python
import httpx

async def store_memory(user_id: str, content: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://memory-service:8004/memories",
            json={
                "user_id": user_id,
                "content": content,
                "memory_type": "working",
                "scope": "user"
            }
        )
        return response.json()
```

### AI Runtime Integration

Services can make LLM calls through the AI Runtime:

```python
async def call_planner_agent(goal: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://ai-runtime:8005/generate",
            json={
                "agent_role": "planner",
                "prompt": goal,
                "model": "gpt-4-turbo-preview"
            }
        )
        return response.json()
```

## Testing Integration

### 1. Unit Tests

Run tests for each service:

```bash
pytest services/auth-service/tests/
pytest services/task-planner/tests/
pytest services/memory-service/tests/
```

### 2. Integration Tests

Run full integration tests:

```bash
pytest tests/integration/
```

Example integration test:

```python
import pytest
import httpx

@pytest.mark.asyncio
async def test_full_task_workflow():
    """Test complete task execution workflow"""

    # 1. Authenticate
    async with httpx.AsyncClient() as client:
        auth_response = await client.post(
            "http://localhost:8001/login",
            json={"username": "test", "password": "test123"}
        )
        token = auth_response.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Submit task
        task_response = await client.post(
            "http://localhost:8000/tasks",
            headers=headers,
            json={"goal": "Create a user registration form"}
        )
        task_id = task_response.json()["task_id"]

        # 3. Check task status
        status_response = await client.get(
            f"http://localhost:8000/tasks/{task_id}",
            headers=headers
        )
        assert status_response.json()["status"] == "pending"
```

### 3. Load Tests

Use locust for load testing:

```bash
pip install locust
locust -f tests/load/locustfile.py --host http://localhost:8000
```

## Monitoring

### Health Checks

Each service exposes `/health` endpoint:

```bash
# Check all services
for port in 8000 8001 8002 8003 8004 8005 8006; do
    echo "Port $port: $(curl -s http://localhost:$port/health | jq -r .status)"
done
```

### Logs

View logs in Docker:

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api-gateway
docker-compose logs -f memory-service
```

View logs locally:

```bash
tail -f logs/api-gateway.log
tail -f logs/memory-service.log
```

### Database Monitoring

```bash
# Connect to PostgreSQL
psql -U cognition -d cognitionos

# View active connections
SELECT * FROM pg_stat_activity;

# View table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

# View memory usage
SELECT COUNT(*) FROM memories;
SELECT COUNT(*) FROM tasks;
```

## Troubleshooting

### Service won't start

1. Check logs: `docker-compose logs <service-name>`
2. Verify environment variables are set
3. Check port availability: `lsof -i :<port>`
4. Ensure database is running: `docker-compose ps postgres`

### Database connection errors

1. Verify PostgreSQL is running
2. Check DATABASE_URL is correct
3. Ensure pgvector extension is installed
4. Run migrations: `python database/run_migrations.py up`

### Memory not persisting

1. Verify PostgreSQL volume is mounted
2. Check database connection in Memory Service
3. View database logs: `docker-compose logs postgres`

### LLM API errors

1. Verify API keys are set in .env
2. Check API quota/limits
3. View AI Runtime logs
4. Test API key manually:
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

## Performance Optimization

### Database

- **Indexes**: Already created on critical columns
- **Connection Pooling**: Configured (pool_size=10, max_overflow=20)
- **Vector Search**: Tune IVFFlat `lists` parameter based on data size

### Caching

- Use Redis for frequently accessed data
- Cache agent prompts and tool definitions
- Implement short-term memory caching

### Load Balancing

For production:

```yaml
# docker-compose.prod.yml
services:
  api-gateway:
    deploy:
      replicas: 3

  memory-service:
    deploy:
      replicas: 2
```

## Security Considerations

- Change default passwords in production
- Use strong JWT secrets
- Enable HTTPS/TLS
- Implement rate limiting
- Sanitize all user inputs
- Regularly update dependencies
- Use Docker secrets for sensitive data
- Enable PostgreSQL SSL connections

## Production Deployment

### Docker Swarm

```bash
docker swarm init
docker stack deploy -c docker-compose.prod.yml cognitionos
```

### Kubernetes

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/services.yaml
```

### AWS ECS/Fargate

1. Push images to ECR
2. Create ECS cluster
3. Define task definitions
4. Configure load balancer
5. Set up RDS PostgreSQL
6. Configure ElastiCache Redis

## Next Steps

1. Set up CI/CD pipeline
2. Configure monitoring (Prometheus + Grafana)
3. Implement distributed tracing (Jaeger)
4. Add API documentation (Swagger/OpenAPI)
5. Build frontend dashboard
6. Implement user management
7. Add authentication providers (OAuth, SAML)
8. Set up backup strategy
