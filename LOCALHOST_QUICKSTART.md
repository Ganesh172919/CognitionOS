# CognitionOS Localhost Setup Guide

Complete guide for running CognitionOS on your local development environment.

## Prerequisites

- **Docker & Docker Compose** (v20+ recommended)
- **Python 3.11+** (for development)
- **Git** (for cloning the repository)
- **8GB+ RAM** (for running all services)
- **10GB+ Disk Space** (for Docker images and data)

## Quick Start (5 Minutes)

```bash
# 1. Clone the repository
git clone https://github.com/Ganesh172919/CognitionOS.git
cd CognitionOS

# 2. Validate your environment
python3 scripts/validate_env_config.py

# 3. Validate code syntax
./scripts/validate_localhost.sh

# 4. Start all services
docker-compose up -d

# 5. Check system health
curl http://localhost:8100/health

# 6. View API documentation
open http://localhost:8100/docs
```

## Detailed Setup

### Step 1: Environment Configuration

The `.env.localhost` file is pre-configured with sensible defaults for local development.

**Validate your configuration:**
```bash
python3 scripts/validate_env_config.py
```

**Key configuration sections:**

- **Database**: PostgreSQL connection (default: localhost:5432)
- **Redis**: Cache and session store (default: localhost:6379)
- **RabbitMQ**: Message broker (default: localhost:5672)
- **API Server**: V3 API configuration (default: port 8100)
- **Security**: Secret keys for JWT and encryption
- **LLM Providers**: OpenAI and Anthropic API keys (optional)

**To use real LLM features**, update these variables:
```bash
LLM_OPENAI_API_KEY=sk-your-real-key-here
LLM_ANTHROPIC_API_KEY=sk-ant-your-real-key-here
```

### Step 2: Code Validation

Ensure all Python code is syntactically valid:

```bash
./scripts/validate_localhost.sh
```

This validates:
- ✓ All 352 Python files compile successfully
- ✓ All infrastructure modules import correctly
- ✓ Required directories and files exist
- ✓ API routes are properly configured

### Step 3: Start Services

#### Option A: Start All Services

```bash
docker-compose up -d
```

**Services started:**
- PostgreSQL (port 5432)
- Redis (port 6379)
- RabbitMQ (ports 5672, 15672)
- PgBouncer (port 6432)
- Prometheus (port 9090)
- Grafana (port 3000)
- Jaeger (port 16686)
- PgAdmin (port 5050)
- etcd (port 2379)
- V3 API (port 8100)
- Legacy services (ports 8000-8007)

#### Option B: Start Infrastructure Only

For development without running full services:

```bash
docker-compose up -d postgres redis rabbitmq
```

### Step 4: Verify System Health

**Check V3 API health:**
```bash
curl http://localhost:8100/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "version": "3.2.0",
  "timestamp": "2024-01-16T10:00:00Z",
  "database": "healthy",
  "redis": "healthy",
  "rabbitmq": "healthy"
}
```

**Check all services:**
```bash
docker-compose ps
```

### Step 5: Explore the API

**Interactive API Documentation:**
- Swagger UI: http://localhost:8100/docs
- ReDoc: http://localhost:8100/redoc
- OpenAPI Spec: http://localhost:8100/openapi.json

**Key API Endpoints:**

| Category | Endpoint | Description |
|----------|----------|-------------|
| **Health** | GET /health | System health status |
| **Workflows** | POST /api/v3/workflows | Create workflow |
| **Execution** | POST /api/v3/workflows/execute | Execute workflow |
| **Developer Tools** | POST /api/v3/developer-tools/sdk/generate | Generate SDK |
| **Developer Tools** | POST /api/v3/developer-tools/docs/generate | Generate API docs |
| **Reliability** | POST /api/v3/reliability/chaos/experiments | Create chaos experiment |
| **Workflows** | POST /api/v3/reliability/workflows/register | Register workflow |

## Development Workflow

### Running Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/

# Run specific test file
pytest tests/integration/test_workflows.py

# Run with coverage
pytest --cov=core --cov=infrastructure --cov=services tests/
```

### Code Quality

```bash
# Format code
black . --line-length 100

# Sort imports
isort .

# Lint code
pylint core infrastructure services

# Type checking
mypy core infrastructure services

# Security scan
bandit -r core infrastructure services
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# View migration history
alembic history
```

## Monitoring & Observability

### Grafana Dashboards

Access: http://localhost:3000
- Username: `admin`
- Password: `admin`

**Available Dashboards:**
- System Overview
- API Performance
- Database Metrics
- Cache Hit Rates
- Error Rates

### Prometheus Metrics

Access: http://localhost:9090

**Key Metrics:**
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency
- `database_connections` - Database connection pool
- `cache_hits_total` - Cache hit rate
- `task_execution_duration_seconds` - Task execution time

### Jaeger Tracing

Access: http://localhost:16686

**Features:**
- Distributed request tracing
- Service dependency graph
- Performance bottleneck identification
- Error propagation tracking

### RabbitMQ Management

Access: http://localhost:15672
- Username: `guest`
- Password: `guest`

**Features:**
- Queue monitoring
- Message rates
- Connection management
- Exchange configuration

### PgAdmin Database Management

Access: http://localhost:5050
- Email: `admin@cognitionos.local`
- Password: `admin`

**Pre-configured server:**
- Name: CognitionOS Local
- Host: postgres
- Port: 5432
- Database: cognitionos
- Username: cognition

## Troubleshooting

### Services Won't Start

**Check Docker is running:**
```bash
docker version
```

**Check for port conflicts:**
```bash
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis
lsof -i :8100  # V3 API
```

**View service logs:**
```bash
docker-compose logs -f api-v3
docker-compose logs -f postgres
docker-compose logs -f redis
```

### Database Connection Issues

**Test PostgreSQL connection:**
```bash
psql -h localhost -p 5432 -U cognition -d cognitionos
# Password: changeme (from .env.localhost)
```

**Reset database:**
```bash
docker-compose down -v  # Remove volumes
docker-compose up -d postgres
```

### API Returns 500 Errors

**Check API logs:**
```bash
docker-compose logs -f api-v3
```

**Common issues:**
- Missing database tables → Run migrations: `alembic upgrade head`
- Redis not accessible → Check Redis is running: `docker-compose ps redis`
- Environment variables missing → Validate: `python3 scripts/validate_env_config.py`

### Import Errors

**Reinstall dependencies:**
```bash
pip install -r requirements.txt
```

**Check Python version:**
```bash
python3 --version  # Should be 3.11+
```

### Performance Issues

**Check resource usage:**
```bash
docker stats
```

**Reduce services for development:**
```bash
# Stop non-essential services
docker-compose stop grafana prometheus jaeger pgadmin
```

**Increase Docker resources:**
- Docker Desktop → Preferences → Resources
- Increase RAM to 8GB+
- Increase CPU cores to 4+

## Advanced Configuration

### Custom Ports

Edit `docker-compose.yml` to change port mappings:

```yaml
services:
  api-v3:
    ports:
      - "9100:8100"  # Change from 8100 to 9100
```

### Production-like Setup

Use production docker-compose:

```bash
docker-compose -f docker-compose.prod.yml up -d
```

**Differences:**
- SSL/TLS enabled
- Resource limits enforced
- Read replicas for database
- Multi-instance services
- Production logging levels

### Environment Variables Reference

See `.env.localhost` for all available variables.

**Critical variables:**

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DB_HOST` | PostgreSQL host | localhost | Yes |
| `DB_PORT` | PostgreSQL port | 5432 | Yes |
| `DB_DATABASE` | Database name | cognitionos_dev | Yes |
| `DB_USERNAME` | Database user | cognition_dev | Yes |
| `DB_PASSWORD` | Database password | dev_password_local | Yes |
| `REDIS_HOST` | Redis host | localhost | Yes |
| `API_PORT` | V3 API port | 8100 | Yes |
| `SECURITY_SECRET_KEY` | Encryption secret | (generated) | Yes |
| `LLM_OPENAI_API_KEY` | OpenAI API key | (optional) | No |

## What's New in v3.2.0

### Phase 2 Features ✅
- **SDK Auto-Generator**: Multi-language SDK generation (Python, TypeScript, Go, Java, Ruby)
- **API Documentation Generator**: Automatic docs from code (Markdown, HTML, OpenAPI)
- **Compliance Automation**: SOC2, GDPR, HIPAA, PCI DSS compliance checking
- **CI/CD Pipeline Automation**: Blue-green, canary, rolling deployments
- **Predictive Analytics**: ML-powered anomaly detection and forecasting

### Phase 3 Features ✅
- **Chaos Engineering Framework**: Production resilience testing
- **Advanced Workflow Orchestration**: Complex workflows with branching and compensation
- **Production-grade Code**: 10,200+ lines, zero placeholders
- **Comprehensive Validation**: Automated syntax and environment checking

## Next Steps

1. **Explore the API**: http://localhost:8100/docs
2. **Create your first workflow**: Follow examples in `/examples`
3. **Set up monitoring**: Configure Grafana dashboards
4. **Read the docs**: Check `/docs` directory for detailed guides
5. **Join the community**: GitHub Discussions

## Getting Help

- **Documentation**: `/docs` directory
- **GitHub Issues**: https://github.com/Ganesh172919/CognitionOS/issues
- **GitHub Discussions**: https://github.com/Ganesh172919/CognitionOS/discussions

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built with 🧠 for autonomous AI execution**
