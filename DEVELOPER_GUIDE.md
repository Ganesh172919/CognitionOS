# CognitionOS - Developer Setup & Troubleshooting Guide

## Table of Contents
1. [Quick Setup](#quick-setup)
2. [Prerequisites](#prerequisites)
3. [Environment Configuration](#environment-configuration)
4. [Build & Run](#build--run)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)
7. [Architecture Overview](#architecture-overview)

---

## Quick Setup

The fastest way to get started (< 2 minutes):

```bash
# Clone repository
git clone https://github.com/Ganesh172919/CognitionOS.git
cd CognitionOS

# One-command setup
./scripts/setup-localhost.sh

# Verify
curl http://localhost:8100/health
```

**That's it!** The system is now running with:
- ✅ PostgreSQL (port 5432)
- ✅ Redis (port 6379)
- ✅ RabbitMQ (port 5672, UI on 15672)
- ✅ API Server (port 8100)

---

## Prerequisites

### Required
- **Docker** v20.10+ 
- **Docker Compose** v2.0+
- **8GB RAM** minimum
- **10GB disk space**

### Optional (for development)
- Python 3.11+ (for local development without Docker)
- Git (for version control)

### Install Docker

**macOS:**
```bash
brew install --cask docker
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install docker.io docker-compose-plugin
sudo usermod -aG docker $USER  # Add yourself to docker group
# Log out and back in for group changes to take effect
```

**Windows:**
- Download Docker Desktop from https://docs.docker.com/desktop/install/windows-install/

---

## Environment Configuration

### Using Default Localhost Config

The system comes with `.env.localhost` pre-configured for local development:

```bash
# Use default config
cp .env.localhost .env
```

### Custom Configuration

Edit `.env` to customize:

```ini
# Database
DB_HOST=localhost
DB_PORT=5432
DB_DATABASE=cognitionos_dev
DB_USERNAME=cognition_dev
DB_PASSWORD=dev_password_local

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# RabbitMQ
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USERNAME=guest
RABBITMQ_PASSWORD=guest

# API
API_PORT=8100
API_LOG_LEVEL=debug

# LLM Providers (Optional - for AI features)
LLM_OPENAI_API_KEY=sk-...
LLM_ANTHROPIC_API_KEY=sk-ant-...

# Billing Provider (mock for local, stripe for production)
BILLING_PROVIDER=mock
# STRIPE_API_KEY=sk_test_...  # Uncomment when using Stripe
```

### Validate Configuration

```bash
python3 scripts/validate_environment.py
```

This will check all required environment variables and warn about any issues.

---

## Build & Run

### Option 1: Automated Setup (Recommended)

```bash
./scripts/setup-localhost.sh
```

This script:
1. ✅ Checks Docker installation
2. ✅ Creates `.env` from `.env.localhost`
3. ✅ Builds Docker images
4. ✅ Starts PostgreSQL, Redis, RabbitMQ
5. ✅ Runs database migrations
6. ✅ Starts API server
7. ✅ Performs health checks

### Option 2: Manual Setup

```bash
# Create environment file
cp .env.localhost .env

# Start infrastructure services
docker-compose -f docker-compose.local.yml up -d postgres redis rabbitmq

# Wait for services to be ready (about 10 seconds)
sleep 10

# Run migrations
for f in database/migrations/*.sql; do
    docker exec -i cognitionos-postgres-local psql -U cognition_dev -d cognitionos_dev < "$f"
done

# Start API service
docker-compose -f docker-compose.local.yml up -d api

# Check logs
docker-compose -f docker-compose.local.yml logs -f api
```

### Option 3: Development Mode (Hot-Reload)

For active development with automatic code reloading:

```bash
# Start services
docker-compose -f docker-compose.local.yml up

# In another terminal, make code changes
# API will automatically reload when files change
```

---

## Testing

### Health Check

```bash
# System health
curl http://localhost:8100/health

# Expected response:
{
  "status": "healthy",
  "version": "3.2.0",
  "timestamp": "2024-02-18T02:52:00",
  "database": "healthy",
  "redis": "healthy",
  "rabbitmq": "healthy"
}
```

### API Documentation

Visit http://localhost:8100/docs for interactive API documentation (Swagger UI).

### Run Tests

```bash
# All tests
docker exec -it cognitionos-api-local pytest

# Unit tests only
docker exec -it cognitionos-api-local pytest tests/unit/

# Integration tests only
docker exec -it cognitionos-api-local pytest tests/integration/

# With coverage
docker exec -it cognitionos-api-local pytest --cov=core --cov=infrastructure
```

### Manual API Testing

```bash
# Register user
curl -X POST http://localhost:8100/api/v3/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123","full_name":"Test User"}'

# Login
curl -X POST http://localhost:8100/api/v3/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123"}'

# Create workflow (use token from login)
curl -X POST http://localhost:8100/api/v3/workflows \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"name":"Test Workflow","description":"Testing"}'
```

---

## Troubleshooting

### Issue: Port Already in Use

**Symptom:** Error like `Bind for 0.0.0.0:8100 failed: port is already allocated`

**Solution:**
```bash
# Find process using port
lsof -i :8100  # or :5432, :6379, etc.

# Kill process
kill -9 <PID>

# Or change port in .env
API_PORT=8200
```

### Issue: Database Connection Failed

**Symptom:** API logs show `could not connect to server`

**Solution:**
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Check PostgreSQL logs
docker logs cognitionos-postgres-local

# Restart PostgreSQL
docker-compose -f docker-compose.local.yml restart postgres

# If still failing, recreate database
docker-compose -f docker-compose.local.yml down -v
docker-compose -f docker-compose.local.yml up -d postgres
sleep 10
# Re-run migrations
```

### Issue: API Won't Start

**Symptom:** API container exits immediately or crashes on startup

**Solution:**
```bash
# Check API logs for errors
docker logs cognitionos-api-local

# Common issues:
# 1. Missing dependencies - Rebuild image
docker-compose -f docker-compose.local.yml build --no-cache api

# 2. Import errors - Reinstall package
docker exec -it cognitionos-api-local pip install -e .

# 3. Database not ready - Wait longer or restart
docker-compose -f docker-compose.local.yml restart api
```

### Issue: Tests Failing

**Symptom:** Many tests fail with import errors or database errors

**Solution:**
```bash
# Ensure you're running tests inside container
docker exec -it cognitionos-api-local pytest

# If imports fail, reinstall package
docker exec -it cognitionos-api-local pip install -e .

# Reset test database
docker exec -it cognitionos-postgres-local psql -U cognition_dev -d cognitionos_dev -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"

# Re-run migrations
for f in database/migrations/*.sql; do
    docker exec -i cognitionos-postgres-local psql -U cognition_dev -d cognitionos_dev < "$f"
done
```

### Issue: Hot-Reload Not Working

**Symptom:** Code changes don't trigger API restart

**Solution:**
```bash
# Ensure you're running in non-detached mode
docker-compose -f docker-compose.local.yml up api

# Check volumes are mounted
docker-compose -f docker-compose.local.yml config | grep volumes

# Restart with fresh mount
docker-compose -f docker-compose.local.yml down
docker-compose -f docker-compose.local.yml up
```

### Issue: Out of Memory

**Symptom:** Docker containers crashing or system slow

**Solution:**
```bash
# Check Docker memory usage
docker stats

# Increase Docker memory in Docker Desktop settings
# Recommended: 4GB minimum, 8GB preferred

# Clean up unused resources
docker system prune -a
docker volume prune
```

### Issue: Migration Errors

**Symptom:** Migrations fail or partially apply

**Solution:**
```bash
# Check which migrations have been applied
docker exec -it cognitionos-postgres-local psql -U cognition_dev -d cognitionos_dev -c "SELECT * FROM schema_migrations ORDER BY version;"

# Manually apply specific migration
docker exec -i cognitionos-postgres-local psql -U cognition_dev -d cognitionos_dev < database/migrations/001_initial_schema.sql

# Start fresh (WARNING: deletes all data)
docker-compose -f docker-compose.local.yml down -v
./scripts/setup-localhost.sh
```

### Issue: LLM API Errors

**Symptom:** AI features fail with authentication errors

**Solution:**
```bash
# Check API keys are set
echo $LLM_OPENAI_API_KEY
echo $LLM_ANTHROPIC_API_KEY

# Update .env with valid keys
# Then restart API
docker-compose -f docker-compose.local.yml restart api

# Or use mock LLM for testing (set in code)
```

---

## Architecture Overview

### Service Architecture

```
┌─────────────────────────────────────────┐
│         API Gateway (FastAPI)           │  Port 8100
│  • REST endpoints                       │
│  • WebSocket support                    │
│  • Request routing                      │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
    ▼          ▼          ▼
┌─────────┐ ┌────────┐ ┌──────────┐
│Workflows│ │ Agents │ │  Memory  │
│ Service │ │Service │ │ Service  │
└─────────┘ └────────┘ └──────────┘
    │          │          │
    └──────────┼──────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
    ▼          ▼          ▼
┌──────────┐┌────────┐┌──────────┐
│PostgreSQL││ Redis  ││ RabbitMQ │
│  5432    ││ 6379   ││   5672   │
└──────────┘└────────┘└──────────┘
```

### Key Components

**Core Modules:**
- `core/domain/` - Business logic & entities
- `core/application/` - Use cases & orchestration
- `infrastructure/` - External integrations (DB, cache, LLM)
- `services/api/` - REST API endpoints

**Data Flow:**
1. Request → API Gateway → Request ID Middleware
2. Authentication → JWT validation
3. Route Handler → Use Case → Repository
4. Database / Cache / Message Queue
5. Response → Error Handler → Client

### Package Structure

Since v3.2.0, CognitionOS uses proper Python packaging:
- `pyproject.toml` - Package definition
- `setup.py` - Installation script
- All modules importable as `from core.domain import ...`
- No more `sys.path.insert` hacks!

---

## Development Workflow

### Making Changes

1. **Edit code** in your IDE
2. **Save file** - API auto-reloads (hot-reload)
3. **Test change** - `curl` or visit `/docs`
4. **Run tests** - `docker exec -it cognitionos-api-local pytest`
5. **Commit** - `git add . && git commit`

### Adding a New API Endpoint

1. Create route in `services/api/src/routes/your_route.py`
2. Use custom exceptions from `core/exceptions.py`
3. Add use case in `core/application/`
4. Add tests in `tests/integration/`
5. Update API docs (auto-generated from docstrings)

### Adding a Database Migration

1. Create SQL file: `database/migrations/010_your_migration.sql`
2. Add migration logic
3. Test locally:
   ```bash
   docker exec -i cognitionos-postgres-local psql -U cognition_dev -d cognitionos_dev < database/migrations/010_your_migration.sql
   ```
4. Restart API to apply changes

---

## Security Notes

### Development vs Production

**Development (localhost):**
- ⚠️ Weak passwords OK
- ⚠️ Debug mode enabled
- ⚠️ All ports exposed
- ⚠️ CORS allows all origins

**Production:**
- ✅ Strong passwords required (12+ chars)
- ✅ Debug mode OFF
- ✅ Limited port exposure
- ✅ CORS restricted to known domains
- ✅ HTTPS only
- ✅ Rate limiting enabled

### Before Going to Production

- [ ] Generate strong secrets: `openssl rand -hex 32`
- [ ] Change all default passwords
- [ ] Set `ENVIRONMENT=production` in `.env`
- [ ] Set `DEBUG=false`
- [ ] Use real LLM API keys
- [ ] Use Stripe for billing: `BILLING_PROVIDER=stripe`
- [ ] Enable rate limiting
- [ ] Set up monitoring & alerts
- [ ] Review security checklist in `SECURITY.md`

---

## Performance Optimization

### Docker Build Caching

To speed up builds:
```bash
# Use BuildKit
export DOCKER_BUILDKIT=1

# Build with cache
docker-compose -f docker-compose.local.yml build

# Clean build (when needed)
docker-compose -f docker-compose.local.yml build --no-cache
```

### Database Query Optimization

- Use indexes (defined in migrations)
- Enable query logging: `API_LOG_LEVEL=debug`
- Use Redis cache for frequently accessed data
- Connection pooling is automatic (default: 10 connections)

### Memory Usage

Typical localhost usage:
- PostgreSQL: ~150MB
- Redis: ~50MB
- RabbitMQ: ~100MB
- API: ~200MB
- **Total: ~500MB** (leaves plenty for your work)

---

## Getting Help

### Check Logs

```bash
# All services
docker-compose -f docker-compose.local.yml logs -f

# Specific service
docker-compose -f docker-compose.local.yml logs -f api

# Last 100 lines
docker-compose -f docker-compose.local.yml logs --tail=100
```

### Reset Everything

Nuclear option - start completely fresh:

```bash
# Stop and remove all containers, volumes, images
docker-compose -f docker-compose.local.yml down -v --rmi all

# Clean Docker system
docker system prune -a --volumes

# Re-run setup
./scripts/setup-localhost.sh
```

### Report Issues

1. **Collect logs:**
   ```bash
   docker-compose -f docker-compose.local.yml logs > logs.txt
   ```

2. **Document steps to reproduce**

3. **Include your environment:**
   - OS and version
   - Docker version: `docker --version`
   - Docker Compose version: `docker-compose --version`

4. **Create GitHub issue** with all details

---

## Useful Commands Cheat Sheet

```bash
# Start everything
./scripts/setup-localhost.sh

# Start services
docker-compose -f docker-compose.local.yml up -d

# Stop services
docker-compose -f docker-compose.local.yml down

# Restart API only
docker-compose -f docker-compose.local.yml restart api

# View logs (follow mode)
docker-compose -f docker-compose.local.yml logs -f api

# Enter API container
docker exec -it cognitionos-api-local bash

# Enter PostgreSQL
docker exec -it cognitionos-postgres-local psql -U cognition_dev -d cognitionos_dev

# Enter Redis CLI
docker exec -it cognitionos-redis-local redis-cli

# Run tests
docker exec -it cognitionos-api-local pytest

# Check Docker stats
docker stats

# Clean up
docker-compose -f docker-compose.local.yml down -v
docker system prune -a
```

---

**Happy Coding! 🚀**

For more information:
- API Docs: http://localhost:8100/docs
- Architecture: `README.md`
- Security: `SECURITY.md`
- Deployment: `DEPLOYMENT.md`
