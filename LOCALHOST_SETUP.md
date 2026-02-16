# 🏠 Localhost Development Setup Guide

Complete guide to running CognitionOS locally for development.

---

## ⚡ Quick Start (30 seconds)

```bash
# One-command setup
./scripts/setup-localhost.sh

# That's it! Your local environment is ready.
```

Visit: **http://localhost:8100/docs** for API documentation

---

## 📋 Prerequisites

- **Docker** (v20.10+)
- **Docker Compose** (v2.0+)
- **8GB RAM** minimum
- **10GB disk space**

### Install Docker

**macOS:**
```bash
brew install --cask docker
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install docker.io docker-compose-plugin
```

**Windows:**
Download from https://docs.docker.com/desktop/install/windows-install/

---

## 🚀 Manual Setup (if needed)

### Step 1: Create Environment File
```bash
cp .env.localhost .env
```

### Step 2: Start Services
```bash
docker-compose -f docker-compose.local.yml up -d
```

### Step 3: Apply Migrations
```bash
# Migrations are automatically applied on first start
# Or manually:
for f in database/migrations/*.sql; do
    docker exec -i cognitionos-postgres-local psql -U cognition_dev -d cognitionos_dev < "$f"
done
```

### Step 4: Verify
```bash
curl http://localhost:8100/api/v3/health/system
```

---

## 🎯 Available Services

| Service | URL | Credentials |
|---------|-----|-------------|
| API Server | http://localhost:8100 | - |
| API Docs | http://localhost:8100/docs | - |
| Health Check | http://localhost:8100/api/v3/health/system | - |
| PostgreSQL | localhost:5432 | cognition_dev / dev_password_local |
| Redis | localhost:6379 | no password |
| RabbitMQ Management | http://localhost:15672 | guest / guest |

---

## 🔧 Common Commands

### Service Management
```bash
# Start all services
docker-compose -f docker-compose.local.yml up -d

# Stop all services
docker-compose -f docker-compose.local.yml down

# Restart API only
docker-compose -f docker-compose.local.yml restart api

# View all logs
docker-compose -f docker-compose.local.yml logs -f

# View API logs only
docker-compose -f docker-compose.local.yml logs -f api
```

### Development
```bash
# Enter API container
docker exec -it cognitionos-api-local bash

# Run tests inside container
docker exec -it cognitionos-api-local pytest

# Access PostgreSQL
docker exec -it cognitionos-postgres-local psql -U cognition_dev -d cognitionos_dev

# Access Redis CLI
docker exec -it cognitionos-redis-local redis-cli
```

### Database Management
```bash
# Backup database
docker exec cognitionos-postgres-local pg_dump -U cognition_dev cognitionos_dev > backup.sql

# Restore database
docker exec -i cognitionos-postgres-local psql -U cognition_dev -d cognitionos_dev < backup.sql

# Reset database (WARNING: deletes all data)
docker-compose -f docker-compose.local.yml down -v
./scripts/setup-localhost.sh
```

---

## 🧪 Testing Locally

### Run Unit Tests
```bash
# Inside container
docker exec -it cognitionos-api-local pytest tests/unit/

# From host (requires Python)
pytest tests/unit/
```

### Run Integration Tests
```bash
docker exec -it cognitionos-api-local pytest tests/integration/
```

### Run All Tests
```bash
docker exec -it cognitionos-api-local pytest
```

### Test Coverage
```bash
docker exec -it cognitionos-api-local pytest --cov=core --cov=infrastructure --cov=services
```

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find process using port 8100
lsof -i :8100

# Kill process
kill -9 <PID>

# Or change port in .env
API_PORT=8200
```

### Database Connection Failed
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Check PostgreSQL logs
docker logs cognitionos-postgres-local

# Recreate database
docker-compose -f docker-compose.local.yml down -v
docker-compose -f docker-compose.local.yml up -d postgres
```

### API Won't Start
```bash
# View detailed logs
docker-compose -f docker-compose.local.yml logs api

# Rebuild image
docker-compose -f docker-compose.local.yml build --no-cache api
docker-compose -f docker-compose.local.yml up -d api
```

### Slow Performance
```bash
# Check Docker resources
docker stats

# Increase Docker memory (Docker Desktop settings)
# Recommended: 4GB RAM minimum

# Clean up unused containers/images
docker system prune -a
```

### Hot-Reload Not Working
```bash
# Ensure volumes are mounted correctly
docker-compose -f docker-compose.local.yml config

# Restart with fresh volumes
docker-compose -f docker-compose.local.yml down
docker-compose -f docker-compose.local.yml up -d
```

---

## 🔥 Quick Test Commands

### Test Auth Endpoints
```bash
# Register user
curl -X POST http://localhost:8100/api/v3/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"testpass123","full_name":"Test User"}'

# Login
curl -X POST http://localhost:8100/api/v3/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"testpass123"}'
```

### Test Health Endpoints
```bash
# System health
curl http://localhost:8100/api/v3/health/system | jq

# Readiness
curl http://localhost:8100/api/v3/health/ready

# Liveness
curl http://localhost:8100/api/v3/health/live
```

### Create Workflow
```bash
# Example workflow creation
curl -X POST http://localhost:8100/api/v3/workflows/create \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{"name":"Test Workflow","description":"Testing localhost"}'
```

---

## 🎓 Development Workflow

### 1. Make Code Changes
Edit files in your IDE. Changes are automatically detected (hot-reload enabled).

### 2. View Changes
API server restarts automatically. Refresh http://localhost:8100/docs

### 3. Run Tests
```bash
docker exec -it cognitionos-api-local pytest tests/unit/
```

### 4. Commit Changes
```bash
git add .
git commit -m "Your changes"
git push
```

---

## 🔐 Security Notes

### Development Environment
- **NOT for production** - uses weak passwords and debug mode
- Database password: `dev_password_local`
- JWT secret: `localhost_dev_secret_key_change_in_production_12345678`
- All ports exposed for debugging

### Before Production
1. Change all passwords
2. Generate strong JWT secrets
3. Disable debug mode
4. Use production docker-compose
5. Enable HTTPS
6. Configure firewall rules

---

## 📊 Performance Benchmarks

Expected local performance:
- **Startup time:** < 30 seconds
- **API response:** < 100ms
- **Memory usage:** ~1.5GB total
- **Database queries:** < 50ms

---

## 🆘 Getting Help

1. **Check logs:**
   ```bash
   docker-compose -f docker-compose.local.yml logs -f
   ```

2. **Check service health:**
   ```bash
   curl http://localhost:8100/api/v3/health/system
   ```

3. **Reset everything:**
   ```bash
   docker-compose -f docker-compose.local.yml down -v
   ./scripts/setup-localhost.sh
   ```

4. **Report issue:**
   - Collect logs
   - Document steps to reproduce
   - Create GitHub issue

---

## ✅ Verification Checklist

After setup, verify:
- [ ] PostgreSQL accessible on port 5432
- [ ] Redis accessible on port 6379
- [ ] RabbitMQ UI accessible at http://localhost:15672
- [ ] API docs accessible at http://localhost:8100/docs
- [ ] Health endpoint returns "healthy"
- [ ] Can register a user
- [ ] Can login and get JWT token
- [ ] API logs visible in real-time
- [ ] Hot-reload working (edit file, see restart)

---

## 🚀 Next Steps

- Read API documentation: http://localhost:8100/docs
- Run integration tests
- Create your first workflow
- Explore monitoring at http://localhost:15672
- Review database schema in PostgreSQL

---

**Happy Development! 🎉**
