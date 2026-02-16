# CognitionOS - Production Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose installed
- Python 3.12+
- PostgreSQL 14+
- Redis 7+
- RabbitMQ 3+

### Environment Setup

1. **Copy environment template:**
```bash
cp .env.example .env
```

2. **Configure required variables:**
Edit `.env` and set:
- `SECURITY_SECRET_KEY` (generate with: `openssl rand -hex 32`)
- `JWT_SECRET` (generate with: `openssl rand -hex 32`)
- `DB_PASSWORD` (strong database password)
- `LLM_OPENAI_API_KEY` (your OpenAI key)
- `LLM_ANTHROPIC_API_KEY` (your Anthropic key)

3. **Validate environment:**
```bash
python scripts/validate_env.py
```

## 📦 Deployment Options

### Option 1: Docker Compose (Recommended for Development/Staging)

**Development:**
```bash
docker-compose up -d
```

**Production:**
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Option 2: Kubernetes (Recommended for Production)

**Apply manifests:**
```bash
kubectl apply -f kubernetes/base/namespace.yaml
kubectl apply -f kubernetes/base/configmap.yaml
kubectl apply -f kubernetes/base/statefulsets.yaml
kubectl apply -f kubernetes/base/api-v3-deployment.yaml
kubectl apply -f kubernetes/base/ingress.yaml
```

**Verify deployment:**
```bash
kubectl get pods -n cognitionos
kubectl get svc -n cognitionos
```

## 🔧 Database Setup

### Run Migrations

```bash
# Using Docker
docker-compose exec api alembic upgrade head

# Or directly
cd database
psql -h localhost -U cognition -d cognitionos -f migrations/001_initial_schema.sql
psql -h localhost -U cognition -d cognitionos -f migrations/002_agent_workflow.sql
psql -h localhost -U cognition -d cognitionos -f migrations/003_phase3_extended_operation.sql
psql -h localhost -U cognition -d cognitionos -f migrations/004_phase4_task_decomposition.sql
psql -h localhost -U cognition -d cognitionos -f migrations/005_phase5_v4_evolution.sql
psql -h localhost -U cognition -d cognitionos -f migrations/006_phase6_intelligence_layer.sql
psql -h localhost -U cognition -d cognitionos -f migrations/007_auth_users.sql
```

## 🏥 Health Checks

### Verify Services

```bash
# API Health
curl http://localhost:8100/api/v3/health/live

# Comprehensive Health
curl http://localhost:8100/api/v3/health/system

# Readiness (Kubernetes)
curl http://localhost:8100/api/v3/health/ready
```

Expected responses:
- `/health/live`: `{"status": "alive"}`
- `/health/system`: Complete health report with all services
- `/health/ready`: `{"status": "ready"}` when all dependencies healthy

## 📊 Monitoring

### Access Monitoring Stack

- **Grafana:** http://localhost:3000 (admin/admin)
- **Prometheus:** http://localhost:9090
- **Jaeger:** http://localhost:16686

### Key Dashboards

1. **System Health** - Service status, latency, errors
2. **LLM Performance** - Cache hits, costs, tokens
3. **Business Metrics** - Workflows, tasks, success rate
4. **Cost Tracking** - Billing, budgets, savings

## 🔐 Security Checklist

### Before Production

- [ ] Change all default passwords
- [ ] Generate strong SECRET_KEY and JWT_SECRET
- [ ] Enable HTTPS/TLS
- [ ] Configure CORS properly
- [ ] Set API_DEBUG=false
- [ ] Review firewall rules
- [ ] Enable rate limiting
- [ ] Set up secrets management (Vault/AWS Secrets Manager)
- [ ] Configure backup strategy
- [ ] Set up log rotation

## 🧪 Testing

### Run Test Suite

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests only
make test-integration

# With coverage
make test-coverage
```

### Performance Testing

```bash
# Load test (requires locust)
locust -f tests/performance/locustfile.py --host=http://localhost:8100
```

## 📈 Scaling

### Horizontal Scaling (Kubernetes)

```bash
# Scale API pods
kubectl scale deployment api-v3 --replicas=5 -n cognitionos

# HPA is configured for auto-scaling 3-10 replicas
```

### Vertical Scaling (Docker Compose)

Edit `docker-compose.prod.yml`:
```yaml
api:
  deploy:
    resources:
      limits:
        cpus: '4'  # Increase CPU
        memory: 4G  # Increase memory
```

## 🔄 CI/CD

### GitHub Actions Pipeline

Automatically triggered on:
- **Push to main:** Production deployment
- **Push to develop:** Staging deployment
- **Pull request:** Full test suite

### Manual Deployment

```bash
# Trigger via GitHub CLI
gh workflow run ci.yml
```

## 📝 Troubleshooting

### Common Issues

**Database Connection Failed:**
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres
```

**Redis Connection Failed:**
```bash
# Verify Redis
redis-cli ping

# Check logs
docker-compose logs redis
```

**API Not Responding:**
```bash
# Check API logs
docker-compose logs api

# Restart API
docker-compose restart api
```

### Debug Mode

```bash
# Enable debug logging
export API_DEBUG=true
export API_LOG_LEVEL=debug

# Restart services
docker-compose restart
```

## 🔙 Rollback Procedures

### Docker Compose

```bash
# Stop current version
docker-compose down

# Pull previous image
docker pull cognitionos/api:previous-tag

# Start with previous version
docker-compose up -d
```

### Kubernetes

```bash
# Rollback deployment
kubectl rollout undo deployment/api-v3 -n cognitionos

# Check rollout status
kubectl rollout status deployment/api-v3 -n cognitionos
```

## 📊 Performance Benchmarks

### Expected Performance

- **Health Check:** <100ms
- **Simple API Call:** <200ms
- **Cache Hit (L1):** <10ms
- **Cache Hit (L3):** <100ms
- **Database Query:** <50ms
- **Concurrent Requests:** 1000+ req/s

### Resource Usage (per instance)

- **CPU:** 0.5-2 cores
- **Memory:** 512MB-2GB
- **Disk:** 10GB minimum
- **Network:** 100Mbps

## 🆘 Support

### Get Help

- **Documentation:** https://github.com/Ganesh172919/CognitionOS
- **Issues:** https://github.com/Ganesh172919/CognitionOS/issues
- **Logs:** Check `docker-compose logs` or `kubectl logs`

### Emergency Contacts

- **On-call:** [Configure PagerDuty/Opsgenie]
- **Slack:** [Configure Slack alerts]
- **Email:** [Configure email notifications]

## 📅 Maintenance

### Regular Tasks

**Daily:**
- Monitor health dashboards
- Review error logs
- Check resource usage

**Weekly:**
- Review performance metrics
- Update dependencies
- Backup verification

**Monthly:**
- Security audit
- Capacity planning
- Performance optimization
- Documentation updates

---

**Last Updated:** 2024-02-16  
**Version:** 4.0  
**Production Ready:** ✅ YES
