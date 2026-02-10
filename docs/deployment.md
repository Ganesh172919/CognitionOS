# Deployment Guide

## Overview

This guide covers deploying CognitionOS in various environments: local development, staging, and production.

## Prerequisites

- Docker 24.0+
- Docker Compose 2.20+
- Kubernetes 1.28+ (for production)
- PostgreSQL 15+
- Redis 7+
- RabbitMQ 3.12+

## Local Development

### Using Docker Compose

1. **Clone repository**
```bash
git clone https://github.com/Ganesh172919/CognitionOS.git
cd CognitionOS
```

2. **Create environment file**
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Start services**
```bash
docker-compose up -d
```

4. **Verify health**
```bash
curl http://localhost:8000/health  # API Gateway
curl http://localhost:8001/health  # Auth Service
curl http://localhost:8002/health  # Task Planner
```

### Manual Setup (Without Docker)

1. **Install dependencies**
```bash
# Install Python 3.11
# Install PostgreSQL, Redis, RabbitMQ

# Install service dependencies
cd services/api-gateway
pip install -r requirements.txt

# Repeat for each service
```

2. **Set up database**
```bash
createdb cognitionos
psql cognitionos < schema.sql
```

3. **Start services**
```bash
# Terminal 1
cd services/auth-service
python src/main.py

# Terminal 2
cd services/api-gateway
python src/main.py

# Continue for each service...
```

## Environment Variables

### Required Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/cognitionos

# Redis
REDIS_URL=redis://localhost:6379/0

# Message Queue
MESSAGE_QUEUE_URL=amqp://guest:guest@localhost:5672/

# Security
SECRET_KEY=<generate-random-secret>
JWT_SECRET=<generate-random-secret>

# AI Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Services
AUTH_SERVICE_URL=http://localhost:8001
TASK_SERVICE_URL=http://localhost:8002
AGENT_SERVICE_URL=http://localhost:8003
```

### Optional Variables

```bash
# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60

# Circuit Breaker
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT=60

# Sandboxing
SANDBOX_ENABLED=true
SANDBOX_NETWORK_ENABLED=false
SANDBOX_MEMORY_LIMIT=512m
```

## Docker Compose

### Development Configuration

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: cognitionos
      POSTGRES_USER: cognition
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  rabbitmq:
    image: rabbitmq:3.12-management
    ports:
      - "5672:5672"
      - "15672:15672"

  api-gateway:
    build:
      context: .
      dockerfile: services/api-gateway/Dockerfile
    environment:
      - DATABASE_URL=postgresql://cognition:${DB_PASSWORD}@postgres:5432/cognitionos
      - REDIS_URL=redis://redis:6379/0
      - AUTH_SERVICE_URL=http://auth-service:8001
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis

  auth-service:
    build:
      context: .
      dockerfile: services/auth-service/Dockerfile
    environment:
      - DATABASE_URL=postgresql://cognition:${DB_PASSWORD}@postgres:5432/cognitionos
      - REDIS_URL=redis://redis:6379/0
      - JWT_SECRET=${JWT_SECRET}
    ports:
      - "8001:8001"
    depends_on:
      - postgres
      - redis

volumes:
  postgres_data:
```

## Kubernetes Deployment

### Architecture

```
┌─────────────────────────────────────┐
│         Load Balancer (Ingress)     │
└─────────────────┬───────────────────┘
                  │
      ┌───────────┴───────────┐
      │                       │
┌─────▼─────┐         ┌───────▼───────┐
│API Gateway│         │  Auth Service │
│ (3 pods)  │         │   (2 pods)    │
└─────┬─────┘         └───────┬───────┘
      │                       │
      └───────────┬───────────┘
                  │
      ┌───────────┴───────────┐
      │                       │
┌─────▼────────┐      ┌───────▼──────────┐
│Task Planner  │      │Agent Orchestrator│
│  (2 pods)    │      │    (3 pods)      │
└──────────────┘      └──────────────────┘
```

### Deployment Files

**api-gateway-deployment.yaml**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
      - name: api-gateway
        image: cognitionos/api-gateway:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: api-gateway
spec:
  selector:
    app: api-gateway
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Applying Configurations

```bash
# Create namespace
kubectl create namespace cognitionos

# Apply secrets
kubectl apply -f k8s/secrets.yaml -n cognitionos

# Deploy services
kubectl apply -f k8s/ -n cognitionos

# Verify deployments
kubectl get pods -n cognitionos
kubectl get services -n cognitionos
```

## Database Migrations

### Using Alembic

```bash
# Create migration
alembic revision --autogenerate -m "Initial schema"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Monitoring

### Prometheus + Grafana

1. **Install Prometheus**
```bash
helm install prometheus prometheus-community/prometheus
```

2. **Install Grafana**
```bash
helm install grafana grafana/grafana
```

3. **Configure dashboards**
- Import CognitionOS dashboard: `dashboards/cognitionos.json`

### Key Metrics

- Request rate (requests/sec)
- Response latency (p50, p95, p99)
- Error rate
- Active agents
- Task queue depth
- LLM token usage
- Cost per hour

## Scaling

### Horizontal Scaling

```bash
# Scale API Gateway
kubectl scale deployment api-gateway --replicas=5

# Scale Agent Orchestrator
kubectl scale deployment agent-orchestrator --replicas=10
```

### Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-gateway-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-gateway
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Backup & Recovery

### Database Backups

```bash
# Daily backup
pg_dump -h localhost -U cognition cognitionos | gzip > backup-$(date +%Y%m%d).sql.gz

# Restore
gunzip -c backup-20260209.sql.gz | psql -h localhost -U cognition cognitionos
```

### Redis Backups

```bash
# Enable AOF persistence
redis-cli CONFIG SET appendonly yes

# Manual snapshot
redis-cli BGSAVE
```

## Security Hardening

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-gateway-policy
spec:
  podSelector:
    matchLabels:
      app: api-gateway
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector: {}
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: auth-service
```

### TLS Configuration

```bash
# Generate TLS certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt

# Create Kubernetes secret
kubectl create secret tls cognitionos-tls \
  --cert=tls.crt --key=tls.key
```

## Troubleshooting

### Common Issues

1. **Service won't start**
```bash
# Check logs
kubectl logs -f pod/api-gateway-xxx

# Check events
kubectl describe pod api-gateway-xxx
```

2. **Database connection failed**
```bash
# Verify connection
psql $DATABASE_URL

# Check network policy
kubectl get networkpolicy
```

3. **High memory usage**
```bash
# Check resource usage
kubectl top pods

# Adjust limits in deployment
```

## Production Checklist

- [ ] HTTPS enabled
- [ ] Database backups configured
- [ ] Monitoring and alerting set up
- [ ] Log aggregation configured
- [ ] Secrets stored in vault (not environment)
- [ ] Rate limiting enabled
- [ ] CORS properly configured
- [ ] Health checks configured
- [ ] Auto-scaling enabled
- [ ] Disaster recovery plan documented
- [ ] Security scan passed
- [ ] Load testing completed
