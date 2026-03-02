"""
CognitionOS Deployment Configuration

Docker Compose and infrastructure configuration for
production multi-service deployment.
"""

# ── docker-compose.yml content ──
DOCKER_COMPOSE = """
version: '3.9'

services:
  api:
    build:
      context: .
      dockerfile: services/api/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://cognitionos:${DB_PASSWORD}@postgres:5432/cognitionos
      - REDIS_URL=redis://redis:6379/0
      - COGNITIONOS_WORKERS=4
      - COGNITIONOS_LOG_LEVEL=INFO
      - COGNITIONOS_SECRET_KEY=${SECRET_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 15s
      timeout: 5s
      retries: 3

  worker:
    build:
      context: .
      dockerfile: services/api/Dockerfile
    command: ["python", "-m", "infrastructure.scheduler.task_scheduler"]
    environment:
      - DATABASE_URL=postgresql+asyncpg://cognitionos:${DB_PASSWORD}@postgres:5432/cognitionos
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - api
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1.0'
          memory: 1G

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: cognitionos
      POSTGRES_USER: cognitionos
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U cognitionos"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redisdata:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 3
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./infrastructure/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - promdata:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
    volumes:
      - grafanadata:/var/lib/grafana

volumes:
  pgdata:
  redisdata:
  promdata:
  grafanadata:

networks:
  default:
    name: cognitionos-network
"""

# ── Kubernetes Deployment ──
K8S_DEPLOYMENT = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cognitionos-api
  labels:
    app: cognitionos
    component: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cognitionos
      component: api
  template:
    metadata:
      labels:
        app: cognitionos
        component: api
    spec:
      containers:
      - name: api
        image: cognitionos/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: cognitionos-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: cognitionos-secrets
              key: redis-url
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "2000m"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: cognitionos-api
spec:
  selector:
    app: cognitionos
    component: api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cognitionos-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cognitionos-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""


def write_docker_compose(path: str = "docker-compose.yml"):
    """Write Docker Compose file."""
    with open(path, "w") as f:
        f.write(DOCKER_COMPOSE.strip())


def write_k8s_manifests(path: str = "k8s-deployment.yml"):
    """Write Kubernetes deployment manifests."""
    with open(path, "w") as f:
        f.write(K8S_DEPLOYMENT.strip())


if __name__ == "__main__":
    write_docker_compose()
    write_k8s_manifests()
    print("Deployment configurations written:")
    print("  - docker-compose.yml")
    print("  - k8s-deployment.yml")
