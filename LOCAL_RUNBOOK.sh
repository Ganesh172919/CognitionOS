#!/bin/bash
# CognitionOS SaaS Platform - Local Development Runbook
# One-command startup for complete development environment

set -e  # Exit on error

echo "═══════════════════════════════════════════════════════════════"
echo "   CognitionOS SaaS Platform - Local Development Startup"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
info "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    error "docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

success "Prerequisites check passed"

# Configuration
export COMPOSE_PROJECT_NAME="cognitionos"
export COMPOSE_FILE="docker-compose.local.yml"

# Clean up any existing containers
info "Cleaning up existing containers..."
docker-compose -f $COMPOSE_FILE down -v 2>/dev/null || true

# Start infrastructure services
info "Starting infrastructure services (PostgreSQL, Redis, RabbitMQ)..."
docker-compose -f $COMPOSE_FILE up -d postgres redis rabbitmq

# Wait for databases to be ready
info "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if docker-compose -f $COMPOSE_FILE exec -T postgres pg_isready -U cognition &> /dev/null; then
        success "PostgreSQL is ready"
        break
    fi
    echo -n "."
    sleep 1
    
    if [ $i -eq 30 ]; then
        error "PostgreSQL failed to start within 30 seconds"
        exit 1
    fi
done

info "Waiting for Redis to be ready..."
for i in {1..30}; do
    if docker-compose -f $COMPOSE_FILE exec -T redis redis-cli ping &> /dev/null; then
        success "Redis is ready"
        break
    fi
    echo -n "."
    sleep 1
    
    if [ $i -eq 30 ]; then
        error "Redis failed to start within 30 seconds"
        exit 1
    fi
done

info "Waiting for RabbitMQ to be ready..."
for i in {1..60}; do
    if docker-compose -f $COMPOSE_FILE exec -T rabbitmq rabbitmqctl status &> /dev/null; then
        success "RabbitMQ is ready"
        break
    fi
    echo -n "."
    sleep 1
    
    if [ $i -eq 60 ]; then
        error "RabbitMQ failed to start within 60 seconds"
        exit 1
    fi
done

# Run database migrations
info "Running database migrations..."
if [ -f "./scripts/run-migrations.sh" ]; then
    ./scripts/run-migrations.sh
else
    warning "Migration script not found. Attempting direct migration..."
    # Run migrations directly
    docker-compose -f $COMPOSE_FILE exec -T postgres psql -U cognition -d cognitionos -f /docker-entrypoint-initdb.d/001_initial_schema.sql 2>/dev/null || true
    docker-compose -f $COMPOSE_FILE exec -T postgres psql -U cognition -d cognitionos -f /docker-entrypoint-initdb.d/009_multi_tenancy_billing.sql 2>/dev/null || true
fi

success "Database migrations completed"

# Start API service
info "Starting API service..."
docker-compose -f $COMPOSE_FILE up -d api

# Wait for API to be ready
info "Waiting for API to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        success "API is ready"
        break
    fi
    echo -n "."
    sleep 1
    
    if [ $i -eq 60 ]; then
        error "API failed to start within 60 seconds"
        docker-compose -f $COMPOSE_FILE logs api
        exit 1
    fi
done

# Create default tenant (if not exists)
info "Creating default tenant..."
curl -s -X POST http://localhost:8100/api/v3/tenants \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Default Tenant",
    "slug": "default",
    "owner_email": "admin@cognitionos.local",
    "subscription_tier": "pro"
  }' > /dev/null 2>&1 || warning "Tenant may already exist"

echo ""
echo "═══════════════════════════════════════════════════════════════"
success "CognitionOS SaaS Platform is now running!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📊 Service Endpoints:"
echo "  • API:              http://localhost:8100"
echo "  • API Docs:         http://localhost:8100/docs"
echo "  • ReDoc:            http://localhost:8100/redoc"
echo "  • Health Check:     http://localhost:8100/health"
echo ""
echo "🗄️  Infrastructure:"
echo "  • PostgreSQL:       localhost:5432"
echo "  • Redis:            localhost:6379"
echo "  • RabbitMQ:         localhost:5672"
echo "  • RabbitMQ Admin:   http://localhost:15672 (guest/guest)"
echo ""
echo "🔧 Management Commands:"
echo "  • View logs:        docker-compose -f $COMPOSE_FILE logs -f"
echo "  • Stop services:    docker-compose -f $COMPOSE_FILE down"
echo "  • Restart API:      docker-compose -f $COMPOSE_FILE restart api"
echo "  • Shell access:     docker-compose -f $COMPOSE_FILE exec api bash"
echo ""
echo "📚 Quick Start Examples:"
echo ""
echo "  # Create a tenant"
echo "  curl -X POST http://localhost:8100/api/v3/tenants \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -H 'X-Tenant-Slug: default' \\"
echo "    -d '{\"name\": \"My Company\", \"slug\": \"mycompany\", \"owner_email\": \"me@company.com\"}'"
echo ""
echo "  # Get subscription"
echo "  curl http://localhost:8100/api/v3/subscriptions/current \\"
echo "    -H 'X-Tenant-Slug: default'"
echo ""
echo "  # List plugins"
echo "  curl http://localhost:8100/api/v3/plugins \\"
echo "    -H 'X-Tenant-Slug: default'"
echo ""
echo "🧪 Run Tests:"
echo "  • Unit tests:       make test-unit"
echo "  • Integration:      make test-integration"
echo "  • All tests:        make test"
echo ""
echo "📖 Documentation:"
echo "  • Architecture:     ./docs/architecture.md"
echo "  • API Docs:         http://localhost:8100/docs"
echo "  • Localhost Guide:  ./LOCALHOST_SETUP.md"
echo ""
success "Ready for development! 🚀"
echo ""
