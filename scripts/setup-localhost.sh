#!/bin/bash

# CognitionOS Localhost Setup Script
# One-command setup for local development

set -e

echo "=========================================="
echo "🏠 CognitionOS Localhost Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
print_status "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_success "Docker and Docker Compose are installed"

# Stop any existing containers
print_status "Stopping any existing CognitionOS containers..."
docker-compose -f docker-compose.local.yml down 2>/dev/null || true
print_success "Cleanup complete"

# Create .env file if it doesn't exist
print_status "Setting up environment variables..."
if [ ! -f .env ]; then
    cp .env.localhost .env
    print_success "Created .env from .env.localhost"
else
    print_warning ".env already exists, skipping..."
fi

# Build Docker images
print_status "Building Docker images (this may take a few minutes)..."
docker-compose -f docker-compose.local.yml build --no-cache

print_success "Docker images built successfully"

# Start services
print_status "Starting services..."
docker-compose -f docker-compose.local.yml up -d postgres redis rabbitmq

# Wait for databases to be ready
print_status "Waiting for databases to be ready..."
sleep 10

# Check PostgreSQL health
print_status "Checking PostgreSQL connection..."
for i in {1..30}; do
    if docker exec cognitionos-postgres-local pg_isready -U cognition_dev -d cognitionos_dev &>/dev/null; then
        print_success "PostgreSQL is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "PostgreSQL failed to start"
        exit 1
    fi
    sleep 1
done

# Check Redis health
print_status "Checking Redis connection..."
for i in {1..30}; do
    if docker exec cognitionos-redis-local redis-cli ping &>/dev/null; then
        print_success "Redis is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "Redis failed to start"
        exit 1
    fi
    sleep 1
done

# Run database migrations (if migration tool exists)
print_status "Running database migrations..."
if [ -d "database/migrations" ]; then
    # Apply migrations directly to PostgreSQL
    for migration in database/migrations/*.sql; do
        if [ -f "$migration" ]; then
            print_status "Applying $(basename $migration)..."
            docker exec -i cognitionos-postgres-local psql -U cognition_dev -d cognitionos_dev < "$migration" 2>/dev/null || print_warning "Migration may have already been applied"
        fi
    done
    print_success "Database migrations complete"
else
    print_warning "No migration directory found, skipping..."
fi

# Start API service
print_status "Starting API service..."
docker-compose -f docker-compose.local.yml up -d api

# Wait for API to be ready
print_status "Waiting for API to be ready..."
for i in {1..60}; do
    if curl -f http://localhost:8100/api/v3/health/live &>/dev/null; then
        print_success "API is ready"
        break
    fi
    if [ $i -eq 60 ]; then
        print_warning "API health check timeout, but continuing..."
        break
    fi
    sleep 1
done

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Services running:"
echo "  • API Server:       http://localhost:8100"
echo "  • API Docs:         http://localhost:8100/docs"
echo "  • Health Check:     http://localhost:8100/api/v3/health/system"
echo "  • PostgreSQL:       localhost:5432"
echo "  • Redis:            localhost:6379"
echo "  • RabbitMQ UI:      http://localhost:15672 (guest/guest)"
echo ""
echo "Quick commands:"
echo "  • View logs:        docker-compose -f docker-compose.local.yml logs -f"
echo "  • Stop services:    docker-compose -f docker-compose.local.yml down"
echo "  • Restart API:      docker-compose -f docker-compose.local.yml restart api"
echo "  • View API logs:    docker-compose -f docker-compose.local.yml logs -f api"
echo ""
echo "Test the API:"
echo "  curl http://localhost:8100/api/v3/health/system"
echo ""
print_success "Happy coding! 🚀"
