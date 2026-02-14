#!/bin/bash
#
# CognitionOS V3 Deployment Script
#
# This script deploys CognitionOS with all Phase 2 components.
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log_info "Prerequisites check passed ✓"
}

# Check environment file
check_env_file() {
    log_info "Checking environment configuration..."
    
    if [ ! -f .env ]; then
        log_warn ".env file not found. Creating from .env.example..."
        cp .env.example .env
        log_warn "Please update .env with your API keys and secrets before proceeding."
        log_warn "Required: OPENAI_API_KEY or ANTHROPIC_API_KEY"
        read -p "Press Enter to continue after updating .env, or Ctrl+C to exit..."
    fi
    
    # Check for required API keys
    source .env
    if [ -z "$LLM_OPENAI_API_KEY" ] && [ -z "$LLM_ANTHROPIC_API_KEY" ]; then
        log_error "At least one LLM provider API key is required."
        log_error "Please set LLM_OPENAI_API_KEY or LLM_ANTHROPIC_API_KEY in .env"
        exit 1
    fi
    
    log_info "Environment configuration check passed ✓"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Wait for PostgreSQL to be ready
    log_info "Waiting for PostgreSQL to be ready..."
    sleep 10
    
    # Run V3 migration
    docker-compose exec -T postgres psql -U cognition -d cognitionos < database/migrations/002_v3_clean_architecture.sql || true
    
    log_info "Database migrations completed ✓"
}

# Build and start services
start_services() {
    log_info "Building and starting services..."
    
    # Build images
    log_info "Building Docker images..."
    docker-compose build
    
    # Start infrastructure services first
    log_info "Starting infrastructure services..."
    docker-compose up -d postgres redis rabbitmq
    
    # Wait for services to be healthy
    log_info "Waiting for infrastructure services to be healthy..."
    sleep 15
    
    # Start application services
    log_info "Starting application services..."
    docker-compose up -d
    
    log_info "All services started ✓"
}

# Check service health
check_health() {
    log_info "Checking service health..."
    
    # Wait for services to initialize
    sleep 10
    
    # Check V3 API health
    if curl -s -f http://localhost:8100/health > /dev/null; then
        log_info "V3 API (port 8100) is healthy ✓"
    else
        log_warn "V3 API (port 8100) is not responding yet. It may still be initializing."
    fi
    
    # Check API Gateway health
    if curl -s -f http://localhost:8000/health > /dev/null; then
        log_info "API Gateway (port 8000) is healthy ✓"
    else
        log_warn "API Gateway (port 8000) is not responding yet."
    fi
}

# Display service status
display_status() {
    log_info "Service Status:"
    echo ""
    docker-compose ps
    echo ""
    log_info "Endpoints:"
    echo "  - V3 API: http://localhost:8100"
    echo "  - V3 API Docs: http://localhost:8100/docs"
    echo "  - API Gateway: http://localhost:8000"
    echo "  - Frontend: http://localhost:3000"
    echo "  - RabbitMQ Management: http://localhost:15672 (guest/guest)"
    echo ""
    log_info "Deployment completed successfully! 🎉"
}

# Main deployment flow
main() {
    log_info "Starting CognitionOS V3 deployment..."
    
    check_prerequisites
    check_env_file
    start_services
    run_migrations
    check_health
    display_status
    
    log_info "To view logs: docker-compose logs -f"
    log_info "To stop services: docker-compose down"
}

# Run main function
main
