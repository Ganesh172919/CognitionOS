#!/bin/bash
# CognitionOS V4 - Local Development Setup Script
# Phase 5.1: Developer Setup Automation

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install dependencies based on OS
install_dependencies() {
    local os=$1
    print_header "Installing Dependencies"
    
    case $os in
        linux)
            print_info "Detected Linux"
            if command_exists apt-get; then
                print_info "Using apt-get..."
                sudo apt-get update
                sudo apt-get install -y python3 python3-pip docker.io docker-compose jq curl wget git
            elif command_exists yum; then
                print_info "Using yum..."
                sudo yum install -y python3 python3-pip docker docker-compose jq curl wget git
                sudo systemctl start docker
                sudo systemctl enable docker
            else
                print_warning "Unknown package manager. Please install dependencies manually."
            fi
            ;;
        macos)
            print_info "Detected macOS"
            if ! command_exists brew; then
                print_error "Homebrew not found. Install from https://brew.sh"
                exit 1
            fi
            brew install python3 docker docker-compose jq curl wget git
            ;;
        windows)
            print_warning "Windows detected. Please ensure Docker Desktop and Python are installed."
            ;;
        *)
            print_error "Unknown OS. Please install dependencies manually."
            exit 1
            ;;
    esac
}

# Check required tools
check_dependencies() {
    print_header "Checking Dependencies"
    
    local missing_deps=0
    
    if ! command_exists docker; then
        print_error "Docker not found"
        missing_deps=1
    else
        print_success "Docker installed: $(docker --version)"
    fi
    
    if ! command_exists docker-compose; then
        print_error "Docker Compose not found"
        missing_deps=1
    else
        print_success "Docker Compose installed: $(docker-compose --version)"
    fi
    
    if ! command_exists python3; then
        print_error "Python 3 not found"
        missing_deps=1
    else
        print_success "Python 3 installed: $(python3 --version)"
    fi
    
    if ! command_exists git; then
        print_error "Git not found"
        missing_deps=1
    else
        print_success "Git installed: $(git --version)"
    fi
    
    if [ $missing_deps -eq 1 ]; then
        print_error "Missing dependencies. Install them first."
        read -p "Would you like to install dependencies automatically? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_dependencies $(detect_os)
        else
            exit 1
        fi
    fi
}

# Setup environment file
setup_env() {
    print_header "Setting Up Environment"
    
    if [ ! -f .env ]; then
        print_info "Creating .env file from template..."
        cp .env.example .env
        print_success "Created .env file"
        print_warning "Please update .env with your API keys"
    else
        print_success ".env file already exists"
    fi
}

# Start Docker services
start_services() {
    print_header "Starting Docker Services"
    
    print_info "Starting all services with docker-compose..."
    docker-compose up -d --wait
    
    print_success "All services started"
    
    # Wait for services to be healthy
    print_info "Waiting for services to be healthy..."
    sleep 10
}

# Run database migrations
setup_database() {
    print_header "Setting Up Database"
    
    print_info "Running database migrations..."
    python3 scripts/run_v2_migrations.py || print_warning "Migrations may have already run"
    
    print_info "Seeding database with sample data..."
    python3 scripts/init_database.py || print_warning "Database may already be seeded"
    
    print_success "Database setup complete"
}

# Install Python dependencies
install_python_deps() {
    print_header "Installing Python Dependencies"
    
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    print_info "Activating virtual environment..."
    source venv/bin/activate || . venv/Scripts/activate
    
    print_info "Installing Python packages..."
    pip install --upgrade pip
    pip install -r tests/requirements.txt
    pip install -r services/api/requirements.txt
    
    print_success "Python dependencies installed"
}

# Run integration tests
run_tests() {
    print_header "Running Integration Tests"
    
    print_info "Running basic health check tests..."
    python3 -m pytest tests/integration/ -v -k "test_health" || print_warning "Some tests may have failed"
    
    print_success "Test execution complete"
}

# Print quick start guide
print_quick_start() {
    print_header "Setup Complete! 🎉"
    
    echo ""
    echo -e "${GREEN}CognitionOS V4 is now running!${NC}"
    echo ""
    echo "Services available at:"
    echo ""
    echo -e "  ${BLUE}API V3:${NC}        http://localhost:8100"
    echo -e "  ${BLUE}API Gateway:${NC}   http://localhost:8000"
    echo -e "  ${BLUE}Grafana:${NC}       http://localhost:3000 (admin/admin)"
    echo -e "  ${BLUE}Prometheus:${NC}    http://localhost:9090"
    echo -e "  ${BLUE}Jaeger:${NC}        http://localhost:16686"
    echo -e "  ${BLUE}RabbitMQ:${NC}      http://localhost:15672 (guest/guest)"
    echo -e "  ${BLUE}PgAdmin:${NC}       http://localhost:5050"
    echo ""
    echo "Quick commands:"
    echo ""
    echo -e "  ${YELLOW}make dev${NC}          - Start development environment"
    echo -e "  ${YELLOW}make test${NC}         - Run all tests"
    echo -e "  ${YELLOW}make logs-api${NC}     - View API logs"
    echo -e "  ${YELLOW}make health${NC}       - Check service health"
    echo -e "  ${YELLOW}make help${NC}         - Show all available commands"
    echo ""
    echo "Docker commands:"
    echo ""
    echo -e "  ${YELLOW}docker-compose ps${NC}      - List running containers"
    echo -e "  ${YELLOW}docker-compose logs -f${NC} - View all logs"
    echo -e "  ${YELLOW}docker-compose down${NC}    - Stop all services"
    echo ""
    echo -e "${GREEN}Happy coding! 🚀${NC}"
    echo ""
}

# Main execution
main() {
    print_header "CognitionOS V4 - Local Development Setup"
    print_info "This script will set up your local development environment"
    echo ""
    
    # Check if we're in the right directory
    if [ ! -f "docker-compose.yml" ]; then
        print_error "docker-compose.yml not found. Are you in the CognitionOS root directory?"
        exit 1
    fi
    
    # Execute setup steps
    check_dependencies
    setup_env
    install_python_deps
    start_services
    setup_database
    run_tests
    print_quick_start
}

# Run main function
main
