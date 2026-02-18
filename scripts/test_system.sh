#!/bin/bash

# CognitionOS System Integration Test
# Tests the complete system setup and functionality

# Don't exit on first error - we want to run all tests
# set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[TEST]${NC} $1"; }
print_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[FAIL]${NC} $1"; }

FAILED=0
PASSED=0
WARNINGS=0

echo "=========================================="
echo "🧪 CognitionOS System Integration Test"
echo "=========================================="
echo ""

# Test 1: Docker availability
print_status "Checking Docker availability..."
if command -v docker &> /dev/null; then
    print_success "Docker is installed"
    ((PASSED++))
else
    print_error "Docker is not installed"
    ((FAILED++))
    exit 1
fi

# Test 2: Docker Compose availability
print_status "Checking Docker Compose..."
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    print_success "Docker Compose is available"
    ((PASSED++))
else
    print_error "Docker Compose is not available"
    ((FAILED++))
    exit 1
fi

# Test 3: Environment file exists
print_status "Checking environment configuration..."
if [ -f .env ] || [ -f .env.localhost ]; then
    print_success "Environment configuration found"
    ((PASSED++))
else
    print_warning "No .env file found, will be created"
    ((WARNINGS++))
fi

# Test 4: Package structure
print_status "Checking package structure..."
if [ -f pyproject.toml ] && [ -f setup.py ]; then
    print_success "Package structure is correct (pyproject.toml + setup.py)"
    ((PASSED++))
else
    print_error "Package structure is incomplete"
    ((FAILED++))
fi

# Test 5: Critical directories exist
print_status "Checking project structure..."
REQUIRED_DIRS=("core" "infrastructure" "services" "database/migrations")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        print_success "  $dir/ exists"
    else
        print_error "  $dir/ missing"
        ((FAILED++))
    fi
done
((PASSED++))

# Test 6: Migrations exist
print_status "Checking database migrations..."
MIGRATION_COUNT=$(ls -1 database/migrations/*.sql 2>/dev/null | wc -l)
if [ "$MIGRATION_COUNT" -ge 9 ]; then
    print_success "Found $MIGRATION_COUNT migrations"
    ((PASSED++))
else
    print_error "Expected at least 9 migrations, found $MIGRATION_COUNT"
    ((FAILED++))
fi

# Test 7: Key Python files exist
print_status "Checking key modules..."
KEY_FILES=(
    "core/exceptions.py"
    "core/config.py"
    "services/api/src/main.py"
    "services/api/src/error_handlers.py"
    "services/api/src/middleware/request_id.py"
    "services/api/src/dependencies/injection.py"
)
for file in "${KEY_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "  $file exists"
    else
        print_error "  $file missing"
        ((FAILED++))
    fi
done
((PASSED++))

# Test 8: Scripts are executable
print_status "Checking scripts..."
SCRIPTS=(
    "scripts/setup-localhost.sh"
    "scripts/validate_environment.py"
    "scripts/fix_imports.py"
)
for script in "${SCRIPTS[@]}"; do
    if [ -x "$script" ]; then
        print_success "  $script is executable"
    elif [ -f "$script" ]; then
        print_warning "  $script exists but not executable"
        chmod +x "$script"
        print_success "  Made $script executable"
    else
        print_error "  $script missing"
        ((FAILED++))
    fi
done
((PASSED++))

# Test 9: Docker compose config is valid
print_status "Validating docker-compose configuration..."
if docker-compose -f docker-compose.local.yml config > /dev/null 2>&1 || docker compose -f docker-compose.local.yml config > /dev/null 2>&1; then
    print_success "docker-compose.local.yml is valid"
    ((PASSED++))
else
    print_error "docker-compose.local.yml has errors"
    ((FAILED++))
fi

# Test 10: No sys.path.insert remnants
print_status "Checking for sys.path.insert usage..."
if grep -r "sys\.path\.insert" --include="*.py" core infrastructure services tests 2>/dev/null | grep -v "Binary" | grep -v ".pyc"; then
    print_error "Found sys.path.insert usage (should be removed)"
    ((FAILED++))
else
    print_success "No sys.path.insert found (correctly removed)"
    ((PASSED++))
fi

# Test 11: Custom exceptions defined
print_status "Checking exception hierarchy..."
if grep -q "class CognitionOSException" core/exceptions.py; then
    print_success "Custom exception hierarchy exists"
    ((PASSED++))
else
    print_error "CognitionOSException base class not found"
    ((FAILED++))
fi

# Test 12: Error handlers registered
print_status "Checking error handler registration..."
if grep -q "register_error_handlers" services/api/src/main.py; then
    print_success "Error handlers are registered in main.py"
    ((PASSED++))
else
    print_error "Error handlers not registered"
    ((FAILED++))
fi

# Test 13: Request ID middleware registered
print_status "Checking request ID middleware..."
if grep -q "RequestIDMiddleware" services/api/src/main.py; then
    print_success "Request ID middleware is registered"
    ((PASSED++))
else
    print_error "Request ID middleware not registered"
    ((FAILED++))
fi

# Test 14: Health checks implemented
print_status "Checking health check implementations..."
if grep -q "aioredis" services/api/src/dependencies/injection.py && \
   grep -q "aio_pika" services/api/src/dependencies/injection.py; then
    print_success "Health checks implemented for Redis and RabbitMQ"
    ((PASSED++))
else
    print_error "Health checks not fully implemented"
    ((FAILED++))
fi

# Test 15: Billing provider configurable
print_status "Checking billing provider configuration..."
if grep -q "BILLING_PROVIDER" services/api/src/dependencies/injection.py; then
    print_success "Billing provider is configurable"
    ((PASSED++))
else
    print_error "Billing provider not configurable"
    ((FAILED++))
fi

echo ""
echo "=========================================="
echo "Test Results:"
echo "  ✅ Passed:   $PASSED"
echo "  ❌ Failed:   $FAILED"
echo "  ⚠️  Warnings: $WARNINGS"
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo ""
    echo "System is ready. To start:"
    echo "  ./scripts/setup-localhost.sh"
    exit 0
else
    echo -e "${RED}❌ Some tests failed.${NC}"
    echo ""
    echo "Please fix the failed tests before proceeding."
    exit 1
fi
