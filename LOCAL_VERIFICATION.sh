#!/bin/bash
# CognitionOS SaaS Platform - Local Verification Script
# Validates that all systems are operational

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "   CognitionOS SaaS Platform - System Verification"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASS_COUNT++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAIL_COUNT++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARN_COUNT++))
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check Docker services
echo "1. Infrastructure Services"
echo "─────────────────────────────────────────────────────────────"

if docker ps | grep -q cognitionos-postgres; then
    check_pass "PostgreSQL container running"
else
    check_fail "PostgreSQL container not running"
fi

if docker ps | grep -q cognitionos-redis; then
    check_pass "Redis container running"
else
    check_fail "Redis container not running"
fi

if docker ps | grep -q cognitionos-rabbitmq; then
    check_pass "RabbitMQ container running"
else
    check_fail "RabbitMQ container not running"
fi

if docker ps | grep -q cognitionos-api; then
    check_pass "API container running"
else
    check_fail "API container not running"
fi

echo ""

# Check service health
echo "2. Service Health Checks"
echo "─────────────────────────────────────────────────────────────"

# PostgreSQL
if docker exec cognitionos-postgres-1 pg_isready -U cognition &> /dev/null; then
    check_pass "PostgreSQL accepting connections"
else
    check_fail "PostgreSQL not accepting connections"
fi

# Redis
if docker exec cognitionos-redis-1 redis-cli ping &> /dev/null; then
    check_pass "Redis responding to ping"
else
    check_fail "Redis not responding"
fi

# RabbitMQ
if docker exec cognitionos-rabbitmq-1 rabbitmqctl status &> /dev/null; then
    check_pass "RabbitMQ status healthy"
else
    check_fail "RabbitMQ status check failed"
fi

echo ""

# Check API endpoints
echo "3. API Endpoint Verification"
echo "─────────────────────────────────────────────────────────────"

# Root endpoint
if curl -s http://localhost:8100/ | grep -q "CognitionOS"; then
    check_pass "Root endpoint responding"
else
    check_fail "Root endpoint not responding"
fi

# Health endpoint
HEALTH_RESPONSE=$(curl -s http://localhost:8100/health 2>/dev/null)
if echo "$HEALTH_RESPONSE" | grep -q "healthy\|degraded"; then
    check_pass "Health endpoint responding"
    
    # Check individual health statuses
    if echo "$HEALTH_RESPONSE" | grep -q '"database":"healthy"'; then
        check_pass "Database health check passing"
    else
        check_warn "Database health check not passing"
    fi
else
    check_fail "Health endpoint not responding"
fi

# OpenAPI docs
if curl -s http://localhost:8100/openapi.json | grep -q "openapi"; then
    check_pass "OpenAPI specification available"
else
    check_fail "OpenAPI specification not available"
fi

echo ""

# Check new SaaS endpoints
echo "4. Multi-Tenancy & Billing Endpoints"
echo "─────────────────────────────────────────────────────────────"

# Test tenant endpoint (list should work without tenant context)
TENANT_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8100/api/v3/tenants 2>/dev/null)
if [ "$TENANT_RESPONSE" == "200" ] || [ "$TENANT_RESPONSE" == "400" ]; then
    check_pass "Tenant endpoint accessible"
else
    check_warn "Tenant endpoint returned: $TENANT_RESPONSE"
fi

# Test subscription endpoint (with mock tenant header)
SUB_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -H "X-Tenant-Slug: default" http://localhost:8100/api/v3/subscriptions/current 2>/dev/null)
if [ "$SUB_RESPONSE" == "200" ] || [ "$SUB_RESPONSE" == "404" ] || [ "$SUB_RESPONSE" == "400" ]; then
    check_pass "Subscription endpoint accessible"
else
    check_warn "Subscription endpoint returned: $SUB_RESPONSE"
fi

# Test plugin endpoint
PLUGIN_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8100/api/v3/plugins 2>/dev/null)
if [ "$PLUGIN_RESPONSE" == "200" ] || [ "$PLUGIN_RESPONSE" == "400" ]; then
    check_pass "Plugin endpoint accessible"
else
    check_warn "Plugin endpoint returned: $PLUGIN_RESPONSE"
fi

echo ""

# Check database schema
echo "5. Database Schema Verification"
echo "─────────────────────────────────────────────────────────────"

# Check if multi-tenancy tables exist
if docker exec cognitionos-postgres-1 psql -U cognition -d cognitionos -c "\dt tenants" 2>/dev/null | grep -q "tenants"; then
    check_pass "Tenants table exists"
else
    check_warn "Tenants table not found (migration may not have run)"
fi

if docker exec cognitionos-postgres-1 psql -U cognition -d cognitionos -c "\dt subscriptions" 2>/dev/null | grep -q "subscriptions"; then
    check_pass "Subscriptions table exists"
else
    check_warn "Subscriptions table not found (migration may not have run)"
fi

if docker exec cognitionos-postgres-1 psql -U cognition -d cognitionos -c "\dt plugins" 2>/dev/null | grep -q "plugins"; then
    check_pass "Plugins table exists"
else
    check_warn "Plugins table not found (migration may not have run)"
fi

echo ""

# Check file structure
echo "6. Codebase Structure Verification"
echo "─────────────────────────────────────────────────────────────"

# Domain layer
if [ -d "core/domain/tenant" ]; then
    check_pass "Tenant domain exists"
else
    check_fail "Tenant domain not found"
fi

if [ -d "core/domain/billing" ]; then
    check_pass "Billing domain exists"
else
    check_fail "Billing domain not found"
fi

if [ -d "core/domain/plugin" ]; then
    check_pass "Plugin domain exists"
else
    check_fail "Plugin domain not found"
fi

# Infrastructure layer
if [ -f "infrastructure/middleware/tenant_context.py" ]; then
    check_pass "Tenant context middleware exists"
else
    check_fail "Tenant context middleware not found"
fi

if [ -f "infrastructure/middleware/rate_limiting.py" ]; then
    check_pass "Rate limiting middleware exists"
else
    check_fail "Rate limiting middleware not found"
fi

if [ -f "infrastructure/billing/provider.py" ]; then
    check_pass "Billing provider exists"
else
    check_fail "Billing provider not found"
fi

# API routes
if [ -f "services/api/src/routes/tenants.py" ]; then
    check_pass "Tenant routes exist"
else
    check_fail "Tenant routes not found"
fi

if [ -f "services/api/src/routes/subscriptions.py" ]; then
    check_pass "Subscription routes exist"
else
    check_fail "Subscription routes not found"
fi

if [ -f "services/api/src/routes/plugins.py" ]; then
    check_pass "Plugin routes exist"
else
    check_fail "Plugin routes not found"
fi

echo ""

# Summary
echo "═══════════════════════════════════════════════════════════════"
echo "                    Verification Summary"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}Passed:${NC}  $PASS_COUNT checks"
echo -e "${YELLOW}Warnings:${NC} $WARN_COUNT checks"
echo -e "${RED}Failed:${NC}  $FAIL_COUNT checks"
echo ""

if [ $FAIL_COUNT -eq 0 ] && [ $WARN_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ All systems operational! 🚀${NC}"
    echo ""
    exit 0
elif [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${YELLOW}⚠ System operational with warnings${NC}"
    echo ""
    echo "Review warnings above. System should be functional."
    exit 0
else
    echo -e "${RED}✗ System has failures${NC}"
    echo ""
    echo "Please review failed checks above and run:"
    echo "  docker-compose -f docker-compose.local.yml logs"
    exit 1
fi
