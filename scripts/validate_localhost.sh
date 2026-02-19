#!/usr/bin/env bash
#
# Comprehensive Localhost Validation Script
# Validates the complete CognitionOS system for production-ready localhost operation
#
set -e

echo "========================================="
echo "CognitionOS Localhost Validation"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0

# Helper functions
pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
}

fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# ====================
# 1. Python Syntax Validation
# ====================
echo "1. Validating Python Syntax..."
echo "----------------------------"

# Check all Python files compile
PYTHON_FILES=$(find . -name "*.py" -not -path "./venv/*" -not -path "./.venv/*" -not -path "./build/*" -not -path "./dist/*" 2>/dev/null | wc -l)
echo "Found $PYTHON_FILES Python files"

SYNTAX_ERRORS=0
for file in $(find . -name "*.py" -not -path "./venv/*" -not -path "./.venv/*" -not -path "./build/*" -not -path "./dist/*" 2>/dev/null); do
    if ! python3 -m py_compile "$file" 2>/dev/null; then
        fail "Syntax error in $file"
        ((SYNTAX_ERRORS++))
    fi
done

if [ $SYNTAX_ERRORS -eq 0 ]; then
    pass "All Python files have valid syntax"
else
    fail "$SYNTAX_ERRORS files have syntax errors"
fi

echo ""

# ====================
# 2. Import Validation
# ====================
echo "2. Validating Python Imports..."
echo "--------------------------------"

# Check core modules can be imported
python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')

try:
    from infrastructure.sdk.auto_generator import SDKAutoGenerator
    print("✓ SDK generator imports successfully")
except Exception as e:
    print(f"✗ SDK generator import failed: {e}")
    sys.exit(1)

try:
    from infrastructure.dev_tools.api_doc_generator import APIDocumentationGenerator
    print("✓ API doc generator imports successfully")
except Exception as e:
    print(f"✗ API doc generator import failed: {e}")
    sys.exit(1)

try:
    from infrastructure.reliability.chaos_engineering import ChaosEngineeringFramework
    print("✓ Chaos engineering imports successfully")
except Exception as e:
    print(f"✗ Chaos engineering import failed: {e}")
    sys.exit(1)

try:
    from infrastructure.workflow.orchestrator import WorkflowOrchestrationEngine
    print("✓ Workflow orchestrator imports successfully")
except Exception as e:
    print(f"✗ Workflow orchestrator import failed: {e}")
    sys.exit(1)
PYEOF

if [ $? -eq 0 ]; then
    pass "All infrastructure modules import successfully"
else
    fail "Some modules failed to import"
fi

echo ""

# ====================
# 3. Configuration Validation
# ====================
echo "3. Validating Configuration..."
echo "------------------------------"

if [ -f ".env.localhost" ]; then
    pass ".env.localhost file exists"

    # Check required environment variables
    REQUIRED_VARS=(
        "DB_HOST"
        "DB_PORT"
        "DB_DATABASE"
        "REDIS_HOST"
        "REDIS_PORT"
        "API_HOST"
        "API_PORT"
    )

    for var in "${REQUIRED_VARS[@]}"; do
        if grep -q "^$var=" .env.localhost; then
            pass "$var is configured"
        else
            fail "$var is missing from .env.localhost"
        fi
    done
else
    fail ".env.localhost file not found"
fi

echo ""

# ====================
# 4. Directory Structure Validation
# ====================
echo "4. Validating Directory Structure..."
echo "-------------------------------------"

REQUIRED_DIRS=(
    "infrastructure/sdk"
    "infrastructure/dev_tools"
    "infrastructure/reliability"
    "infrastructure/workflow"
    "infrastructure/security"
    "infrastructure/devops"
    "infrastructure/intelligence"
    "services/api/src/routes"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        pass "$dir/ exists"
    else
        fail "$dir/ is missing"
    fi
done

echo ""

# ====================
# 5. Module __init__.py Validation
# ====================
echo "5. Validating Module Initialization..."
echo "---------------------------------------"

REQUIRED_INITS=(
    "infrastructure/sdk/__init__.py"
    "infrastructure/dev_tools/__init__.py"
    "infrastructure/reliability/__init__.py"
    "infrastructure/workflow/__init__.py"
)

for init in "${REQUIRED_INITS[@]}"; do
    if [ -f "$init" ]; then
        pass "$init exists"
    else
        fail "$init is missing"
    fi
done

echo ""

# ====================
# 6. API Routes Validation
# ====================
echo "6. Validating API Routes..."
echo "---------------------------"

REQUIRED_ROUTES=(
    "services/api/src/routes/developer_tools.py"
    "services/api/src/routes/reliability_workflows.py"
)

for route in "${REQUIRED_ROUTES[@]}"; do
    if [ -f "$route" ]; then
        pass "$route exists"
    else
        fail "$route is missing"
    fi
done

echo ""

# ====================
# 7. Documentation Validation
# ====================
echo "7. Validating Documentation..."
echo "-------------------------------"

REQUIRED_DOCS=(
    "README.md"
    "PHASE_2_TRANSFORMATION_COMPLETE.md"
)

for doc in "${REQUIRED_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        pass "$doc exists"
    else
        warn "$doc is missing"
    fi
done

echo ""

# ====================
# 8. Docker Configuration Validation
# ====================
echo "8. Validating Docker Configuration..."
echo "-------------------------------------"

if [ -f "docker-compose.yml" ]; then
    pass "docker-compose.yml exists"
else
    fail "docker-compose.yml is missing"
fi

if [ -f "Dockerfile.dev" ]; then
    pass "Dockerfile.dev exists"
else
    warn "Dockerfile.dev is missing"
fi

echo ""

# ====================
# Summary
# ====================
echo "========================================="
echo "Validation Summary"
echo "========================================="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All validations passed!${NC}"
    echo ""
    echo "System is ready for localhost operation."
    echo ""
    echo "Next steps:"
    echo "  1. Start services: docker-compose up -d"
    echo "  2. Check health: curl http://localhost:8100/health"
    echo "  3. View API docs: http://localhost:8100/docs"
    exit 0
else
    echo -e "${RED}✗ Some validations failed${NC}"
    echo ""
    echo "Please fix the failures before starting the system."
    exit 1
fi
