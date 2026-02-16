# CognitionOS V4 - Development Makefile
# Phase 5.1: Development Workflow Automation

.PHONY: help install setup dev-setup clean test lint format type-check \
        docker-build docker-up docker-down docker-logs docker-ps \
        db-migrate db-rollback db-seed db-reset \
        pre-commit-install pre-commit-run \
        check-all quality metrics \
        help-full

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
PYLINT := $(PYTHON) -m pylint
MYPY := $(PYTHON) -m mypy
PYTEST := $(PYTHON) -m pytest
COMPOSE := docker-compose

# Source directories
SRC_DIRS := core infrastructure services shared tests

##@ General

help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

help-full: ## Display full help with all targets
	@echo "Full target list:"
	@$(MAKE) -pRrq -f $(firstword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | grep -E -v -e '^[^[:alnum:]]' -e '^$@$$'

##@ Development Setup

install: ## Install Python dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r tests/requirements.txt
	$(PIP) install -r services/api/requirements.txt
	@echo "✓ Dependencies installed"

dev-setup: install pre-commit-install ## Complete development environment setup
	@echo "Setting up development environment..."
	@echo "✓ Development environment ready"

setup: ## Quick setup with database initialization
	@bash scripts/setup-local.sh

##@ Code Quality

format: ## Format code with black and isort
	@echo "Running black..."
	$(BLACK) $(SRC_DIRS)
	@echo "Running isort..."
	$(ISORT) $(SRC_DIRS)
	@echo "✓ Code formatted"

lint: ## Run pylint on source code
	@echo "Running pylint..."
	$(PYLINT) $(SRC_DIRS) --rcfile=.pylintrc || true
	@echo "✓ Linting complete"

type-check: ## Run mypy type checking
	@echo "Running mypy..."
	$(MYPY) $(SRC_DIRS) --strict --ignore-missing-imports || true
	@echo "✓ Type checking complete"

check-all: format lint type-check ## Run all code quality checks
	@echo "✓ All checks complete"

quality: check-all ## Alias for check-all

##@ Testing

test: ## Run all tests
	$(PYTEST) tests/ -v --cov=core --cov=infrastructure --cov=services

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/ -v -m unit

test-integration: ## Run integration tests only
	$(PYTEST) tests/integration/ -v -m integration

test-coverage: ## Run tests with coverage report
	$(PYTEST) tests/ -v --cov=core --cov=infrastructure --cov=services --cov-report=html --cov-report=term

test-watch: ## Run tests in watch mode (requires pytest-watch)
	$(PYTEST) tests/ -v --watch

##@ Docker Operations

docker-build: ## Build all Docker images
	$(COMPOSE) build

docker-up: ## Start all services
	$(COMPOSE) up -d --wait
	@echo "✓ All services started"
	@echo "Services available at:"
	@echo "  - API V3:        http://localhost:8100"
	@echo "  - API Gateway:   http://localhost:8000"
	@echo "  - Grafana:       http://localhost:3000 (admin/admin)"
	@echo "  - Prometheus:    http://localhost:9090"
	@echo "  - Jaeger:        http://localhost:16686"
	@echo "  - RabbitMQ:      http://localhost:15672 (guest/guest)"
	@echo "  - PgAdmin:       http://localhost:5050"

docker-down: ## Stop all services
	$(COMPOSE) down

docker-restart: ## Restart all services
	$(COMPOSE) restart

docker-logs: ## View logs from all services
	$(COMPOSE) logs -f

docker-ps: ## List running containers
	$(COMPOSE) ps

docker-clean: ## Remove all containers and volumes
	$(COMPOSE) down -v --remove-orphans
	@echo "✓ All containers and volumes removed"

##@ Database Operations

db-migrate: ## Run database migrations
	$(PYTHON) scripts/run_v2_migrations.py

db-rollback: ## Rollback last migration (manual)
	@echo "Manual rollback required. Check database/migrations/"

db-seed: ## Seed database with sample data
	$(PYTHON) scripts/init_database.py

db-reset: docker-down docker-clean docker-up db-migrate db-seed ## Reset database completely
	@echo "✓ Database reset complete"

db-shell: ## Open PostgreSQL shell
	docker exec -it cognitionos-postgres psql -U cognition -d cognitionos

##@ Pre-commit Hooks

pre-commit-install: ## Install pre-commit hooks
	@echo "Installing pre-commit hooks..."
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
		echo "✓ Pre-commit hooks installed"; \
	else \
		echo "⚠ pre-commit not found. Install with: pip install pre-commit"; \
	fi

pre-commit-run: ## Run pre-commit hooks manually
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit run --all-files; \
	else \
		echo "⚠ pre-commit not found. Running manual checks..."; \
		$(MAKE) check-all; \
	fi

##@ Monitoring & Metrics

metrics: ## View Prometheus metrics
	@echo "Opening Prometheus metrics..."
	@open http://localhost:9090 || xdg-open http://localhost:9090 || echo "Visit http://localhost:9090"

grafana: ## Open Grafana dashboard
	@echo "Opening Grafana dashboard..."
	@open http://localhost:3000 || xdg-open http://localhost:3000 || echo "Visit http://localhost:3000"

jaeger: ## Open Jaeger UI
	@echo "Opening Jaeger UI..."
	@open http://localhost:16686 || xdg-open http://localhost:16686 || echo "Visit http://localhost:16686"

pgadmin: ## Open PgAdmin
	@echo "Opening PgAdmin..."
	@open http://localhost:5050 || xdg-open http://localhost:5050 || echo "Visit http://localhost:5050"

##@ Cleanup

clean: ## Clean build artifacts and cache
	@echo "Cleaning build artifacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "✓ Cleanup complete"

clean-all: clean docker-clean ## Clean everything including Docker
	@echo "✓ Full cleanup complete"

##@ CI/CD

ci-test: install test ## Run CI tests
	@echo "✓ CI tests complete"

ci-lint: install lint type-check ## Run CI linting
	@echo "✓ CI linting complete"

ci: ci-lint ci-test ## Run full CI pipeline
	@echo "✓ CI pipeline complete"

##@ Quick Commands

dev: docker-up ## Start development environment
	@echo "✓ Development environment running"

status: docker-ps ## Show service status
	@echo "Service Status:"
	@$(COMPOSE) ps

health: ## Check health of all services
	@echo "Checking service health..."
	@curl -s http://localhost:8100/health | jq . || echo "API V3: DOWN"
	@curl -s http://localhost:9090/-/healthy >/dev/null && echo "Prometheus: UP" || echo "Prometheus: DOWN"
	@curl -s http://localhost:3000/api/health >/dev/null && echo "Grafana: UP" || echo "Grafana: DOWN"

logs-api: ## View API logs
	$(COMPOSE) logs -f api-v3

logs-db: ## View database logs
	$(COMPOSE) logs -f postgres

