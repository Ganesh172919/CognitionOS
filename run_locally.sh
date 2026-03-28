#!/usr/bin/env bash
# =============================================================================
# CognitionOS — Local Development Startup Script
# =============================================================================
# One command to clone, configure, and run the full CognitionOS stack locally.
#
# Usage:
#   ./run_locally.sh            # Start with Docker Compose (recommended)
#   ./run_locally.sh --stop     # Stop all containers
#   ./run_locally.sh --logs     # Tail all container logs
#   ./run_locally.sh --status   # Show container status
#   ./run_locally.sh --clean    # Remove containers and volumes
#
# Requirements:
#   - Docker 20.10+ and Docker Compose v2+ (docker compose)
#   - 4 GB free RAM, 10 GB free disk
#   - (Optional) Node.js 18+ for the Next.js frontend
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*" >&2; }
header()  { echo -e "\n${BOLD}${BLUE}==> $*${NC}"; }

COMPOSE_FILE="docker-compose.local.yml"
PROJECT_NAME="cognitionos"

# ---------------------------------------------------------------------------
# Subcommand handling
# ---------------------------------------------------------------------------
case "${1:-}" in
  --stop)
    info "Stopping CognitionOS containers..."
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down
    success "Stopped."
    exit 0
    ;;
  --logs)
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs -f
    exit 0
    ;;
  --status)
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps
    exit 0
    ;;
  --clean)
    warn "This will remove all containers AND persistent volumes (database data will be lost)."
    read -rp "Are you sure? [y/N] " answer
    [[ "$answer" =~ ^[Yy]$ ]] || { info "Cancelled."; exit 0; }
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down -v --remove-orphans
    success "Cleaned up."
    exit 0
    ;;
esac

# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║         CognitionOS — Local Development Startup          ║${NC}"
echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------
header "Checking prerequisites"

if ! command -v docker &>/dev/null; then
  error "Docker is not installed. Install it from https://docs.docker.com/get-docker/"
  exit 1
fi
success "Docker $(docker --version | awk '{print $3}' | tr -d ',')"

if ! docker compose version &>/dev/null 2>&1; then
  error "Docker Compose v2 plugin is not available. Install it alongside Docker."
  exit 1
fi
success "Docker Compose $(docker compose version --short 2>/dev/null || echo 'v2+')"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
header "Environment configuration"

if [ ! -f .env ]; then
  cp .env.localhost .env
  success "Created .env from .env.localhost"
  echo ""
  warn "No API keys are set — AI-backed features will use placeholder values."
  warn "Edit .env and set LLM_OPENAI_API_KEY / LLM_ANTHROPIC_API_KEY to enable them."
  echo ""
else
  success ".env already exists — skipping copy"
fi

# ---------------------------------------------------------------------------
# Start infrastructure services
# ---------------------------------------------------------------------------
header "Starting infrastructure services (PostgreSQL, Redis, RabbitMQ)"

docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d postgres redis rabbitmq

# ---------------------------------------------------------------------------
# Wait for each service to become healthy
# ---------------------------------------------------------------------------
header "Waiting for services to become healthy"

wait_healthy() {
  local name="$1"
  local max="${2:-30}"
  local i=0
  echo -n "  Waiting for $name"
  while [ $i -lt "$max" ]; do
    status=$(docker inspect --format='{{.State.Health.Status}}' "${PROJECT_NAME}-${name}-local" 2>/dev/null || echo "missing")
    if [ "$status" = "healthy" ]; then
      echo " ✓"
      return 0
    fi
    echo -n "."
    sleep 1
    i=$((i + 1))
  done
  echo " ✗"
  error "$name failed to become healthy within ${max}s"
  docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs "$name" | tail -20
  exit 1
}

wait_healthy postgres 60
wait_healthy redis 30
wait_healthy rabbitmq 60

success "All infrastructure services are healthy"

# ---------------------------------------------------------------------------
# Start API service
# ---------------------------------------------------------------------------
header "Starting CognitionOS API (port 8100)"

docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d api

# ---------------------------------------------------------------------------
# Wait for API to respond
# ---------------------------------------------------------------------------
header "Waiting for API to respond"

API_URL="http://localhost:8100"
max_wait=90
i=0
echo -n "  Waiting for $API_URL/health"
while [ $i -lt $max_wait ]; do
  if curl -sf "$API_URL/health" -o /dev/null 2>/dev/null; then
    echo " ✓"
    break
  fi
  echo -n "."
  sleep 1
  i=$((i + 1))
  if [ $i -eq $max_wait ]; then
    echo " ✗"
    warn "API did not respond within ${max_wait}s. Showing recent logs:"
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs api | tail -30
    warn "Services are still running — the API may still be initialising."
    warn "Run: docker compose -f $COMPOSE_FILE logs -f api"
  fi
done

# ---------------------------------------------------------------------------
# Optional: Frontend (Next.js)
# ---------------------------------------------------------------------------
LAUNCH_FRONTEND=false
if command -v node &>/dev/null && command -v npm &>/dev/null; then
  header "Frontend (Next.js)"
  if [ ! -f frontend/.env.local ]; then
    if [ -f frontend/.env.local.example ]; then
      cp frontend/.env.local.example frontend/.env.local
      success "Created frontend/.env.local from example"
    else
      echo "NEXT_PUBLIC_API_URL=http://localhost:8100" > frontend/.env.local
      success "Created minimal frontend/.env.local"
    fi
  fi

  # Install deps only when node_modules is absent or package.json is newer
  if [ ! -d frontend/node_modules ] || [ frontend/package.json -nt frontend/node_modules/.package-lock.json ]; then
    info "Installing frontend dependencies (npm ci)..."
    (cd frontend && npm ci --silent)
    success "Frontend dependencies installed"
  else
    success "Frontend dependencies already up-to-date"
  fi

  info "Starting Next.js dev server on http://localhost:3000 (background)..."
  mkdir -p "$HOME/.cognitionos"
  FRONTEND_LOG="$HOME/.cognitionos/frontend.log"
  (cd frontend && npm run dev > "$FRONTEND_LOG" 2>&1) &
  FRONTEND_PID=$!
  LAUNCH_FRONTEND=true

  # Short wait to catch immediate crashes
  sleep 3
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    warn "Frontend process exited immediately — check $FRONTEND_LOG"
    LAUNCH_FRONTEND=false
  else
    success "Frontend started (PID $FRONTEND_PID)"
  fi
else
  info "Node.js not found — skipping frontend. Install Node 18+ to run the dashboard."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║           CognitionOS is up and running! 🚀              ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BOLD}Backend API${NC}"
echo    "    REST API:          http://localhost:8100"
echo    "    Interactive Docs:  http://localhost:8100/docs"
echo    "    ReDoc:             http://localhost:8100/redoc"
echo    "    Health:            http://localhost:8100/health"
echo    "    Metrics:           http://localhost:8100/metrics"
echo ""
echo -e "  ${BOLD}Infrastructure${NC}"
echo    "    PostgreSQL:        localhost:5432"
echo    "    Redis:             localhost:6379"
echo    "    RabbitMQ:          localhost:5672"
echo    "    RabbitMQ Admin UI: http://localhost:15672  (guest / guest)"
echo ""

if $LAUNCH_FRONTEND; then
  echo -e "  ${BOLD}Frontend Dashboard${NC}"
  echo    "    Next.js App:       http://localhost:3000"
  echo    "    Logs:              $HOME/.cognitionos/frontend.log"
  echo    "    Stop:              kill $FRONTEND_PID"
  echo ""
fi

echo -e "  ${BOLD}Quick smoke-test${NC}"
echo    "    curl http://localhost:8100/health"
echo    "    curl http://localhost:8100/api/v3/health/live"
echo ""
echo -e "  ${BOLD}Management${NC}"
echo    "    Logs (all):   ./run_locally.sh --logs"
echo    "    Stop:         ./run_locally.sh --stop"
echo    "    Clean up:     ./run_locally.sh --clean"
echo    "    Container PS: ./run_locally.sh --status"
echo ""
echo -e "  ${BOLD}Example API calls${NC}"
echo    "    # Create a workflow"
cat <<'EOF'
    curl -s -X POST http://localhost:8100/api/v3/workflows \
      -H "Content-Type: application/json" \
      -d '{"workflow_id":"demo","version":"1.0","name":"Demo","description":"Test","steps":[{"step_id":"s1","name":"Step 1","agent_capability":"general","inputs":{},"depends_on":[]}]}' \
      | python3 -m json.tool
EOF
echo ""
