# Quick Start Guide

Welcome to CognitionOS! This guide will get you up and running in 10 minutes.

## Prerequisites

- Python 3.11+
- PostgreSQL 15+ (optional - works in-memory mode)
- Redis 7+ (optional - for caching)
- OpenAI or Anthropic API key (optional - has simulation mode)

## Installation

### Option 1: Quick Demo (No Dependencies)

Run individual services in simulation mode:

```bash
# Clone the repository
git clone https://github.com/Ganesh172919/CognitionOS.git
cd CognitionOS

# Install dependencies for a service
cd services/auth-service
pip install -r requirements.txt

# Run the service
python src/main.py
```

The service will start at `http://localhost:8001` and run in simulation mode (no database required).

### Option 2: Full Stack with Docker

```bash
# Clone the repository
git clone https://github.com/Ganesh172919/CognitionOS.git
cd CognitionOS

# Copy environment template
cp .env.example .env

# Edit .env and add your API keys (optional)
nano .env

# Start all services
docker-compose up -d

# Check status
docker-compose ps
curl http://localhost:8000/health
```

### Option 3: Manual Setup (All Services)

```bash
# Install system dependencies
brew install postgresql redis rabbitmq  # macOS
# or
apt-get install postgresql redis-server rabbitmq-server  # Ubuntu

# Start services
pg_ctl -D /usr/local/var/postgres start
redis-server --daemonize yes
rabbitmq-server -detached

# Create database
createdb cognitionos

# Install Python dependencies for each service
for service in api-gateway auth-service task-planner agent-orchestrator ai-runtime tool-runner; do
    cd services/$service
    pip install -r requirements.txt
    cd ../..
done

# Set environment variables
export DATABASE_URL=postgresql://localhost/cognitionos
export REDIS_URL=redis://localhost:6379
export JWT_SECRET=your-secret-here

# Start each service in a separate terminal
cd services/auth-service && python src/main.py &
cd services/api-gateway && python src/main.py &
cd services/task-planner && python src/main.py &
# ... etc
```

## First API Calls

### 1. Register a User

```bash
curl -X POST http://localhost:8001/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "alice@example.com",
    "username": "alice",
    "password": "securepass123"
  }'
```

Response:
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "email": "alice@example.com",
  "username": "alice",
  "role": "user",
  "created_at": "2026-02-09T16:00:00Z"
}
```

### 2. Login

```bash
curl -X POST http://localhost:8001/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "alice@example.com",
    "password": "securepass123"
  }'
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "a1b2c3d4e5f6...",
  "token_type": "bearer",
  "expires_in": 900
}
```

Save the `access_token` for subsequent requests!

### 3. Create a Task Plan

```bash
export TOKEN="eyJ0eXAiOiJKV1QiLCJhbGc..."  # From login response

curl -X POST http://localhost:8002/plan \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "user_id": "123e4567-e89b-12d3-a456-426614174000",
    "goal": "Build a web application with user authentication",
    "context": {},
    "max_depth": 10
  }'
```

Response:
```json
{
  "id": "plan-456...",
  "goal_id": "goal-789...",
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "goal": "Build a web application with user authentication",
  "tasks": [
    {
      "task_id": "task-1...",
      "name": "Design Database Schema",
      "description": "Design Database Schema for: Build a web application...",
      "required_capabilities": ["database_design"],
      "dependencies": [],
      "estimated_duration_seconds": 300
    },
    {
      "task_id": "task-2...",
      "name": "Implement Backend API",
      "description": "Implement Backend API for: Build a web application...",
      "required_capabilities": ["backend_development", "api_development"],
      "dependencies": ["task-1..."],
      "estimated_duration_seconds": 300
    }
  ],
  "execution_order": [
    ["task-1..."],
    ["task-2...", "task-3..."],
    ["task-4..."]
  ],
  "total_estimated_duration_seconds": 1500
}
```

### 4. Execute a Task

```bash
curl -X POST http://localhost:8003/tasks/execute \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "task_id": "task-1...",
    "user_id": "123e4567-e89b-12d3-a456-426614174000",
    "required_capabilities": ["database_design"],
    "task_data": {
      "description": "Design database schema for user authentication"
    }
  }'
```

Response:
```json
{
  "execution_id": "exec-abc...",
  "task_id": "task-1...",
  "agent_id": "agent-def...",
  "success": true,
  "output": {
    "result": "Completed: Design database schema...",
    "agent_role": "planner"
  },
  "duration_seconds": 0.1,
  "cost": 0.01
}
```

## Service Endpoints

| Service | Port | Health Check | Purpose |
|---------|------|--------------|---------|
| API Gateway | 8000 | `http://localhost:8000/health` | Entry point |
| Auth Service | 8001 | `http://localhost:8001/health` | Authentication |
| Task Planner | 8002 | `http://localhost:8002/health` | Task planning |
| Agent Orchestrator | 8003 | `http://localhost:8003/health` | Agent management |
| AI Runtime | 8005 | `http://localhost:8005/health` | LLM routing |
| Tool Runner | 8006 | `http://localhost:8006/health` | Tool execution |

## Testing Tools

### Interactive API Documentation

Each service has auto-generated API docs:

- Auth Service: `http://localhost:8001/docs`
- Task Planner: `http://localhost:8002/docs`
- Agent Orchestrator: `http://localhost:8003/docs`
- AI Runtime: `http://localhost:8005/docs`
- Tool Runner: `http://localhost:8006/docs`

### Using Postman

Import the Postman collection (if available) or create requests manually using the examples above.

### Using cURL

All examples in this guide use cURL. They work on macOS, Linux, and Windows (with WSL or Git Bash).

## Common Issues

### Port Already in Use

```bash
# Find process using port
lsof -i :8001

# Kill process
kill -9 <PID>
```

### Database Connection Error

```bash
# Check PostgreSQL is running
pg_isready

# If not running
pg_ctl -D /usr/local/var/postgres start
```

### Redis Connection Error

```bash
# Check Redis is running
redis-cli ping

# If not running
redis-server --daemonize yes
```

### Module Not Found

```bash
# Make sure you're in the right directory
cd services/auth-service

# Install dependencies
pip install -r requirements.txt

# Or use virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Development Workflow

### Making Changes

1. Edit code in `services/<service-name>/src/`
2. Test locally by running the service
3. Verify with health check
4. Test API endpoints
5. Commit changes

### Adding a New Tool

1. Edit `services/tool-runner/src/main.py`
2. Add tool definition to `TOOLS` dict
3. Implement execution method
4. Test with API call
5. Update documentation

### Adding a New Agent Type

1. Edit `services/agent-orchestrator/src/main.py`
2. Create `AgentDefinition` in `_initialize_default_agents()`
3. Define capabilities and tools
4. Test with task execution
5. Update documentation

## Next Steps

- Read [Architecture Documentation](docs/architecture.md)
- Explore [Agent Model](docs/agent_model.md)
- Review [Security Guidelines](docs/security.md)
- Check [Deployment Guide](docs/deployment.md)

## Getting Help

- GitHub Issues: Report bugs or request features
- GitHub Discussions: Ask questions
- Documentation: Read the docs/ folder

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

---

Happy building! 🚀
