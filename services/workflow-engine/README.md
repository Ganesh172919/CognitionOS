# Workflow Engine Service

Port: **8010**

## Overview

The Workflow Engine service enables declarative workflow execution using YAML/JSON DSL. It provides:

- **Workflow Definition**: Define workflows as code (YAML/JSON)
- **DAG Execution**: Automatic topological ordering and parallel execution
- **Template Variables**: Dynamic parameter substitution (`${{ inputs.repo_url }}`)
- **Retry Logic**: Configurable retry policies per step
- **Conditional Execution**: Skip steps based on conditions
- **Version Management**: Track workflow versions
- **Execution Replay**: Re-run workflows with different parameters
- **Visualization**: Execution graph data for UI

## Features

### Workflow DSL

Example workflow definition:

```yaml
workflow:
  id: "deploy-web-app"
  version: "1.0.0"
  name: "Deploy Web Application"
  description: "Deploy a web application with tests"

  inputs:
    - name: repo_url
      type: string
      required: true
      description: "Git repository URL"

    - name: environment
      type: enum
      values: [dev, staging, prod]
      default: dev

  outputs:
    - name: deployment_url
      type: string
      description: "URL of deployed application"

  steps:
    - id: clone_repo
      type: git_clone
      params:
        url: ${{ inputs.repo_url }}
      agent_role: executor
      timeout: 60s

    - id: run_tests
      type: execute_python
      depends_on: [clone_repo]
      params:
        script: pytest tests/
      agent_role: executor
      retry: 3

    - id: deploy
      type: kubernetes_apply
      depends_on: [run_tests]
      params:
        environment: ${{ inputs.environment }}
      agent_role: executor
      approval_required: true
```

### Supported Step Types

- **Execution**: `execute_python`, `execute_javascript`, `execute_task`
- **HTTP**: `http_request`, `api_call`
- **Database**: `query_database`, `update_database`
- **File**: `read_file`, `write_file`
- **Git**: `git_clone`, `git_commit`, `git_push`
- **Docker**: `docker_build`, `docker_run`
- **Kubernetes**: `kubernetes_apply`, `kubernetes_delete`
- **AI**: `ai_generate`, `ai_embedding`, `ai_validate`
- **Memory**: `memory_store`, `memory_retrieve`
- **Control**: `conditional`, `loop`, `parallel`
- **Notifications**: `send_email`, `send_slack`, `send_alert`

## API Endpoints

### Workflow Definitions

- `POST /workflows` - Create workflow definition
- `GET /workflows/{id}` - Get workflow definition
- `GET /workflows/{id}/versions` - List workflow versions
- `GET /workflows` - List all workflows

### Workflow Execution

- `POST /workflows/{id}/execute` - Execute workflow
- `GET /executions/{id}` - Get execution status
- `GET /executions/{id}/graph` - Get execution graph
- `GET /executions` - List executions
- `POST /executions/{id}/replay` - Replay execution (not yet implemented)

### Health Check

- `GET /health` - Service health status

## Usage Examples

### Create Workflow

```bash
curl -X POST http://localhost:8010/workflows \
  -H "Content-Type: application/json" \
  -d @workflow.json
```

### Execute Workflow

```bash
curl -X POST http://localhost:8010/workflows/deploy-web-app/execute \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "deploy-web-app",
    "workflow_version": "1.0.0",
    "inputs": {
      "repo_url": "https://github.com/example/app.git",
      "environment": "dev"
    },
    "user_id": "00000000-0000-0000-0000-000000000000"
  }'
```

### Get Execution Status

```bash
curl http://localhost:8010/executions/{execution_id}
```

### Get Execution Graph

```bash
curl http://localhost:8010/executions/{execution_id}/graph
```

## Architecture

### Components

1. **DSL Parser** (`core/dsl_parser.py`):
   - Parses YAML/JSON workflow definitions
   - Validates workflow structure
   - Detects DAG cycles
   - Validates input types

2. **Executor** (`core/executor.py`):
   - Executes workflows step-by-step
   - Manages DAG execution order
   - Handles parallel execution
   - Template variable substitution
   - Retry logic
   - Conditional execution

3. **Models** (`models/__init__.py`):
   - Pydantic models for workflows, executions, steps
   - Type-safe API contracts

### Data Flow

```
User → POST /workflows → Parser → Validation → Storage
User → POST /workflows/{id}/execute → Executor → Services → Results
```

### Integration

The workflow engine integrates with:

- **Agent Orchestrator**: For task execution
- **AI Runtime**: For AI operations
- **Memory Service**: For memory operations
- **Tool Runner**: For tool execution

## Database Schema

Tables:
- `workflows` - Workflow definitions
- `workflow_executions` - Execution records
- `workflow_execution_steps` - Step execution details

See `database/migrations/v2/002_workflow_tables.sql` for schema.

## Development

### Running Locally

```bash
cd services/workflow-engine
pip install -r requirements.txt
python -m src.main
```

### Running with Docker

```bash
docker build -t workflow-engine .
docker run -p 8010:8010 workflow-engine
```

### Testing

```bash
pytest tests/
```

## Configuration

Environment variables:

- `SERVICE_NAME` - Service name (default: "workflow-engine")
- `PORT` - Service port (default: 8010)
- `LOG_LEVEL` - Logging level (default: "INFO")
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string (optional)

## Future Enhancements

- [ ] Database persistence (currently in-memory)
- [ ] Workflow replay functionality
- [ ] Workflow rollback
- [ ] Scheduled workflows (cron)
- [ ] Workflow approval workflows
- [ ] Workflow metrics and analytics
- [ ] Workflow templates library
- [ ] Visual workflow editor

## See Also

- [Expansion Plan](../../docs/v2/expansion_plan.md)
- [Refactor Plan](../../docs/v2/refactor_plan.md)
- [Implementation Plan](../../docs/v2/IMPLEMENTATION_PLAN.md)
