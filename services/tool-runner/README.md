# Tool Runner

Sandboxed execution environment for agent tools.

## Purpose

The Tool Runner provides a secure, isolated environment for executing agent tools. It prevents unauthorized access, enforces resource limits, and logs all operations.

## Features

- **Sandboxing**: Docker containers for isolation
- **Permission System**: Tools require explicit permissions
- **Resource Limits**: CPU, memory, time constraints
- **Execution Logging**: All tool calls audited
- **Timeout Protection**: Automatic termination
- **Network Isolation**: Configurable network access

## Supported Tools

### Code Execution
- Python interpreter
- JavaScript (Node.js)
- Shell commands (restricted)

### API Operations
- HTTP requests
- GraphQL queries
- gRPC calls

### File Operations
- Read files
- Write files
- File manipulation

### Database Operations
- SQL queries (read-only by default)
- NoSQL operations

## Security Model

```
Agent Request → Permission Check → Sandbox → Execute → Audit Log
```

### Permission Levels
- `code_execution`: Run code
- `network`: Make HTTP requests
- `filesystem_read`: Read files
- `filesystem_write`: Write files
- `database_read`: Query databases
- `database_write`: Modify databases

## Environment Variables

```
TOOL_RUNNER_PORT=8006
SANDBOX_ENABLED=true
SANDBOX_NETWORK_ENABLED=false
SANDBOX_MEMORY_LIMIT=512m
SANDBOX_CPU_LIMIT=1.0
DEFAULT_TOOL_TIMEOUT=30
MAX_CONCURRENT_EXECUTIONS=10
```

## Tech Stack

- Python 3.11+
- Docker SDK for sandboxing
- FastAPI for REST API
