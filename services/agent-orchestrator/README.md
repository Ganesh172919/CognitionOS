# Agent Orchestrator

Manages agent lifecycle, task assignment, and execution coordination.

## Purpose

The Agent Orchestrator is the "conductor" that manages all agents in the system. It spawns agents, assigns tasks, monitors execution, handles failures, and coordinates communication.

## Features

- **Agent Registry**: Catalog of available agent types and their capabilities
- **Agent Pool Management**: Spawns, reuses, and terminates agent instances
- **Task Assignment**: Matches tasks to agents based on capabilities
- **Execution Monitoring**: Tracks agent status and progress
- **Failure Handling**: Retries, reassignment, and graceful degradation
- **Message Bus**: Agent-to-agent communication
- **Load Balancing**: Distributes work across agents

## Agent Lifecycle

```
Created → Idle → Assigned → Reasoning → Executing → Validating → Completed
                    ↑                                               │
                    └───────────────────────────────────────────────┘
                                    (reuse)
```

## Architecture

```
Task Planner → Agent Orchestrator → [Agent Pool] → AI Runtime
                       │                             Tool Runner
                       ↓                             Memory Service
                  Message Bus
```

## Execution Modes

### 1. Parallel Execution
Independent tasks run simultaneously:
```
Task 1 → Agent A ──┐
Task 2 → Agent B ──┤→ Complete
Task 3 → Agent C ──┘
```

### 2. Sequential Execution
Dependent tasks run in order:
```
Task 1 → Agent A → Task 2 → Agent B → Task 3 → Agent C → Complete
```

### 3. Hierarchical Execution
Complex tasks spawn sub-agents:
```
       Coordinator Agent
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
Agent A  Agent B  Agent C
```

## Environment Variables

```
AGENT_ORCHESTRATOR_PORT=8003
MESSAGE_QUEUE_URL=amqp://guest:guest@localhost:5672/
MAX_AGENTS=100
AGENT_TIMEOUT_SECONDS=300
MAX_RETRIES=3
```

## Tech Stack

- Python 3.11+
- FastAPI for REST API
- Celery for task queue
- RabbitMQ for message bus
- Redis for state management
