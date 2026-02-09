# Task Planner

Decomposes high-level user goals into executable task DAGs (Directed Acyclic Graphs).

## Purpose

The Task Planner is the "brain" that translates natural language goals into structured, executable plans. It analyzes dependencies, identifies required capabilities, and creates an optimal execution strategy.

## Features

- **Goal Decomposition**: Break complex goals into atomic tasks
- **DAG Generation**: Create dependency graphs with parallelization opportunities
- **Capability Matching**: Identify which agent types can execute each task
- **Re-planning**: Adapt plans when tasks fail
- **Cost Estimation**: Predict time and resource requirements
- **Conflict Detection**: Identify circular dependencies and resource conflicts

## Architecture

```
User Goal → Task Planner → DAG of Tasks → Agent Orchestrator
```

## Task DAG Structure

```
Goal: "Build and deploy a web application"

    ┌─────────────────┐
    │ Design Database │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Create Schema   │
    └────────┬────────┘
             │
       ┌─────┴─────┐
       ▼           ▼
┌────────────┐ ┌────────────┐
│ Write API  │ │Build Frontend│
└─────┬──────┘ └──────┬─────┘
      │               │
      └───────┬───────┘
              ▼
       ┌────────────┐
       │Write Tests │
       └──────┬─────┘
              ▼
       ┌────────────┐
       │   Deploy   │
       └────────────┘
```

## Planning Algorithms

### 1. Deterministic Planning
- Rule-based decomposition
- Predefined templates for common tasks
- Fast and predictable

### 2. AI-Powered Planning
- LLM-based goal understanding
- Creative problem solving
- Handles novel tasks

### 3. Hybrid Planning
- Use templates when available
- Fall back to AI for novel scenarios
- Best of both worlds

## Environment Variables

```
TASK_PLANNER_PORT=8002
AI_RUNTIME_URL=http://localhost:8005
DATABASE_URL=postgresql://user:pass@localhost:5432/cognition
MAX_TASK_DEPTH=10
MAX_PARALLEL_TASKS=20
```

## Tech Stack

- Python 3.11+
- NetworkX for graph operations
- FastAPI for REST API
- LangChain for AI planning
