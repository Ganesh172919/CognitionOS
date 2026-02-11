"""
Core Domain Package

CognitionOS V3 Clean Architecture - Domain Layer

This package contains the core business logic organized into bounded contexts.
Following Domain-Driven Design and Clean Architecture principles.

Bounded Contexts:
- Workflow: Workflow definitions, executions, and orchestration
- Agent: AI agents, capabilities, and lifecycle management
- Memory: Long-term memory storage, retrieval, and lifecycle
- Task: Task planning, assignment, and execution
- Execution: Execution tracking and observability

Architecture:
- ZERO external dependencies (only Python stdlib)
- All dependencies point inward toward this layer
- Repository interfaces defined here, implemented in infrastructure
- Domain services contain business logic that doesn't fit in entities
- Domain events represent facts about state changes
"""

from . import workflow
from . import agent
from . import memory
from . import task
from . import execution

__all__ = [
    "workflow",
    "agent",
    "memory",
    "task",
    "execution",
]

__version__ = "3.0.0"
