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
- Checkpoint: Checkpoint/resume for 24+ hour autonomous workflows (Phase 3)
- Health Monitoring: Agent health tracking and incident management (Phase 3)
- Cost Governance: Budget enforcement and cost tracking (Phase 3)

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
from . import checkpoint
from . import health_monitoring
from . import cost_governance
from . import memory_hierarchy

__all__ = [
    "workflow",
    "agent",
    "memory",
    "task",
    "execution",
    "checkpoint",
    "health_monitoring",
    "cost_governance",
    "memory_hierarchy",
]

__version__ = "3.1.0"  # Phase 3 Extended Agent Operation
