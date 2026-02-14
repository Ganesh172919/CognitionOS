"""
Task Decomposition Domain - Phase 4

Domain model for hierarchical task decomposition supporting 10,000+ interconnected tasks.
Enables recursive decomposition with 100+ depth levels, dependency validation, and cycle detection.
"""

from .entities import (
    TaskNode,
    TaskNodeId,
    TaskDecomposition,
    Dependency,
    DecompositionStrategy,
    DependencyType,
    TaskNodeStatus,
)
from .services import (
    RecursiveDecomposer,
    DependencyValidator,
    CycleDetector,
    IntegrityEnforcer,
)
from .events import (
    TaskDecomposed,
    DependencyAdded,
    CycleDetected,
    DecompositionCompleted,
    DecompositionStarted,
    TaskNodeStatusChanged,
    IntegrityViolationDetected,
)
from .repositories import (
    TaskDecompositionRepository,
    TaskNodeRepository,
)

__all__ = [
    # Entities
    "TaskNode",
    "TaskNodeId",
    "TaskDecomposition",
    "Dependency",
    "DecompositionStrategy",
    "DependencyType",
    "TaskNodeStatus",
    # Services
    "RecursiveDecomposer",
    "DependencyValidator",
    "CycleDetector",
    "IntegrityEnforcer",
    # Events
    "TaskDecomposed",
    "DependencyAdded",
    "CycleDetected",
    "DecompositionCompleted",
    "DecompositionStarted",
    "TaskNodeStatusChanged",
    "IntegrityViolationDetected",
    # Repositories
    "TaskDecompositionRepository",
    "TaskNodeRepository",
]
