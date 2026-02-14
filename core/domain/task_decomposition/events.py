"""
Task Decomposition Domain - Events

Domain events for task decomposition operations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List
from uuid import UUID


@dataclass(frozen=True)
class TaskDecomposed:
    """
    Event: Task has been decomposed into subtasks.
    """
    decomposition_id: UUID
    parent_task_id: UUID
    parent_task_name: str
    child_task_ids: List[UUID]
    child_count: int
    depth_level: int
    timestamp: datetime
    metadata: Dict[str, Any]

    @classmethod
    def create(
        cls,
        decomposition_id: UUID,
        parent_task_id: UUID,
        parent_task_name: str,
        child_task_ids: List[UUID],
        depth_level: int,
        metadata: Dict[str, Any] = None
    ) -> "TaskDecomposed":
        """Create task decomposed event"""
        return cls(
            decomposition_id=decomposition_id,
            parent_task_id=parent_task_id,
            parent_task_name=parent_task_name,
            child_task_ids=child_task_ids,
            child_count=len(child_task_ids),
            depth_level=depth_level,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )


@dataclass(frozen=True)
class DependencyAdded:
    """
    Event: Dependency has been added between tasks.
    """
    decomposition_id: UUID
    from_task_id: UUID
    to_task_id: UUID
    dependency_type: str
    timestamp: datetime
    metadata: Dict[str, Any]

    @classmethod
    def create(
        cls,
        decomposition_id: UUID,
        from_task_id: UUID,
        to_task_id: UUID,
        dependency_type: str,
        metadata: Dict[str, Any] = None
    ) -> "DependencyAdded":
        """Create dependency added event"""
        return cls(
            decomposition_id=decomposition_id,
            from_task_id=from_task_id,
            to_task_id=to_task_id,
            dependency_type=dependency_type,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )


@dataclass(frozen=True)
class CycleDetected:
    """
    Event: Cycle detected in task dependency graph.
    """
    decomposition_id: UUID
    cycle_nodes: List[UUID]
    cycle_description: str
    timestamp: datetime
    severity: str  # "warning" or "error"

    @classmethod
    def create(
        cls,
        decomposition_id: UUID,
        cycle_nodes: List[UUID],
        cycle_description: str,
        severity: str = "error"
    ) -> "CycleDetected":
        """Create cycle detected event"""
        return cls(
            decomposition_id=decomposition_id,
            cycle_nodes=cycle_nodes,
            cycle_description=cycle_description,
            timestamp=datetime.utcnow(),
            severity=severity
        )


@dataclass(frozen=True)
class DecompositionCompleted:
    """
    Event: Task decomposition has been completed.
    """
    decomposition_id: UUID
    workflow_execution_id: UUID
    total_nodes: int
    max_depth_reached: int
    leaf_node_count: int
    has_cycles: bool
    timestamp: datetime
    duration_seconds: float
    metadata: Dict[str, Any]

    @classmethod
    def create(
        cls,
        decomposition_id: UUID,
        workflow_execution_id: UUID,
        total_nodes: int,
        max_depth_reached: int,
        leaf_node_count: int,
        has_cycles: bool,
        duration_seconds: float,
        metadata: Dict[str, Any] = None
    ) -> "DecompositionCompleted":
        """Create decomposition completed event"""
        return cls(
            decomposition_id=decomposition_id,
            workflow_execution_id=workflow_execution_id,
            total_nodes=total_nodes,
            max_depth_reached=max_depth_reached,
            leaf_node_count=leaf_node_count,
            has_cycles=has_cycles,
            timestamp=datetime.utcnow(),
            duration_seconds=duration_seconds,
            metadata=metadata or {}
        )


@dataclass(frozen=True)
class DecompositionStarted:
    """
    Event: Task decomposition has started.
    """
    decomposition_id: UUID
    workflow_execution_id: UUID
    root_task_name: str
    strategy: str
    timestamp: datetime
    metadata: Dict[str, Any]

    @classmethod
    def create(
        cls,
        decomposition_id: UUID,
        workflow_execution_id: UUID,
        root_task_name: str,
        strategy: str,
        metadata: Dict[str, Any] = None
    ) -> "DecompositionStarted":
        """Create decomposition started event"""
        return cls(
            decomposition_id=decomposition_id,
            workflow_execution_id=workflow_execution_id,
            root_task_name=root_task_name,
            strategy=strategy,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )


@dataclass(frozen=True)
class TaskNodeStatusChanged:
    """
    Event: Task node status has changed.
    """
    decomposition_id: UUID
    task_node_id: UUID
    task_name: str
    old_status: str
    new_status: str
    timestamp: datetime
    reason: str

    @classmethod
    def create(
        cls,
        decomposition_id: UUID,
        task_node_id: UUID,
        task_name: str,
        old_status: str,
        new_status: str,
        reason: str = ""
    ) -> "TaskNodeStatusChanged":
        """Create task node status changed event"""
        return cls(
            decomposition_id=decomposition_id,
            task_node_id=task_node_id,
            task_name=task_name,
            old_status=old_status,
            new_status=new_status,
            timestamp=datetime.utcnow(),
            reason=reason
        )


@dataclass(frozen=True)
class IntegrityViolationDetected:
    """
    Event: Integrity violation detected in decomposition.
    """
    decomposition_id: UUID
    violation_type: str
    description: str
    affected_nodes: List[UUID]
    timestamp: datetime
    severity: str  # "low", "medium", "high", "critical"

    @classmethod
    def create(
        cls,
        decomposition_id: UUID,
        violation_type: str,
        description: str,
        affected_nodes: List[UUID],
        severity: str = "medium"
    ) -> "IntegrityViolationDetected":
        """Create integrity violation detected event"""
        return cls(
            decomposition_id=decomposition_id,
            violation_type=violation_type,
            description=description,
            affected_nodes=affected_nodes,
            timestamp=datetime.utcnow(),
            severity=severity
        )
