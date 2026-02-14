"""
Task Decomposition Domain - Entities

Pure domain entities for hierarchical task decomposition (Phase 4).
Supports 10,000+ interconnected tasks with 100+ depth levels.
NO external dependencies except Python stdlib.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4


class TaskNodeStatus(str, Enum):
    """Task node status in decomposition"""
    PENDING = "pending"
    DECOMPOSING = "decomposing"
    DECOMPOSED = "decomposed"
    READY = "ready"
    BLOCKED = "blocked"
    FAILED = "failed"


class DependencyType(str, Enum):
    """Type of dependency between task nodes"""
    SEQUENTIAL = "sequential"  # Must complete before next starts
    PARALLEL = "parallel"  # Can run in parallel
    CONDITIONAL = "conditional"  # Depends on condition
    RESOURCE = "resource"  # Shares resources


class DecompositionStrategy(str, Enum):
    """Strategy for task decomposition"""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


# ==================== Value Objects ====================

@dataclass(frozen=True)
class TaskNodeId:
    """Task node identifier value object"""
    value: UUID

    @classmethod
    def generate(cls) -> "TaskNodeId":
        """Generate a new task node ID"""
        return cls(value=uuid4())

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Dependency:
    """Dependency between task nodes"""
    from_node_id: UUID
    to_node_id: UUID
    dependency_type: DependencyType
    condition: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "from_node_id": str(self.from_node_id),
            "to_node_id": str(self.to_node_id),
            "dependency_type": self.dependency_type.value,
            "condition": self.condition,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dependency":
        """Create from dictionary"""
        return cls(
            from_node_id=UUID(data["from_node_id"]),
            to_node_id=UUID(data["to_node_id"]),
            dependency_type=DependencyType(data["dependency_type"]),
            condition=data.get("condition"),
            metadata=data.get("metadata", {}),
        )


# ==================== Entities ====================

@dataclass
class TaskNode:
    """
    Task node entity representing a single task in decomposition hierarchy.
    
    Supports parent-child relationships with depth tracking up to 100+ levels.
    """
    id: TaskNodeId
    decomposition_id: UUID
    name: str
    description: str
    
    # Hierarchy
    parent_id: Optional[UUID] = None
    depth_level: int = 0
    is_leaf: bool = True
    
    # Status
    status: TaskNodeStatus = TaskNodeStatus.PENDING
    
    # Dependencies
    dependencies: List[Dependency] = field(default_factory=list)
    
    # Decomposition
    child_node_ids: List[UUID] = field(default_factory=list)
    estimated_complexity: float = 1.0  # 0-1 scale
    actual_subtask_count: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    decomposed_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate task node invariants"""
        if self.depth_level < 0:
            raise ValueError("Depth level cannot be negative")
        
        if self.depth_level > 200:  # Safety limit
            raise ValueError(f"Depth level {self.depth_level} exceeds maximum (200)")
        
        if self.estimated_complexity < 0 or self.estimated_complexity > 1:
            raise ValueError("Estimated complexity must be between 0 and 1")

    def add_child(self, child_id: UUID) -> None:
        """Add a child node"""
        if child_id not in self.child_node_ids:
            self.child_node_ids.append(child_id)
            self.is_leaf = False
            self.actual_subtask_count = len(self.child_node_ids)

    def add_dependency(self, dependency: Dependency) -> None:
        """Add a dependency to this node"""
        # Check if dependency already exists
        for dep in self.dependencies:
            if (dep.from_node_id == dependency.from_node_id and
                dep.to_node_id == dependency.to_node_id):
                return  # Already exists
        
        self.dependencies.append(dependency)

    def remove_dependency(self, from_node_id: UUID, to_node_id: UUID) -> bool:
        """Remove a dependency"""
        original_count = len(self.dependencies)
        self.dependencies = [
            dep for dep in self.dependencies
            if not (dep.from_node_id == from_node_id and dep.to_node_id == to_node_id)
        ]
        return len(self.dependencies) < original_count

    def mark_decomposing(self) -> None:
        """Mark node as being decomposed"""
        if self.status != TaskNodeStatus.PENDING:
            raise ValueError(f"Cannot mark as decomposing from status: {self.status}")
        self.status = TaskNodeStatus.DECOMPOSING

    def mark_decomposed(self) -> None:
        """Mark node as decomposed"""
        if self.status != TaskNodeStatus.DECOMPOSING:
            raise ValueError(f"Cannot mark as decomposed from status: {self.status}")
        self.status = TaskNodeStatus.DECOMPOSED
        self.decomposed_at = datetime.utcnow()

    def mark_ready(self) -> None:
        """Mark node as ready for execution"""
        self.status = TaskNodeStatus.READY

    def mark_blocked(self, reason: str) -> None:
        """Mark node as blocked"""
        self.status = TaskNodeStatus.BLOCKED
        self.metadata["blocked_reason"] = reason
        self.metadata["blocked_at"] = datetime.utcnow().isoformat()

    def mark_failed(self, error: str) -> None:
        """Mark node as failed"""
        self.status = TaskNodeStatus.FAILED
        self.metadata["error"] = error
        self.metadata["failed_at"] = datetime.utcnow().isoformat()

    def get_all_dependencies(self) -> List[UUID]:
        """Get all dependency node IDs (nodes this depends on)"""
        return [dep.from_node_id for dep in self.dependencies]

    def has_dependencies(self) -> bool:
        """Check if node has dependencies"""
        return len(self.dependencies) > 0

    def is_decomposable(self) -> bool:
        """Check if node can be further decomposed"""
        return (
            self.status == TaskNodeStatus.PENDING and
            self.is_leaf and
            self.estimated_complexity > 0.3  # Complex enough to warrant decomposition
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": str(self.id.value),
            "decomposition_id": str(self.decomposition_id),
            "name": self.name,
            "description": self.description,
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "depth_level": self.depth_level,
            "is_leaf": self.is_leaf,
            "status": self.status.value,
            "dependencies": [dep.to_dict() for dep in self.dependencies],
            "child_node_ids": [str(cid) for cid in self.child_node_ids],
            "estimated_complexity": self.estimated_complexity,
            "actual_subtask_count": self.actual_subtask_count,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "decomposed_at": self.decomposed_at.isoformat() if self.decomposed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskNode":
        """Create from dictionary"""
        return cls(
            id=TaskNodeId(value=UUID(data["id"])),
            decomposition_id=UUID(data["decomposition_id"]),
            name=data["name"],
            description=data["description"],
            parent_id=UUID(data["parent_id"]) if data.get("parent_id") else None,
            depth_level=data.get("depth_level", 0),
            is_leaf=data.get("is_leaf", True),
            status=TaskNodeStatus(data.get("status", "pending")),
            dependencies=[Dependency.from_dict(d) for d in data.get("dependencies", [])],
            child_node_ids=[UUID(cid) for cid in data.get("child_node_ids", [])],
            estimated_complexity=data.get("estimated_complexity", 1.0),
            actual_subtask_count=data.get("actual_subtask_count", 0),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            decomposed_at=datetime.fromisoformat(data["decomposed_at"]) if data.get("decomposed_at") else None,
        )

    @classmethod
    def create(
        cls,
        decomposition_id: UUID,
        name: str,
        description: str,
        parent_id: Optional[UUID] = None,
        depth_level: int = 0,
        estimated_complexity: float = 1.0,
        tags: Optional[List[str]] = None,
    ) -> "TaskNode":
        """Factory method to create a new task node"""
        return cls(
            id=TaskNodeId.generate(),
            decomposition_id=decomposition_id,
            name=name,
            description=description,
            parent_id=parent_id,
            depth_level=depth_level,
            estimated_complexity=estimated_complexity,
            tags=tags or [],
        )


@dataclass
class TaskDecomposition:
    """
    Task decomposition entity representing the entire decomposition tree.
    
    Manages hierarchical task structure with 10,000+ nodes.
    """
    id: UUID
    workflow_execution_id: UUID
    root_task_name: str
    root_task_description: str
    
    # Strategy
    strategy: DecompositionStrategy = DecompositionStrategy.HYBRID
    
    # Statistics
    total_nodes: int = 0
    max_depth_reached: int = 0
    leaf_node_count: int = 0
    
    # Node tracking
    root_node_id: Optional[UUID] = None
    all_node_ids: Set[UUID] = field(default_factory=set)
    
    # Status
    is_complete: bool = False
    has_cycles: bool = False
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    def register_node(self, node_id: UUID, depth: int) -> None:
        """Register a new node in the decomposition"""
        self.all_node_ids.add(node_id)
        self.total_nodes = len(self.all_node_ids)
        
        if depth > self.max_depth_reached:
            self.max_depth_reached = depth

    def register_leaf_node(self, node_id: UUID) -> None:
        """Register a leaf node"""
        self.leaf_node_count += 1

    def set_root_node(self, node_id: UUID) -> None:
        """Set the root node of the decomposition"""
        self.root_node_id = node_id
        self.register_node(node_id, 0)

    def mark_complete(self) -> None:
        """Mark decomposition as complete"""
        self.is_complete = True
        self.completed_at = datetime.utcnow()

    def mark_has_cycles(self) -> None:
        """Mark decomposition as having cycles"""
        self.has_cycles = True

    def get_statistics(self) -> Dict[str, Any]:
        """Get decomposition statistics"""
        return {
            "total_nodes": self.total_nodes,
            "max_depth_reached": self.max_depth_reached,
            "leaf_node_count": self.leaf_node_count,
            "is_complete": self.is_complete,
            "has_cycles": self.has_cycles,
            "strategy": self.strategy.value,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": str(self.id),
            "workflow_execution_id": str(self.workflow_execution_id),
            "root_task_name": self.root_task_name,
            "root_task_description": self.root_task_description,
            "strategy": self.strategy.value,
            "total_nodes": self.total_nodes,
            "max_depth_reached": self.max_depth_reached,
            "leaf_node_count": self.leaf_node_count,
            "root_node_id": str(self.root_node_id) if self.root_node_id else None,
            "all_node_ids": [str(nid) for nid in self.all_node_ids],
            "is_complete": self.is_complete,
            "has_cycles": self.has_cycles,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskDecomposition":
        """Create from dictionary"""
        return cls(
            id=UUID(data["id"]),
            workflow_execution_id=UUID(data["workflow_execution_id"]),
            root_task_name=data["root_task_name"],
            root_task_description=data["root_task_description"],
            strategy=DecompositionStrategy(data.get("strategy", "hybrid")),
            total_nodes=data.get("total_nodes", 0),
            max_depth_reached=data.get("max_depth_reached", 0),
            leaf_node_count=data.get("leaf_node_count", 0),
            root_node_id=UUID(data["root_node_id"]) if data.get("root_node_id") else None,
            all_node_ids=set(UUID(nid) for nid in data.get("all_node_ids", [])),
            is_complete=data.get("is_complete", False),
            has_cycles=data.get("has_cycles", False),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
        )

    @classmethod
    def create(
        cls,
        workflow_execution_id: UUID,
        root_task_name: str,
        root_task_description: str,
        strategy: DecompositionStrategy = DecompositionStrategy.HYBRID,
    ) -> "TaskDecomposition":
        """Factory method to create a new task decomposition"""
        return cls(
            id=uuid4(),
            workflow_execution_id=workflow_execution_id,
            root_task_name=root_task_name,
            root_task_description=root_task_description,
            strategy=strategy,
        )
