"""
Task Decomposition Application Layer - Use Cases

Use cases for hierarchical task decomposition operations supporting 10,000+ node graphs.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from core.domain.task_decomposition import (
    TaskNode,
    TaskDecomposition,
    TaskNodeRepository,
    TaskDecompositionRepository,
    RecursiveDecomposer,
    DependencyValidator,
    CycleDetector,
    IntegrityEnforcer,
    DecompositionStrategy,
    DependencyType,
    TaskNodeStatus,
    # Events
    TaskDecomposed,
    DependencyAdded,
    CycleDetected,
    DecompositionCompleted,
    DecompositionStarted,
)


# ============================================================================
# DTOs - Request Objects
# ============================================================================

@dataclass
class DecomposeTaskRequest:
    """Request to decompose a task into subtasks"""
    workflow_execution_id: UUID
    parent_task_id: Optional[UUID]
    task_name: str
    task_description: str
    estimated_complexity: float
    strategy: DecompositionStrategy
    max_depth: int = 150
    metadata: Dict[str, Any] = None


@dataclass
class SubtaskSpecification:
    """Specification for a subtask during decomposition"""
    name: str
    description: str
    complexity: float
    tags: List[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class ValidateDependenciesRequest:
    """Request to validate dependencies in a decomposition"""
    decomposition_id: UUID


@dataclass
class DetectCyclesRequest:
    """Request to detect cycles in task dependencies"""
    decomposition_id: UUID


@dataclass
class GetDecompositionStatusRequest:
    """Request to get decomposition status"""
    decomposition_id: UUID


@dataclass
class RegisterTaskNodeRequest:
    """Request to register a new task node"""
    decomposition_id: UUID
    name: str
    description: str
    parent_id: Optional[UUID]
    depth_level: int
    estimated_complexity: float
    tags: List[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class GetTaskNodesRequest:
    """Request to get task nodes"""
    decomposition_id: UUID
    depth_level: Optional[int] = None
    parent_id: Optional[UUID] = None
    status: Optional[TaskNodeStatus] = None
    limit: int = 1000


# ============================================================================
# DTOs - Response Objects
# ============================================================================

@dataclass
class TaskNodeDTO:
    """Data transfer object for TaskNode"""
    id: UUID
    decomposition_id: UUID
    name: str
    description: str
    parent_id: Optional[UUID]
    depth_level: int
    estimated_complexity: float
    status: str
    is_leaf: bool
    child_count: int
    dependency_count: int
    tags: List[str]
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class DecompositionDTO:
    """Data transfer object for TaskDecomposition"""
    id: UUID
    workflow_execution_id: UUID
    root_task_name: str
    root_node_id: Optional[UUID]
    strategy: str
    total_nodes: int
    max_depth_reached: int
    leaf_node_count: int
    is_complete: bool
    has_cycles: bool
    created_at: datetime
    completed_at: Optional[datetime]


@dataclass
class DecomposeTaskResponse:
    """Response from task decomposition"""
    decomposition: DecompositionDTO
    created_nodes: List[TaskNodeDTO]
    success: bool
    message: str
    errors: List[str] = None


@dataclass
class ValidateDependenciesResponse:
    """Response from dependency validation"""
    is_valid: bool
    errors: Dict[UUID, List[str]]
    total_errors: int


@dataclass
class CycleInfo:
    """Information about a detected cycle"""
    cycle_nodes: List[UUID]
    cycle_description: str


@dataclass
class DetectCyclesResponse:
    """Response from cycle detection"""
    has_cycles: bool
    cycles: List[CycleInfo]
    cycle_count: int


@dataclass
class GetDecompositionStatusResponse:
    """Response with decomposition status"""
    decomposition: DecompositionDTO
    statistics: Dict[str, Any]
    integrity_check: Dict[str, Any]


@dataclass
class RegisterTaskNodeResponse:
    """Response from task node registration"""
    task_node: TaskNodeDTO
    success: bool
    message: str


@dataclass
class GetTaskNodesResponse:
    """Response with task nodes"""
    nodes: List[TaskNodeDTO]
    total_count: int
    filtered_count: int


# ============================================================================
# Use Cases
# ============================================================================

class DecomposeTaskUseCase:
    """
    Use case: Decompose a task into subtasks.
    
    Creates a hierarchical decomposition with recursive strategy support.
    """
    
    def __init__(
        self,
        task_node_repo: TaskNodeRepository,
        decomposition_repo: TaskDecompositionRepository,
        event_publisher: Any = None
    ):
        self.task_node_repo = task_node_repo
        self.decomposition_repo = decomposition_repo
        self.event_publisher = event_publisher
        self.decomposer = None  # Will be initialized per request
    
    async def execute(
        self,
        request: DecomposeTaskRequest,
        subtask_specifications: List[SubtaskSpecification]
    ) -> DecomposeTaskResponse:
        """Execute task decomposition"""
        try:
            # Initialize decomposer with max_depth
            self.decomposer = RecursiveDecomposer(max_depth=request.max_depth)
            
            # Get or create decomposition
            if request.parent_task_id is None:
                # Create new decomposition (root task)
                decomposition = TaskDecomposition.create(
                    workflow_execution_id=request.workflow_execution_id,
                    root_task_name=request.task_name,
                    root_task_description=request.task_description,
                    strategy=request.strategy
                )
                await self.decomposition_repo.save(decomposition)
                
                # Publish started event
                if self.event_publisher:
                    event = DecompositionStarted.create(
                        decomposition_id=decomposition.id,
                        workflow_execution_id=request.workflow_execution_id,
                        root_task_name=request.task_name,
                        strategy=request.strategy.value
                    )
                    await self.event_publisher.publish(event)
                
                # Create root task node
                parent_task = TaskNode.create(
                    decomposition_id=decomposition.id,
                    name=request.task_name,
                    description=request.task_description,
                    depth_level=0,
                    estimated_complexity=request.estimated_complexity,
                    tags=request.metadata.get("tags", []) if request.metadata else []
                )
                await self.task_node_repo.save(parent_task)
                
                # Set as root
                decomposition.set_root_node(parent_task.id.value)
                await self.decomposition_repo.save(decomposition)
                
            else:
                # Find existing parent task
                parent_task = await self.task_node_repo.find_by_id(request.parent_task_id)
                if not parent_task:
                    return DecomposeTaskResponse(
                        decomposition=None,
                        created_nodes=[],
                        success=False,
                        message=f"Parent task {request.parent_task_id} not found",
                        errors=["Parent task not found"]
                    )
                
                # Get decomposition
                decomposition = await self.decomposition_repo.find_by_id(
                    parent_task.decomposition_id
                )
            
            # Check if can decompose
            if not self.decomposer.can_decompose(parent_task):
                return DecomposeTaskResponse(
                    decomposition=self._to_decomposition_dto(decomposition),
                    created_nodes=[],
                    success=False,
                    message=f"Cannot decompose task at depth {parent_task.depth_level}",
                    errors=["Max depth reached or task not decomposable"]
                )
            
            # Convert specifications to dict format
            specs = [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "complexity": spec.complexity,
                    "tags": spec.tags or [],
                    "metadata": spec.metadata or {}
                }
                for spec in subtask_specifications
            ]
            
            # Perform decomposition
            subtasks = self.decomposer.decompose_task(
                task_node=parent_task,
                decomposition=decomposition,
                subtask_specifications=specs
            )
            
            # Save all subtasks
            for subtask in subtasks:
                await self.task_node_repo.save(subtask)
            
            # Save updated parent and decomposition
            await self.task_node_repo.save(parent_task)
            await self.decomposition_repo.save(decomposition)
            
            # Publish decomposed event
            if self.event_publisher:
                event = TaskDecomposed.create(
                    decomposition_id=decomposition.id,
                    parent_task_id=parent_task.id.value,
                    parent_task_name=parent_task.name,
                    child_task_ids=[st.id.value for st in subtasks],
                    depth_level=parent_task.depth_level
                )
                await self.event_publisher.publish(event)
            
            # Convert to DTOs
            decomposition_dto = self._to_decomposition_dto(decomposition)
            node_dtos = [self._to_task_node_dto(node) for node in subtasks]
            
            return DecomposeTaskResponse(
                decomposition=decomposition_dto,
                created_nodes=node_dtos,
                success=True,
                message=f"Successfully decomposed task into {len(subtasks)} subtasks"
            )
            
        except Exception as e:
            return DecomposeTaskResponse(
                decomposition=None,
                created_nodes=[],
                success=False,
                message=f"Decomposition failed: {str(e)}",
                errors=[str(e)]
            )
    
    def _to_task_node_dto(self, node: TaskNode) -> TaskNodeDTO:
        """Convert TaskNode to DTO"""
        return TaskNodeDTO(
            id=node.id.value,
            decomposition_id=node.decomposition_id,
            name=node.name,
            description=node.description,
            parent_id=node.parent_id,
            depth_level=node.depth_level,
            estimated_complexity=node.estimated_complexity,
            status=node.status.value,
            is_leaf=node.is_leaf,
            child_count=node.actual_subtask_count,
            dependency_count=len(node.dependencies),
            tags=node.tags,
            created_at=node.created_at,
            metadata=node.metadata
        )
    
    def _to_decomposition_dto(self, decomp: TaskDecomposition) -> DecompositionDTO:
        """Convert TaskDecomposition to DTO"""
        return DecompositionDTO(
            id=decomp.id,
            workflow_execution_id=decomp.workflow_execution_id,
            root_task_name=decomp.root_task_name,
            root_node_id=decomp.root_node_id,
            strategy=decomp.strategy.value,
            total_nodes=decomp.total_nodes,
            max_depth_reached=decomp.max_depth_reached,
            leaf_node_count=decomp.leaf_node_count,
            is_complete=decomp.is_complete,
            has_cycles=decomp.has_cycles,
            created_at=decomp.created_at,
            completed_at=decomp.completed_at
        )


class ValidateDependenciesUseCase:
    """
    Use case: Validate task dependencies.
    
    Checks for circular dependencies and validates integrity.
    """
    
    def __init__(
        self,
        task_node_repo: TaskNodeRepository,
        decomposition_repo: TaskDecompositionRepository
    ):
        self.task_node_repo = task_node_repo
        self.decomposition_repo = decomposition_repo
        self.validator = DependencyValidator()
    
    async def execute(self, request: ValidateDependenciesRequest) -> ValidateDependenciesResponse:
        """Execute dependency validation"""
        # Get all nodes for decomposition
        nodes = await self.task_node_repo.find_by_decomposition(request.decomposition_id)
        
        # Convert to dict
        nodes_dict = {node.id.value: node for node in nodes}
        
        # Validate dependencies
        errors = self.validator.validate_all_dependencies(nodes_dict)
        
        return ValidateDependenciesResponse(
            is_valid=len(errors) == 0,
            errors=errors,
            total_errors=sum(len(errs) for errs in errors.values())
        )


class DetectCyclesUseCase:
    """
    Use case: Detect cycles in task dependency graph.
    
    Uses DFS algorithm to find all circular dependencies.
    """
    
    def __init__(
        self,
        task_node_repo: TaskNodeRepository,
        event_publisher: Any = None
    ):
        self.task_node_repo = task_node_repo
        self.event_publisher = event_publisher
        self.detector = CycleDetector()
    
    async def execute(self, request: DetectCyclesRequest) -> DetectCyclesResponse:
        """Execute cycle detection"""
        # Get all nodes for decomposition
        nodes = await self.task_node_repo.find_by_decomposition(request.decomposition_id)
        
        # Convert to dict
        nodes_dict = {node.id.value: node for node in nodes}
        
        # Detect cycles
        cycles = self.detector.detect_cycles(nodes_dict)
        
        # Convert to CycleInfo
        cycle_infos = [
            CycleInfo(
                cycle_nodes=cycle,
                cycle_description=self.detector.get_cycle_description(cycle, nodes_dict)
            )
            for cycle in cycles
        ]
        
        # Publish event if cycles found
        if cycles and self.event_publisher:
            for cycle in cycles:
                event = CycleDetected.create(
                    decomposition_id=request.decomposition_id,
                    cycle_nodes=cycle,
                    cycle_description=self.detector.get_cycle_description(cycle, nodes_dict)
                )
                await self.event_publisher.publish(event)
        
        return DetectCyclesResponse(
            has_cycles=len(cycles) > 0,
            cycles=cycle_infos,
            cycle_count=len(cycles)
        )


class GetDecompositionStatusUseCase:
    """
    Use case: Get decomposition status and statistics.
    """
    
    def __init__(
        self,
        task_node_repo: TaskNodeRepository,
        decomposition_repo: TaskDecompositionRepository
    ):
        self.task_node_repo = task_node_repo
        self.decomposition_repo = decomposition_repo
        self.integrity_enforcer = IntegrityEnforcer()
    
    async def execute(self, request: GetDecompositionStatusRequest) -> GetDecompositionStatusResponse:
        """Execute status retrieval"""
        # Get decomposition
        decomposition = await self.decomposition_repo.find_by_id(request.decomposition_id)
        if not decomposition:
            raise ValueError(f"Decomposition {request.decomposition_id} not found")
        
        # Get all nodes
        nodes = await self.task_node_repo.find_by_decomposition(request.decomposition_id)
        nodes_dict = {node.id.value: node for node in nodes}
        
        # Check integrity
        is_valid, errors = self.integrity_enforcer.validate_decomposition(
            decomposition, nodes_dict
        )
        
        # Get statistics
        stats = decomposition.get_statistics()
        stats["actual_node_count"] = len(nodes)
        
        # Convert to DTO
        decomposition_dto = DecompositionDTO(
            id=decomposition.id,
            workflow_execution_id=decomposition.workflow_execution_id,
            root_task_name=decomposition.root_task_name,
            root_node_id=decomposition.root_node_id,
            strategy=decomposition.strategy.value,
            total_nodes=decomposition.total_nodes,
            max_depth_reached=decomposition.max_depth_reached,
            leaf_node_count=decomposition.leaf_node_count,
            is_complete=decomposition.is_complete,
            has_cycles=decomposition.has_cycles,
            created_at=decomposition.created_at,
            completed_at=decomposition.completed_at
        )
        
        return GetDecompositionStatusResponse(
            decomposition=decomposition_dto,
            statistics=stats,
            integrity_check={
                "is_valid": is_valid,
                "errors": errors,
                "error_count": len(errors)
            }
        )


class RegisterTaskNodeUseCase:
    """
    Use case: Register a new task node in decomposition.
    """
    
    def __init__(
        self,
        task_node_repo: TaskNodeRepository,
        decomposition_repo: TaskDecompositionRepository
    ):
        self.task_node_repo = task_node_repo
        self.decomposition_repo = decomposition_repo
    
    async def execute(self, request: RegisterTaskNodeRequest) -> RegisterTaskNodeResponse:
        """Execute task node registration"""
        try:
            # Get decomposition
            decomposition = await self.decomposition_repo.find_by_id(request.decomposition_id)
            if not decomposition:
                return RegisterTaskNodeResponse(
                    task_node=None,
                    success=False,
                    message=f"Decomposition {request.decomposition_id} not found"
                )
            
            # Create task node
            task_node = TaskNode.create(
                decomposition_id=request.decomposition_id,
                name=request.name,
                description=request.description,
                parent_id=request.parent_id,
                depth_level=request.depth_level,
                estimated_complexity=request.estimated_complexity,
                tags=request.tags or [],
                metadata=request.metadata or {}
            )
            
            # Save task node
            await self.task_node_repo.save(task_node)
            
            # Register with decomposition
            decomposition.register_node(task_node.id.value, request.depth_level)
            
            # If leaf node (complexity < 0.3), register as leaf
            if request.estimated_complexity < 0.3:
                decomposition.register_leaf_node(task_node.id.value)
            
            await self.decomposition_repo.save(decomposition)
            
            # Convert to DTO
            task_node_dto = TaskNodeDTO(
                id=task_node.id.value,
                decomposition_id=task_node.decomposition_id,
                name=task_node.name,
                description=task_node.description,
                parent_id=task_node.parent_id,
                depth_level=task_node.depth_level,
                estimated_complexity=task_node.estimated_complexity,
                status=task_node.status.value,
                is_leaf=task_node.is_leaf,
                child_count=task_node.actual_subtask_count,
                dependency_count=len(task_node.dependencies),
                tags=task_node.tags,
                created_at=task_node.created_at,
                metadata=task_node.metadata
            )
            
            return RegisterTaskNodeResponse(
                task_node=task_node_dto,
                success=True,
                message="Task node registered successfully"
            )
            
        except Exception as e:
            return RegisterTaskNodeResponse(
                task_node=None,
                success=False,
                message=f"Failed to register task node: {str(e)}"
            )


class GetTaskNodesUseCase:
    """
    Use case: Get task nodes with filtering.
    """
    
    def __init__(self, task_node_repo: TaskNodeRepository):
        self.task_node_repo = task_node_repo
    
    async def execute(self, request: GetTaskNodesRequest) -> GetTaskNodesResponse:
        """Execute task node retrieval"""
        # Get nodes based on filters
        if request.parent_id:
            nodes = await self.task_node_repo.find_by_parent(request.parent_id)
        elif request.depth_level is not None:
            nodes = await self.task_node_repo.find_by_depth_level(
                request.decomposition_id,
                request.depth_level
            )
        else:
            nodes = await self.task_node_repo.find_by_decomposition(
                request.decomposition_id,
                limit=request.limit
            )
        
        # Filter by status if specified
        if request.status:
            nodes = [n for n in nodes if n.status == request.status]
        
        # Convert to DTOs
        node_dtos = [
            TaskNodeDTO(
                id=node.id.value,
                decomposition_id=node.decomposition_id,
                name=node.name,
                description=node.description,
                parent_id=node.parent_id,
                depth_level=node.depth_level,
                estimated_complexity=node.estimated_complexity,
                status=node.status.value,
                is_leaf=node.is_leaf,
                child_count=node.actual_subtask_count,
                dependency_count=len(node.dependencies),
                tags=node.tags,
                created_at=node.created_at,
                metadata=node.metadata
            )
            for node in nodes
        ]
        
        return GetTaskNodesResponse(
            nodes=node_dtos,
            total_count=len(node_dtos),
            filtered_count=len(node_dtos)
        )
