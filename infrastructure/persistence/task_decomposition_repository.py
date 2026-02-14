"""
Task Decomposition Infrastructure - PostgreSQL Repository Implementation

Concrete implementation of TaskDecomposition repositories using PostgreSQL.
"""

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, delete, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from core.domain.task_decomposition.entities import (
    TaskNode,
    TaskNodeId,
    TaskDecomposition,
    Dependency,
    DependencyType,
    DecompositionStrategy,
    TaskNodeStatus,
)
from core.domain.task_decomposition.repositories import (
    TaskNodeRepository,
    TaskDecompositionRepository,
)

from infrastructure.persistence.task_decomposition_models import (
    TaskNodeModel,
    TaskDecompositionModel,
)


class PostgreSQLTaskNodeRepository(TaskNodeRepository):
    """
    PostgreSQL implementation of TaskNodeRepository.
    
    Maps between TaskNode domain entities and SQLAlchemy models.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, task_node: TaskNode) -> None:
        """Persist task node to database"""
        # Check if already exists
        stmt = select(TaskNodeModel).where(TaskNodeModel.id == task_node.id.value)
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update existing
            self._update_model(existing, task_node)
        else:
            # Create new
            model = self._to_model(task_node)
            self.session.add(model)
        
        await self.session.flush()
    
    async def find_by_id(self, task_node_id: UUID) -> Optional[TaskNode]:
        """Retrieve task node by ID"""
        stmt = select(TaskNodeModel).where(TaskNodeModel.id == task_node_id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._to_entity(model)
    
    async def find_by_decomposition(
        self,
        decomposition_id: UUID,
        limit: int = 1000
    ) -> List[TaskNode]:
        """Find all task nodes for a decomposition"""
        stmt = (
            select(TaskNodeModel)
            .where(TaskNodeModel.decomposition_id == decomposition_id)
            .order_by(TaskNodeModel.depth_level, TaskNodeModel.created_at)
            .limit(limit)
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    async def find_by_parent(self, parent_id: UUID) -> List[TaskNode]:
        """Find all child nodes of a parent"""
        stmt = (
            select(TaskNodeModel)
            .where(TaskNodeModel.parent_id == parent_id)
            .order_by(TaskNodeModel.created_at)
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    async def find_leaf_nodes(self, decomposition_id: UUID) -> List[TaskNode]:
        """Find all leaf nodes in a decomposition"""
        stmt = (
            select(TaskNodeModel)
            .where(
                and_(
                    TaskNodeModel.decomposition_id == decomposition_id,
                    TaskNodeModel.is_leaf == True
                )
            )
            .order_by(TaskNodeModel.depth_level, TaskNodeModel.created_at)
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    async def find_by_depth_level(
        self,
        decomposition_id: UUID,
        depth_level: int
    ) -> List[TaskNode]:
        """Find all nodes at a specific depth level"""
        stmt = (
            select(TaskNodeModel)
            .where(
                and_(
                    TaskNodeModel.decomposition_id == decomposition_id,
                    TaskNodeModel.depth_level == depth_level
                )
            )
            .order_by(TaskNodeModel.created_at)
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    async def find_root_node(self, decomposition_id: UUID) -> Optional[TaskNode]:
        """Find the root node of a decomposition"""
        stmt = (
            select(TaskNodeModel)
            .where(
                and_(
                    TaskNodeModel.decomposition_id == decomposition_id,
                    TaskNodeModel.depth_level == 0,
                    TaskNodeModel.parent_id == None
                )
            )
            .limit(1)
        )
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._to_entity(model)
    
    async def delete(self, task_node_id: UUID) -> bool:
        """Delete a task node"""
        stmt = delete(TaskNodeModel).where(TaskNodeModel.id == task_node_id)
        result = await self.session.execute(stmt)
        await self.session.flush()
        
        return result.rowcount > 0
    
    async def get_node_count(self, decomposition_id: UUID) -> int:
        """Get total node count for a decomposition"""
        stmt = (
            select(func.count(TaskNodeModel.id))
            .where(TaskNodeModel.decomposition_id == decomposition_id)
        )
        
        result = await self.session.execute(stmt)
        count = result.scalar_one()
        
        return count
    
    async def get_max_depth(self, decomposition_id: UUID) -> int:
        """Get maximum depth reached in a decomposition"""
        stmt = (
            select(func.max(TaskNodeModel.depth_level))
            .where(TaskNodeModel.decomposition_id == decomposition_id)
        )
        
        result = await self.session.execute(stmt)
        max_depth = result.scalar_one()
        
        return max_depth if max_depth is not None else 0
    
    def _to_model(self, entity: TaskNode) -> TaskNodeModel:
        """Convert TaskNode entity to SQLAlchemy model"""
        # Convert dependencies to dict format for JSONB
        dependencies_data = [dep.to_dict() for dep in entity.dependencies]
        
        return TaskNodeModel(
            id=entity.id.value,
            decomposition_id=entity.decomposition_id,
            name=entity.name,
            description=entity.description,
            parent_id=entity.parent_id,
            depth_level=entity.depth_level,
            child_node_ids=list(entity.child_node_ids),
            estimated_complexity=entity.estimated_complexity,
            is_leaf=entity.is_leaf,
            actual_subtask_count=entity.actual_subtask_count,
            status=entity.status.value,
            dependencies=dependencies_data,
            tags=entity.tags,
            metadata=entity.metadata,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            decomposed_at=entity.decomposed_at,
        )
    
    def _update_model(self, model: TaskNodeModel, entity: TaskNode) -> None:
        """Update existing model with entity data"""
        dependencies_data = [dep.to_dict() for dep in entity.dependencies]
        
        model.name = entity.name
        model.description = entity.description
        model.parent_id = entity.parent_id
        model.depth_level = entity.depth_level
        model.child_node_ids = list(entity.child_node_ids)
        model.estimated_complexity = entity.estimated_complexity
        model.is_leaf = entity.is_leaf
        model.actual_subtask_count = entity.actual_subtask_count
        model.status = entity.status.value
        model.dependencies = dependencies_data
        model.tags = entity.tags
        model.metadata = entity.metadata
        model.updated_at = entity.updated_at
        model.decomposed_at = entity.decomposed_at
    
    def _to_entity(self, model: TaskNodeModel) -> TaskNode:
        """Convert SQLAlchemy model to TaskNode entity"""
        # Convert dependencies from JSONB to Dependency objects
        dependencies = [
            Dependency.from_dict(dep_data)
            for dep_data in (model.dependencies or [])
        ]
        
        return TaskNode(
            id=TaskNodeId(model.id),
            decomposition_id=model.decomposition_id,
            name=model.name,
            description=model.description,
            parent_id=model.parent_id,
            depth_level=model.depth_level,
            child_node_ids=set(model.child_node_ids) if model.child_node_ids else set(),
            estimated_complexity=model.estimated_complexity,
            is_leaf=model.is_leaf,
            actual_subtask_count=model.actual_subtask_count,
            status=TaskNodeStatus(model.status),
            dependencies=dependencies,
            tags=model.tags or [],
            metadata=model.metadata or {},
            created_at=model.created_at,
            updated_at=model.updated_at,
            decomposed_at=model.decomposed_at,
        )


class PostgreSQLTaskDecompositionRepository(TaskDecompositionRepository):
    """
    PostgreSQL implementation of TaskDecompositionRepository.
    
    Maps between TaskDecomposition domain entities and SQLAlchemy models.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, decomposition: TaskDecomposition) -> None:
        """Persist task decomposition to database"""
        # Check if already exists
        stmt = select(TaskDecompositionModel).where(
            TaskDecompositionModel.id == decomposition.id
        )
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update existing
            self._update_model(existing, decomposition)
        else:
            # Create new
            model = self._to_model(decomposition)
            self.session.add(model)
        
        await self.session.flush()
    
    async def find_by_id(self, decomposition_id: UUID) -> Optional[TaskDecomposition]:
        """Retrieve task decomposition by ID"""
        stmt = select(TaskDecompositionModel).where(
            TaskDecompositionModel.id == decomposition_id
        )
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._to_entity(model)
    
    async def find_by_workflow_execution(
        self,
        workflow_execution_id: UUID
    ) -> List[TaskDecomposition]:
        """Find all decompositions for a workflow execution"""
        stmt = (
            select(TaskDecompositionModel)
            .where(TaskDecompositionModel.workflow_execution_id == workflow_execution_id)
            .order_by(TaskDecompositionModel.created_at.desc())
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    async def find_latest_by_workflow_execution(
        self,
        workflow_execution_id: UUID
    ) -> Optional[TaskDecomposition]:
        """Find latest decomposition for a workflow execution"""
        stmt = (
            select(TaskDecompositionModel)
            .where(TaskDecompositionModel.workflow_execution_id == workflow_execution_id)
            .order_by(TaskDecompositionModel.created_at.desc())
            .limit(1)
        )
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._to_entity(model)
    
    async def delete(self, decomposition_id: UUID) -> bool:
        """Delete a task decomposition"""
        stmt = delete(TaskDecompositionModel).where(
            TaskDecompositionModel.id == decomposition_id
        )
        result = await self.session.execute(stmt)
        await self.session.flush()
        
        return result.rowcount > 0
    
    async def exists(self, decomposition_id: UUID) -> bool:
        """Check if decomposition exists"""
        stmt = select(func.count(TaskDecompositionModel.id)).where(
            TaskDecompositionModel.id == decomposition_id
        )
        result = await self.session.execute(stmt)
        count = result.scalar_one()
        
        return count > 0
    
    def _to_model(self, entity: TaskDecomposition) -> TaskDecompositionModel:
        """Convert TaskDecomposition entity to SQLAlchemy model"""
        return TaskDecompositionModel(
            id=entity.id,
            workflow_execution_id=entity.workflow_execution_id,
            root_task_name=entity.root_task_name,
            root_task_description=entity.root_task_description,
            root_node_id=entity.root_node_id,
            strategy=entity.strategy.value,
            total_nodes=entity.total_nodes,
            max_depth_reached=entity.max_depth_reached,
            leaf_node_count=entity.leaf_node_count,
            all_node_ids=list(entity.all_node_ids),
            is_complete=entity.is_complete,
            has_cycles=entity.has_cycles,
            created_at=entity.created_at,
            completed_at=entity.completed_at,
            metadata=entity.metadata,
        )
    
    def _update_model(
        self,
        model: TaskDecompositionModel,
        entity: TaskDecomposition
    ) -> None:
        """Update existing model with entity data"""
        model.root_node_id = entity.root_node_id
        model.strategy = entity.strategy.value
        model.total_nodes = entity.total_nodes
        model.max_depth_reached = entity.max_depth_reached
        model.leaf_node_count = entity.leaf_node_count
        model.all_node_ids = list(entity.all_node_ids)
        model.is_complete = entity.is_complete
        model.has_cycles = entity.has_cycles
        model.completed_at = entity.completed_at
        model.metadata = entity.metadata
    
    def _to_entity(self, model: TaskDecompositionModel) -> TaskDecomposition:
        """Convert SQLAlchemy model to TaskDecomposition entity"""
        return TaskDecomposition(
            id=model.id,
            workflow_execution_id=model.workflow_execution_id,
            root_task_name=model.root_task_name,
            root_task_description=model.root_task_description,
            root_node_id=model.root_node_id,
            strategy=DecompositionStrategy(model.strategy),
            total_nodes=model.total_nodes,
            max_depth_reached=model.max_depth_reached,
            leaf_node_count=model.leaf_node_count,
            all_node_ids=set(model.all_node_ids) if model.all_node_ids else set(),
            is_complete=model.is_complete,
            has_cycles=model.has_cycles,
            created_at=model.created_at,
            completed_at=model.completed_at,
            metadata=model.metadata or {},
        )
