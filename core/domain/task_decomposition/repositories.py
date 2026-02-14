"""
Task Decomposition Domain - Repositories

Repository interfaces for task decomposition persistence.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from .entities import TaskNode, TaskDecomposition


class TaskNodeRepository(ABC):
    """
    Repository interface for TaskNode persistence.
    """
    
    @abstractmethod
    async def save(self, task_node: TaskNode) -> None:
        """
        Save a task node.
        
        Args:
            task_node: Task node to save
        """
        pass
    
    @abstractmethod
    async def find_by_id(self, task_node_id: UUID) -> Optional[TaskNode]:
        """
        Find task node by ID.
        
        Args:
            task_node_id: Task node ID
            
        Returns:
            Task node if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def find_by_decomposition(
        self,
        decomposition_id: UUID,
        limit: int = 1000
    ) -> List[TaskNode]:
        """
        Find all task nodes for a decomposition.
        
        Args:
            decomposition_id: Decomposition ID
            limit: Maximum nodes to return
            
        Returns:
            List of task nodes
        """
        pass
    
    @abstractmethod
    async def find_by_parent(
        self,
        parent_id: UUID
    ) -> List[TaskNode]:
        """
        Find all child nodes of a parent.
        
        Args:
            parent_id: Parent node ID
            
        Returns:
            List of child task nodes
        """
        pass
    
    @abstractmethod
    async def find_leaf_nodes(
        self,
        decomposition_id: UUID
    ) -> List[TaskNode]:
        """
        Find all leaf nodes in a decomposition.
        
        Args:
            decomposition_id: Decomposition ID
            
        Returns:
            List of leaf task nodes
        """
        pass
    
    @abstractmethod
    async def find_by_depth_level(
        self,
        decomposition_id: UUID,
        depth_level: int
    ) -> List[TaskNode]:
        """
        Find all nodes at a specific depth level.
        
        Args:
            decomposition_id: Decomposition ID
            depth_level: Depth level
            
        Returns:
            List of task nodes at depth level
        """
        pass
    
    @abstractmethod
    async def find_root_node(
        self,
        decomposition_id: UUID
    ) -> Optional[TaskNode]:
        """
        Find the root node of a decomposition.
        
        Args:
            decomposition_id: Decomposition ID
            
        Returns:
            Root task node if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def delete(self, task_node_id: UUID) -> bool:
        """
        Delete a task node.
        
        Args:
            task_node_id: Task node ID
            
        Returns:
            True if deleted, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_node_count(
        self,
        decomposition_id: UUID
    ) -> int:
        """
        Get total node count for a decomposition.
        
        Args:
            decomposition_id: Decomposition ID
            
        Returns:
            Node count
        """
        pass
    
    @abstractmethod
    async def get_max_depth(
        self,
        decomposition_id: UUID
    ) -> int:
        """
        Get maximum depth reached in a decomposition.
        
        Args:
            decomposition_id: Decomposition ID
            
        Returns:
            Maximum depth level
        """
        pass


class TaskDecompositionRepository(ABC):
    """
    Repository interface for TaskDecomposition persistence.
    """
    
    @abstractmethod
    async def save(self, decomposition: TaskDecomposition) -> None:
        """
        Save a task decomposition.
        
        Args:
            decomposition: Task decomposition to save
        """
        pass
    
    @abstractmethod
    async def find_by_id(
        self,
        decomposition_id: UUID
    ) -> Optional[TaskDecomposition]:
        """
        Find task decomposition by ID.
        
        Args:
            decomposition_id: Decomposition ID
            
        Returns:
            Task decomposition if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def find_by_workflow_execution(
        self,
        workflow_execution_id: UUID
    ) -> List[TaskDecomposition]:
        """
        Find all decompositions for a workflow execution.
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            List of task decompositions
        """
        pass
    
    @abstractmethod
    async def find_latest_by_workflow_execution(
        self,
        workflow_execution_id: UUID
    ) -> Optional[TaskDecomposition]:
        """
        Find latest decomposition for a workflow execution.
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            Latest task decomposition if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def delete(self, decomposition_id: UUID) -> bool:
        """
        Delete a task decomposition.
        
        Args:
            decomposition_id: Decomposition ID
            
        Returns:
            True if deleted, False otherwise
        """
        pass
    
    @abstractmethod
    async def exists(self, decomposition_id: UUID) -> bool:
        """
        Check if decomposition exists.
        
        Args:
            decomposition_id: Decomposition ID
            
        Returns:
            True if exists, False otherwise
        """
        pass
