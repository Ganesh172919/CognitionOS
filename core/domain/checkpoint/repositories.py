"""
Checkpoint Domain - Repository Interfaces

Repository abstractions for checkpoint persistence.
NO implementation details - only interfaces.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from .entities import Checkpoint


class CheckpointRepository(ABC):
    """
    Repository interface for checkpoint persistence.
    
    Implementations will use Redis (fast-layer) + PostgreSQL (durable-layer).
    """

    @abstractmethod
    async def save(self, checkpoint: Checkpoint) -> None:
        """
        Save a checkpoint.
        
        Args:
            checkpoint: Checkpoint to save
            
        Raises:
            RepositoryError: If save fails
        """
        pass

    @abstractmethod
    async def find_by_id(self, checkpoint_id: UUID) -> Optional[Checkpoint]:
        """
        Find checkpoint by ID.
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            Checkpoint if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_workflow_execution(
        self,
        workflow_execution_id: UUID,
        limit: Optional[int] = None,
    ) -> List[Checkpoint]:
        """
        Find checkpoints for a workflow execution.
        
        Args:
            workflow_execution_id: Workflow execution ID
            limit: Maximum number of checkpoints to return (newest first)
            
        Returns:
            List of checkpoints ordered by checkpoint_number DESC
        """
        pass

    @abstractmethod
    async def find_latest_by_workflow_execution(
        self,
        workflow_execution_id: UUID,
    ) -> Optional[Checkpoint]:
        """
        Find the latest checkpoint for a workflow execution.
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            Latest checkpoint if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_workflow_and_number(
        self,
        workflow_execution_id: UUID,
        checkpoint_number: int,
    ) -> Optional[Checkpoint]:
        """
        Find checkpoint by workflow execution ID and checkpoint number.
        
        Args:
            workflow_execution_id: Workflow execution ID
            checkpoint_number: Checkpoint number
            
        Returns:
            Checkpoint if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete(self, checkpoint_id: UUID) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def delete_old_checkpoints(
        self,
        workflow_execution_id: UUID,
        keep_count: int,
    ) -> int:
        """
        Delete old checkpoints, keeping only the N most recent.
        
        Args:
            workflow_execution_id: Workflow execution ID
            keep_count: Number of recent checkpoints to keep
            
        Returns:
            Number of checkpoints deleted
        """
        pass

    @abstractmethod
    async def get_checkpoint_count(
        self,
        workflow_execution_id: UUID,
    ) -> int:
        """
        Get count of checkpoints for a workflow execution.
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            Number of checkpoints
        """
        pass

    @abstractmethod
    async def exists(
        self,
        workflow_execution_id: UUID,
        checkpoint_number: int,
    ) -> bool:
        """
        Check if a checkpoint exists.
        
        Args:
            workflow_execution_id: Workflow execution ID
            checkpoint_number: Checkpoint number
            
        Returns:
            True if checkpoint exists, False otherwise
        """
        pass
