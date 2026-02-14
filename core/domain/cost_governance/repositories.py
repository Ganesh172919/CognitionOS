"""
Cost Governance Domain - Repository Interfaces

Repository abstractions for cost governance persistence.
NO implementation details - only interfaces.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from .entities import CostEntry, OperationType, WorkflowBudget


class WorkflowBudgetRepository(ABC):
    """
    Repository interface for workflow budget persistence.
    
    Implementations will use PostgreSQL for durable storage.
    """

    @abstractmethod
    async def save(self, budget: WorkflowBudget) -> None:
        """
        Save a workflow budget.
        
        Args:
            budget: Budget to save
            
        Raises:
            RepositoryError: If save fails
        """
        pass

    @abstractmethod
    async def find_by_id(self, budget_id: UUID) -> Optional[WorkflowBudget]:
        """
        Find budget by ID.
        
        Args:
            budget_id: Budget ID
            
        Returns:
            Budget if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_workflow_execution(
        self,
        workflow_execution_id: UUID,
    ) -> Optional[WorkflowBudget]:
        """
        Find budget for a workflow execution.
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            Budget if found, None otherwise
        """
        pass

    @abstractmethod
    async def update(self, budget: WorkflowBudget) -> None:
        """
        Update an existing budget.
        
        Args:
            budget: Budget to update
            
        Raises:
            RepositoryError: If update fails
        """
        pass

    @abstractmethod
    async def delete(self, budget_id: UUID) -> bool:
        """
        Delete a budget.
        
        Args:
            budget_id: Budget ID
            
        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def exists(self, workflow_execution_id: UUID) -> bool:
        """
        Check if a budget exists for a workflow execution.
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            True if budget exists, False otherwise
        """
        pass


class CostTrackingRepository(ABC):
    """
    Repository interface for cost entry persistence.
    
    Implementations will use PostgreSQL for durable storage with time-series optimizations.
    """

    @abstractmethod
    async def save(self, cost_entry: CostEntry) -> None:
        """
        Save a cost entry.
        
        Args:
            cost_entry: Cost entry to save
            
        Raises:
            RepositoryError: If save fails
        """
        pass

    @abstractmethod
    async def find_by_id(self, cost_entry_id: UUID) -> Optional[CostEntry]:
        """
        Find cost entry by ID.
        
        Args:
            cost_entry_id: Cost entry ID
            
        Returns:
            Cost entry if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_workflow_execution(
        self,
        workflow_execution_id: UUID,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[CostEntry]:
        """
        Find cost entries for a workflow execution.
        
        Args:
            workflow_execution_id: Workflow execution ID
            limit: Maximum number of entries to return
            offset: Number of entries to skip
            
        Returns:
            List of cost entries ordered by created_at DESC
        """
        pass

    @abstractmethod
    async def find_by_agent(
        self,
        agent_id: UUID,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[CostEntry]:
        """
        Find cost entries for a specific agent.
        
        Args:
            agent_id: Agent ID
            limit: Maximum number of entries to return
            offset: Number of entries to skip
            
        Returns:
            List of cost entries ordered by created_at DESC
        """
        pass

    @abstractmethod
    async def find_by_operation_type(
        self,
        workflow_execution_id: UUID,
        operation_type: OperationType,
        limit: Optional[int] = None,
    ) -> List[CostEntry]:
        """
        Find cost entries by operation type.
        
        Args:
            workflow_execution_id: Workflow execution ID
            operation_type: Operation type to filter by
            limit: Maximum number of entries to return
            
        Returns:
            List of cost entries
        """
        pass

    @abstractmethod
    async def get_total_cost(
        self,
        workflow_execution_id: UUID,
    ) -> float:
        """
        Get total cost for a workflow execution.
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            Total cost amount
        """
        pass

    @abstractmethod
    async def get_cost_by_operation_type(
        self,
        workflow_execution_id: UUID,
    ) -> dict[OperationType, float]:
        """
        Get cost breakdown by operation type.
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            Dictionary mapping operation type to total cost
        """
        pass

    @abstractmethod
    async def get_cost_by_agent(
        self,
        workflow_execution_id: UUID,
    ) -> dict[UUID, float]:
        """
        Get cost breakdown by agent.
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            Dictionary mapping agent ID to total cost
        """
        pass

    @abstractmethod
    async def delete_by_workflow_execution(
        self,
        workflow_execution_id: UUID,
    ) -> int:
        """
        Delete all cost entries for a workflow execution.
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            Number of entries deleted
        """
        pass
