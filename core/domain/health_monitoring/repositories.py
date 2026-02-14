"""
Health Monitoring Domain - Repository Interfaces

Repository abstractions for health monitoring persistence.
NO implementation details - only interfaces.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from .entities import AgentHealthStatus, AgentHealthIncident


class AgentHealthRepository(ABC):
    """
    Repository interface for agent health status persistence.
    
    Implementations will use Redis (fast-layer) + PostgreSQL (durable-layer).
    """

    @abstractmethod
    async def save(self, health_status: AgentHealthStatus) -> None:
        """
        Save agent health status.
        
        Args:
            health_status: Health status to save
            
        Raises:
            RepositoryError: If save fails
        """
        pass

    @abstractmethod
    async def find_by_id(self, health_status_id: UUID) -> Optional[AgentHealthStatus]:
        """
        Find health status by ID.
        
        Args:
            health_status_id: Health status ID
            
        Returns:
            Health status if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_agent(self, agent_id: UUID) -> Optional[AgentHealthStatus]:
        """
        Find current health status for an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Current health status if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_workflow_execution(
        self,
        workflow_execution_id: UUID,
    ) -> List[AgentHealthStatus]:
        """
        Find all health statuses for a workflow execution.
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            List of health statuses
        """
        pass

    @abstractmethod
    async def find_failing_agents(
        self,
        workflow_execution_id: Optional[UUID] = None,
    ) -> List[AgentHealthStatus]:
        """
        Find agents with failing health status.
        
        Args:
            workflow_execution_id: Optional workflow execution ID filter
            
        Returns:
            List of failing agent health statuses
        """
        pass

    @abstractmethod
    async def find_stale_heartbeats(
        self,
        threshold_seconds: int = 30,
    ) -> List[AgentHealthStatus]:
        """
        Find agents with stale heartbeats.
        
        Args:
            threshold_seconds: Heartbeat staleness threshold
            
        Returns:
            List of health statuses with stale heartbeats
        """
        pass

    @abstractmethod
    async def delete(self, health_status_id: UUID) -> bool:
        """
        Delete a health status.
        
        Args:
            health_status_id: Health status ID
            
        Returns:
            True if deleted, False if not found
        """
        pass


class HealthIncidentRepository(ABC):
    """
    Repository interface for health incident persistence.
    
    Implementations will use PostgreSQL for durable storage.
    """

    @abstractmethod
    async def save(self, incident: AgentHealthIncident) -> None:
        """
        Save health incident.
        
        Args:
            incident: Incident to save
            
        Raises:
            RepositoryError: If save fails
        """
        pass

    @abstractmethod
    async def find_by_id(self, incident_id: UUID) -> Optional[AgentHealthIncident]:
        """
        Find incident by ID.
        
        Args:
            incident_id: Incident ID
            
        Returns:
            Incident if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_agent(
        self,
        agent_id: UUID,
        limit: Optional[int] = None,
    ) -> List[AgentHealthIncident]:
        """
        Find incidents for an agent.
        
        Args:
            agent_id: Agent ID
            limit: Maximum number of incidents to return (newest first)
            
        Returns:
            List of incidents ordered by created_at DESC
        """
        pass

    @abstractmethod
    async def find_by_workflow_execution(
        self,
        workflow_execution_id: UUID,
        limit: Optional[int] = None,
    ) -> List[AgentHealthIncident]:
        """
        Find incidents for a workflow execution.
        
        Args:
            workflow_execution_id: Workflow execution ID
            limit: Maximum number of incidents to return (newest first)
            
        Returns:
            List of incidents ordered by created_at DESC
        """
        pass

    @abstractmethod
    async def find_open_incidents(
        self,
        workflow_execution_id: Optional[UUID] = None,
    ) -> List[AgentHealthIncident]:
        """
        Find open (unresolved) incidents.
        
        Args:
            workflow_execution_id: Optional workflow execution ID filter
            
        Returns:
            List of open incidents
        """
        pass

    @abstractmethod
    async def find_critical_incidents(
        self,
        workflow_execution_id: Optional[UUID] = None,
    ) -> List[AgentHealthIncident]:
        """
        Find critical severity incidents.
        
        Args:
            workflow_execution_id: Optional workflow execution ID filter
            
        Returns:
            List of critical incidents
        """
        pass

    @abstractmethod
    async def get_incident_count(
        self,
        agent_id: UUID,
    ) -> int:
        """
        Get count of incidents for an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Number of incidents
        """
        pass

    @abstractmethod
    async def delete(self, incident_id: UUID) -> bool:
        """
        Delete an incident.
        
        Args:
            incident_id: Incident ID
            
        Returns:
            True if deleted, False if not found
        """
        pass
