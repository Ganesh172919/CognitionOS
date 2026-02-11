"""
Agent Domain - Repository Interfaces

Pure interfaces for agent persistence.
Implementations provided by infrastructure layer.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from .entities import (
    Agent,
    AgentDefinition,
    AgentId,
    AgentRole,
    AgentStatus
)


class AgentRepository(ABC):
    """
    Repository interface for Agent aggregate.

    Handles persistence of agent instances.
    """

    @abstractmethod
    async def save(self, agent: Agent) -> None:
        """
        Persist agent instance.

        Args:
            agent: Agent to save
        """
        pass

    @abstractmethod
    async def get_by_id(self, agent_id: AgentId) -> Optional[Agent]:
        """
        Retrieve agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_by_status(
        self,
        status: AgentStatus,
        limit: int = 100
    ) -> List[Agent]:
        """
        Get agents by status.

        Args:
            status: Agent status filter
            limit: Maximum number of agents to return

        Returns:
            List of agents matching status
        """
        pass

    @abstractmethod
    async def get_idle_agents(
        self,
        required_capabilities: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Agent]:
        """
        Get idle agents, optionally filtered by capabilities.

        Args:
            required_capabilities: Optional capability filter
            limit: Maximum number of agents to return

        Returns:
            List of idle agents
        """
        pass

    @abstractmethod
    async def get_active_agents(self) -> List[Agent]:
        """
        Get all currently active agents.

        Returns:
            List of active agents (ASSIGNED, REASONING, EXECUTING, VALIDATING)
        """
        pass

    @abstractmethod
    async def get_by_task(self, task_id: UUID) -> Optional[Agent]:
        """
        Get agent assigned to a specific task.

        Args:
            task_id: Task identifier

        Returns:
            Agent if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete(self, agent_id: AgentId) -> bool:
        """
        Delete agent.

        Args:
            agent_id: Agent identifier

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def count_by_status(self, status: AgentStatus) -> int:
        """
        Count agents by status.

        Args:
            status: Agent status

        Returns:
            Number of agents in that status
        """
        pass


class AgentDefinitionRepository(ABC):
    """
    Repository interface for AgentDefinition aggregate.

    Handles persistence of agent blueprints/templates.
    """

    @abstractmethod
    async def save(self, definition: AgentDefinition) -> None:
        """
        Persist agent definition.

        Args:
            definition: Agent definition to save
        """
        pass

    @abstractmethod
    async def get_by_id(self, definition_id: UUID) -> Optional[AgentDefinition]:
        """
        Retrieve agent definition by ID.

        Args:
            definition_id: Definition identifier

        Returns:
            Agent definition if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[AgentDefinition]:
        """
        Retrieve agent definition by name.

        Args:
            name: Definition name

        Returns:
            Agent definition if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_by_role(self, role: AgentRole) -> List[AgentDefinition]:
        """
        Get all definitions for a specific role.

        Args:
            role: Agent role

        Returns:
            List of agent definitions
        """
        pass

    @abstractmethod
    async def get_compatible_definitions(
        self,
        required_capabilities: List[str]
    ) -> List[AgentDefinition]:
        """
        Get agent definitions that support required capabilities.

        Args:
            required_capabilities: Required capabilities

        Returns:
            List of compatible agent definitions
        """
        pass

    @abstractmethod
    async def list_all(self, limit: int = 100) -> List[AgentDefinition]:
        """
        List all agent definitions.

        Args:
            limit: Maximum number to return

        Returns:
            List of agent definitions
        """
        pass

    @abstractmethod
    async def delete(self, definition_id: UUID) -> bool:
        """
        Delete agent definition.

        Args:
            definition_id: Definition identifier

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def exists(self, name: str, version: str) -> bool:
        """
        Check if agent definition exists.

        Args:
            name: Definition name
            version: Definition version

        Returns:
            True if exists, False otherwise
        """
        pass
