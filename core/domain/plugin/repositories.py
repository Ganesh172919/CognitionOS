"""
Plugin Domain - Repository Interfaces

Pure interfaces for plugin persistence.
Implementations provided by infrastructure layer.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from .entities import (
    Plugin,
    PluginExecution,
    PluginId,
    PluginInstallation,
    PluginRuntime,
    PluginStatus,
    ExecutionStatus,
    TrustLevel
)


class PluginRepository(ABC):
    """
    Repository interface for Plugin aggregate.
    
    Handles persistence of plugin definitions and metadata.
    """

    @abstractmethod
    async def save(self, plugin: Plugin) -> None:
        """
        Persist plugin.
        
        Args:
            plugin: Plugin to save
        """
        pass

    @abstractmethod
    async def get_by_id(self, plugin_id: PluginId) -> Optional[Plugin]:
        """
        Retrieve plugin by ID.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Plugin if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_by_name(
        self,
        name: str,
        tenant_id: Optional[UUID] = None
    ) -> Optional[Plugin]:
        """
        Retrieve plugin by name.
        
        Args:
            name: Plugin name
            tenant_id: Tenant ID (None for global plugins)
            
        Returns:
            Plugin if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete(self, plugin_id: PluginId) -> bool:
        """
        Delete plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def list_by_status(
        self,
        status: PluginStatus,
        limit: int = 100,
        offset: int = 0
    ) -> List[Plugin]:
        """
        List plugins by status.
        
        Args:
            status: Plugin status filter
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of plugins
        """
        pass

    @abstractmethod
    async def list_by_tenant(
        self,
        tenant_id: Optional[UUID],
        limit: int = 100,
        offset: int = 0
    ) -> List[Plugin]:
        """
        List plugins by tenant.
        
        Args:
            tenant_id: Tenant ID (None for global plugins)
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of plugins
        """
        pass

    @abstractmethod
    async def list_approved(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Plugin]:
        """
        List approved (active) plugins available in marketplace.
        
        Args:
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of approved plugins
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        runtime: Optional[PluginRuntime] = None,
        min_trust_score: Optional[int] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Plugin]:
        """
        Search plugins by various criteria.
        
        Args:
            query: Search query (matches name, description, author)
            runtime: Filter by runtime
            min_trust_score: Minimum trust score
            tags: Filter by tags
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of matching plugins
        """
        pass

    @abstractmethod
    async def get_by_runtime(
        self,
        runtime: PluginRuntime,
        limit: int = 100,
        offset: int = 0
    ) -> List[Plugin]:
        """
        Get plugins by runtime type.
        
        Args:
            runtime: Plugin runtime
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of plugins
        """
        pass

    @abstractmethod
    async def get_popular(
        self,
        limit: int = 20,
        min_trust_score: int = 50
    ) -> List[Plugin]:
        """
        Get most popular plugins (by install count).
        
        Args:
            limit: Maximum number of results
            min_trust_score: Minimum trust score threshold
            
        Returns:
            List of popular plugins
        """
        pass

    @abstractmethod
    async def get_by_trust_level(
        self,
        trust_level: TrustLevel,
        limit: int = 100,
        offset: int = 0
    ) -> List[Plugin]:
        """
        Get plugins by trust level.
        
        Args:
            trust_level: Trust level filter
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of plugins
        """
        pass

    @abstractmethod
    async def count_by_status(self, status: PluginStatus) -> int:
        """
        Count plugins by status.
        
        Args:
            status: Plugin status
            
        Returns:
            Number of plugins with given status
        """
        pass


class PluginExecutionRepository(ABC):
    """
    Repository interface for PluginExecution entity.
    
    Handles persistence of plugin execution history and tracking.
    """

    @abstractmethod
    async def save(self, execution: PluginExecution) -> None:
        """
        Persist plugin execution.
        
        Args:
            execution: Plugin execution to save
        """
        pass

    @abstractmethod
    async def get_by_id(self, execution_id: UUID) -> Optional[PluginExecution]:
        """
        Retrieve execution by ID.
        
        Args:
            execution_id: Execution identifier
            
        Returns:
            PluginExecution if found, None otherwise
        """
        pass

    @abstractmethod
    async def list_by_plugin(
        self,
        plugin_id: PluginId,
        limit: int = 100,
        offset: int = 0
    ) -> List[PluginExecution]:
        """
        List executions for a plugin.
        
        Args:
            plugin_id: Plugin identifier
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of executions
        """
        pass

    @abstractmethod
    async def list_by_tenant(
        self,
        tenant_id: UUID,
        limit: int = 100,
        offset: int = 0
    ) -> List[PluginExecution]:
        """
        List executions for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of executions
        """
        pass

    @abstractmethod
    async def list_by_status(
        self,
        status: ExecutionStatus,
        limit: int = 100,
        offset: int = 0
    ) -> List[PluginExecution]:
        """
        List executions by status.
        
        Args:
            status: Execution status filter
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of executions
        """
        pass

    @abstractmethod
    async def list_by_plugin_and_tenant(
        self,
        plugin_id: PluginId,
        tenant_id: UUID,
        limit: int = 100,
        offset: int = 0
    ) -> List[PluginExecution]:
        """
        List executions for a specific plugin and tenant.
        
        Args:
            plugin_id: Plugin identifier
            tenant_id: Tenant identifier
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of executions
        """
        pass

    @abstractmethod
    async def get_recent_by_tenant(
        self,
        tenant_id: UUID,
        limit: int = 50
    ) -> List[PluginExecution]:
        """
        Get recent executions for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            limit: Maximum number of results
            
        Returns:
            List of recent executions, newest first
        """
        pass

    @abstractmethod
    async def get_running_executions(
        self,
        tenant_id: Optional[UUID] = None
    ) -> List[PluginExecution]:
        """
        Get currently running executions.
        
        Args:
            tenant_id: Optional tenant filter
            
        Returns:
            List of running executions
        """
        pass

    @abstractmethod
    async def count_by_plugin(
        self,
        plugin_id: PluginId,
        status: Optional[ExecutionStatus] = None
    ) -> int:
        """
        Count executions for a plugin.
        
        Args:
            plugin_id: Plugin identifier
            status: Optional status filter
            
        Returns:
            Number of executions
        """
        pass

    @abstractmethod
    async def get_execution_stats(
        self,
        plugin_id: PluginId,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> dict:
        """
        Get execution statistics for a plugin.
        
        Args:
            plugin_id: Plugin identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dictionary containing stats (total, success, failure, avg_duration, etc.)
        """
        pass

    @abstractmethod
    async def delete_old_executions(
        self,
        before_date: datetime,
        limit: int = 1000
    ) -> int:
        """
        Delete old execution records.
        
        Args:
            before_date: Delete executions before this date
            limit: Maximum number to delete in one call
            
        Returns:
            Number of deleted executions
        """
        pass


class PluginInstallationRepository(ABC):
    """
    Repository interface for PluginInstallation entity.
    
    Handles persistence of plugin installations per tenant.
    """

    @abstractmethod
    async def save(self, installation: PluginInstallation) -> None:
        """
        Persist plugin installation.
        
        Args:
            installation: Plugin installation to save
        """
        pass

    @abstractmethod
    async def get_by_id(self, installation_id: UUID) -> Optional[PluginInstallation]:
        """
        Retrieve installation by ID.
        
        Args:
            installation_id: Installation identifier
            
        Returns:
            PluginInstallation if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_by_plugin_and_tenant(
        self,
        plugin_id: PluginId,
        tenant_id: UUID
    ) -> Optional[PluginInstallation]:
        """
        Get installation for a specific plugin and tenant.
        
        Args:
            plugin_id: Plugin identifier
            tenant_id: Tenant identifier
            
        Returns:
            PluginInstallation if found, None otherwise
        """
        pass

    @abstractmethod
    async def list_by_tenant(
        self,
        tenant_id: UUID,
        enabled_only: bool = False
    ) -> List[PluginInstallation]:
        """
        List all plugin installations for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            enabled_only: If True, only return enabled installations
            
        Returns:
            List of installations
        """
        pass

    @abstractmethod
    async def list_by_plugin(
        self,
        plugin_id: PluginId,
        limit: int = 100,
        offset: int = 0
    ) -> List[PluginInstallation]:
        """
        List all installations of a specific plugin.
        
        Args:
            plugin_id: Plugin identifier
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of installations
        """
        pass

    @abstractmethod
    async def delete(self, installation_id: UUID) -> bool:
        """
        Delete plugin installation.
        
        Args:
            installation_id: Installation identifier
            
        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def count_by_plugin(self, plugin_id: PluginId) -> int:
        """
        Count installations for a plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Number of installations
        """
        pass

    @abstractmethod
    async def is_installed(
        self,
        plugin_id: PluginId,
        tenant_id: UUID
    ) -> bool:
        """
        Check if plugin is installed for a tenant.
        
        Args:
            plugin_id: Plugin identifier
            tenant_id: Tenant identifier
            
        Returns:
            True if installed, False otherwise
        """
        pass
