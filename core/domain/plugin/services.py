"""
Plugin Domain - Domain Services

Domain services for plugin-related business logic.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import UUID

from .entities import (
    Plugin,
    PluginExecution,
    PluginId,
    PluginInstallation,
    PluginManifest,
    PluginStatus,
    PluginTrustScore,
    TrustFactor,
    TrustLevel,
    ExecutionStatus
)
from .repositories import (
    PluginRepository,
    PluginExecutionRepository,
    PluginInstallationRepository
)


class PluginRegistryService:
    """
    Domain service for plugin registration and lifecycle management.
    
    Handles plugin registration, validation, approval, and deprecation.
    """

    def __init__(self, plugin_repository: PluginRepository):
        self.plugin_repository = plugin_repository

    async def register_plugin(
        self,
        manifest: PluginManifest,
        tenant_id: Optional[UUID] = None
    ) -> Plugin:
        """
        Register a new plugin.
        
        Args:
            manifest: Plugin manifest
            tenant_id: Optional tenant ID for private plugins
            
        Returns:
            Registered plugin
            
        Raises:
            ValueError: If plugin with same name already exists
        """
        # Check if plugin with same name already exists
        existing = await self.plugin_repository.get_by_name(
            manifest.name,
            tenant_id
        )
        if existing:
            raise ValueError(
                f"Plugin '{manifest.name}' already registered for this tenant"
            )
        
        # Validate manifest
        self.validate_manifest(manifest)
        
        # Create plugin with pending status
        plugin = Plugin.create(
            manifest=manifest,
            tenant_id=tenant_id,
            status=PluginStatus.PENDING_REVIEW
        )
        
        await self.plugin_repository.save(plugin)
        return plugin

    def validate_manifest(self, manifest: PluginManifest) -> None:
        """
        Validate plugin manifest.
        
        Args:
            manifest: Plugin manifest to validate
            
        Raises:
            ValueError: If manifest is invalid
        """
        # Check required fields
        if not manifest.name or len(manifest.name) < 3:
            raise ValueError("Plugin name must be at least 3 characters")
        
        if not manifest.description or len(manifest.description) < 10:
            raise ValueError("Plugin description must be at least 10 characters")
        
        if not manifest.entry_point:
            raise ValueError("Plugin entry_point is required")
        
        if not manifest.author or len(manifest.author) < 3:
            raise ValueError("Plugin author must be at least 3 characters")
        
        # Validate entry point format based on runtime
        if manifest.runtime.value == "python":
            if not manifest.entry_point.endswith(".py"):
                raise ValueError("Python plugin entry_point must end with .py")
        elif manifest.runtime.value == "javascript":
            if not manifest.entry_point.endswith(".js"):
                raise ValueError("JavaScript plugin entry_point must end with .js")
        
        # Validate permissions
        if len(manifest.permissions) > 50:
            raise ValueError("Too many permissions requested (max 50)")
        
        # Validate sandbox config
        config = manifest.sandbox_config
        if config.max_memory_mb > 4096:
            raise ValueError("max_memory_mb cannot exceed 4096 MB")
        if config.timeout_seconds > 3600:
            raise ValueError("timeout_seconds cannot exceed 3600 seconds (1 hour)")

    async def approve_plugin(self, plugin_id: PluginId) -> Plugin:
        """
        Approve a plugin for use.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Approved plugin
            
        Raises:
            ValueError: If plugin not found or cannot be approved
        """
        plugin = await self.plugin_repository.get_by_id(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin {plugin_id} not found")
        
        plugin.approve()
        await self.plugin_repository.save(plugin)
        return plugin

    async def deprecate_plugin(
        self,
        plugin_id: PluginId,
        reason: str
    ) -> Plugin:
        """
        Deprecate a plugin.
        
        Args:
            plugin_id: Plugin identifier
            reason: Deprecation reason
            
        Returns:
            Deprecated plugin
            
        Raises:
            ValueError: If plugin not found
        """
        plugin = await self.plugin_repository.get_by_id(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin {plugin_id} not found")
        
        plugin.deprecate()
        plugin.metadata["deprecation_reason"] = reason
        plugin.metadata["deprecated_by"] = "system"
        
        await self.plugin_repository.save(plugin)
        return plugin

    async def activate_plugin(self, plugin_id: PluginId) -> Plugin:
        """
        Activate a plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Activated plugin
        """
        plugin = await self.plugin_repository.get_by_id(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin {plugin_id} not found")
        
        plugin.activate()
        await self.plugin_repository.save(plugin)
        return plugin

    async def disable_plugin(self, plugin_id: PluginId) -> Plugin:
        """
        Disable a plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Disabled plugin
        """
        plugin = await self.plugin_repository.get_by_id(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin {plugin_id} not found")
        
        plugin.disable()
        await self.plugin_repository.save(plugin)
        return plugin


class PluginTrustScoringService:
    """
    Domain service for calculating plugin trust scores.
    
    Calculates trust scores based on multiple factors.
    """

    def __init__(
        self,
        plugin_repository: PluginRepository,
        execution_repository: PluginExecutionRepository
    ):
        self.plugin_repository = plugin_repository
        self.execution_repository = execution_repository

    async def calculate_trust_score(
        self,
        plugin: Plugin
    ) -> PluginTrustScore:
        """
        Calculate comprehensive trust score for a plugin.
        
        Factors considered:
        - Success rate (30% weight)
        - Install count (20% weight)
        - Age/maturity (15% weight)
        - Author reputation (10% weight)
        - Security review (15% weight)
        - Community feedback (10% weight)
        
        Args:
            plugin: Plugin to score
            
        Returns:
            PluginTrustScore
        """
        factors = []
        
        # Factor 1: Success rate (30% weight)
        success_rate_score = await self._calculate_success_rate_score(plugin)
        factors.append(TrustFactor(
            name="success_rate",
            score=success_rate_score,
            weight=0.30,
            description=f"Plugin success rate: {plugin.success_rate:.2%}"
        ))
        
        # Factor 2: Install count (20% weight)
        install_score = self._calculate_install_score(plugin)
        factors.append(TrustFactor(
            name="install_count",
            score=install_score,
            weight=0.20,
            description=f"Install count: {plugin.install_count}"
        ))
        
        # Factor 3: Age/maturity (15% weight)
        maturity_score = self._calculate_maturity_score(plugin)
        factors.append(TrustFactor(
            name="maturity",
            score=maturity_score,
            weight=0.15,
            description="Plugin age and stability"
        ))
        
        # Factor 4: Author reputation (10% weight)
        author_score = self._calculate_author_score(plugin)
        factors.append(TrustFactor(
            name="author_reputation",
            score=author_score,
            weight=0.10,
            description="Author reputation and history"
        ))
        
        # Factor 5: Security review (15% weight)
        security_score = self._calculate_security_score(plugin)
        factors.append(TrustFactor(
            name="security_review",
            score=security_score,
            weight=0.15,
            description="Security review status"
        ))
        
        # Factor 6: Community feedback (10% weight)
        community_score = self._calculate_community_score(plugin)
        factors.append(TrustFactor(
            name="community_feedback",
            score=community_score,
            weight=0.10,
            description="Community ratings and feedback"
        ))
        
        # Calculate weighted score (0.0 - 1.0)
        weighted_sum = sum(f.weighted_score for f in factors)
        total_weight = sum(f.weight for f in factors)
        normalized_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Convert to 0-100 scale
        final_score = int(normalized_score * 100)
        
        return PluginTrustScore(
            plugin_id=plugin.id,
            score=final_score,
            factors=factors,
            calculation_metadata={
                "execution_count": plugin.execution_count,
                "install_count": plugin.install_count,
                "success_rate": plugin.success_rate,
                "age_days": (datetime.now(timezone.utc) - plugin.created_at).days
            }
        )

    async def _calculate_success_rate_score(self, plugin: Plugin) -> float:
        """Calculate score based on execution success rate"""
        if plugin.execution_count == 0:
            # New plugins start with neutral score
            return 0.5
        
        if plugin.execution_count < 10:
            # Not enough data, moderate score
            return 0.5 + (plugin.success_rate * 0.3)
        
        # Full weight to success rate with sufficient executions
        return plugin.success_rate

    def _calculate_install_score(self, plugin: Plugin) -> float:
        """Calculate score based on install count"""
        # Logarithmic scale: 1 install = 0.1, 10 = 0.4, 100 = 0.7, 1000+ = 1.0
        if plugin.install_count == 0:
            return 0.0
        elif plugin.install_count < 10:
            return 0.1 + (plugin.install_count / 10) * 0.3
        elif plugin.install_count < 100:
            return 0.4 + ((plugin.install_count - 10) / 90) * 0.3
        elif plugin.install_count < 1000:
            return 0.7 + ((plugin.install_count - 100) / 900) * 0.3
        else:
            return 1.0

    def _calculate_maturity_score(self, plugin: Plugin) -> float:
        """Calculate score based on plugin age and stability"""
        age_days = (datetime.now(timezone.utc) - plugin.created_at).days
        
        # Age score: 0 days = 0.3, 30 days = 0.6, 90+ days = 1.0
        if age_days < 30:
            age_score = 0.3 + (age_days / 30) * 0.3
        elif age_days < 90:
            age_score = 0.6 + ((age_days - 30) / 60) * 0.4
        else:
            age_score = 1.0
        
        # Stability bonus if published (approved)
        stability_bonus = 0.2 if plugin.published_at else 0.0
        
        return min(1.0, age_score + stability_bonus)

    def _calculate_author_score(self, plugin: Plugin) -> float:
        """Calculate score based on author reputation"""
        # In a real system, this would query author's other plugins
        # For now, use simple heuristics
        
        # Check if author metadata exists
        author_metadata = plugin.metadata.get("author_metadata", {})
        
        # Base score
        score = 0.5
        
        # Bonus for verified authors
        if author_metadata.get("verified", False):
            score += 0.3
        
        # Bonus for established authors
        if author_metadata.get("plugin_count", 0) > 5:
            score += 0.2
        
        return min(1.0, score)

    def _calculate_security_score(self, plugin: Plugin) -> float:
        """Calculate score based on security review status"""
        security_metadata = plugin.metadata.get("security", {})
        
        # Base score
        score = 0.4
        
        # Bonus for security review
        if security_metadata.get("reviewed", False):
            score += 0.3
        
        # Bonus for automated scans
        if security_metadata.get("scanned", False):
            score += 0.2
        
        # Penalty for known vulnerabilities
        vuln_count = security_metadata.get("vulnerability_count", 0)
        if vuln_count > 0:
            score -= min(0.5, vuln_count * 0.1)
        
        # Check permissions - fewer is better
        permission_count = len(plugin.manifest.permissions)
        if permission_count == 0:
            score += 0.1
        elif permission_count > 10:
            score -= 0.1
        
        return max(0.0, min(1.0, score))

    def _calculate_community_score(self, plugin: Plugin) -> float:
        """Calculate score based on community feedback"""
        community_metadata = plugin.metadata.get("community", {})
        
        # Average rating (1-5 stars)
        rating = community_metadata.get("average_rating", 3.0)
        rating_count = community_metadata.get("rating_count", 0)
        
        if rating_count == 0:
            # No ratings yet, neutral score
            return 0.5
        
        # Convert 1-5 rating to 0-1 score
        base_score = (rating - 1.0) / 4.0
        
        # Apply confidence based on rating count
        if rating_count < 5:
            confidence = 0.5
        elif rating_count < 20:
            confidence = 0.7
        else:
            confidence = 1.0
        
        return base_score * confidence + (1 - confidence) * 0.5

    async def update_plugin_trust_score(self, plugin_id: PluginId) -> Plugin:
        """
        Calculate and update trust score for a plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Updated plugin
        """
        plugin = await self.plugin_repository.get_by_id(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin {plugin_id} not found")
        
        trust_score = await self.calculate_trust_score(plugin)
        plugin.update_trust_score(trust_score)
        
        await self.plugin_repository.save(plugin)
        return plugin


class PluginExecutionService:
    """
    Domain service for plugin execution management.
    
    Handles plugin execution lifecycle with sandboxing and safety checks.
    """

    def __init__(
        self,
        plugin_repository: PluginRepository,
        execution_repository: PluginExecutionRepository,
        installation_repository: PluginInstallationRepository
    ):
        self.plugin_repository = plugin_repository
        self.execution_repository = execution_repository
        self.installation_repository = installation_repository

    async def prepare_execution(
        self,
        plugin_id: PluginId,
        tenant_id: UUID,
        input_data: Dict[str, any]
    ) -> PluginExecution:
        """
        Prepare plugin execution with validation and safety checks.
        
        Args:
            plugin_id: Plugin identifier
            tenant_id: Tenant identifier
            input_data: Input data for execution
            
        Returns:
            PluginExecution ready to run
            
        Raises:
            ValueError: If plugin not found or not executable
        """
        # Get plugin
        plugin = await self.plugin_repository.get_by_id(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin {plugin_id} not found")
        
        # Check if plugin can be executed
        if not plugin.can_be_executed():
            raise ValueError(
                f"Plugin {plugin_id} is not active (status: {plugin.status})"
            )
        
        # Check if plugin is installed for tenant
        installation = await self.installation_repository.get_by_plugin_and_tenant(
            plugin_id,
            tenant_id
        )
        if not installation:
            raise ValueError(
                f"Plugin {plugin_id} is not installed for tenant {tenant_id}"
            )
        
        if not installation.is_active:
            raise ValueError(
                f"Plugin {plugin_id} is disabled for tenant {tenant_id}"
            )
        
        # Check trust score if available
        if plugin.trust_score and not plugin.trust_score.is_trusted():
            raise ValueError(
                f"Plugin {plugin_id} has insufficient trust score: "
                f"{plugin.trust_score.score} (minimum: 50)"
            )
        
        # Validate input data
        self._validate_input_data(input_data, plugin)
        
        # Create execution record
        execution = PluginExecution.create(
            plugin_id=plugin_id,
            tenant_id=tenant_id,
            input_data=input_data
        )
        
        # Set execution context
        execution.execution_context = {
            "plugin_name": plugin.manifest.name,
            "plugin_version": str(plugin.manifest.version),
            "runtime": plugin.manifest.runtime.value,
            "sandbox_config": {
                "network_access": plugin.manifest.sandbox_config.network_access,
                "filesystem_access": plugin.manifest.sandbox_config.filesystem_access,
                "max_memory_mb": plugin.manifest.sandbox_config.max_memory_mb,
                "max_cpu_percent": plugin.manifest.sandbox_config.max_cpu_percent,
                "timeout_seconds": plugin.manifest.sandbox_config.timeout_seconds
            }
        }
        
        await self.execution_repository.save(execution)
        return execution

    def _validate_input_data(
        self,
        input_data: Dict[str, any],
        plugin: Plugin
    ) -> None:
        """
        Validate input data for plugin execution.
        
        Args:
            input_data: Input data to validate
            plugin: Plugin being executed
            
        Raises:
            ValueError: If input data is invalid
        """
        if not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary")
        
        # Check for required fields if specified in metadata
        required_fields = plugin.metadata.get("required_input_fields", [])
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Required input field '{field}' is missing")
        
        # Check input size limits
        import sys
        input_size = sys.getsizeof(str(input_data))
        max_input_size = 1024 * 1024  # 1 MB
        if input_size > max_input_size:
            raise ValueError(
                f"Input data too large: {input_size} bytes "
                f"(max: {max_input_size} bytes)"
            )

    async def complete_execution(
        self,
        execution_id: UUID,
        result: Dict[str, any],
        resource_usage: Optional[Dict[str, any]] = None
    ) -> PluginExecution:
        """
        Mark execution as completed.
        
        Args:
            execution_id: Execution identifier
            result: Execution result
            resource_usage: Optional resource usage metrics
            
        Returns:
            Completed execution
        """
        execution = await self.execution_repository.get_by_id(execution_id)
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")
        
        execution.complete(result)
        
        if resource_usage:
            execution.update_resource_usage(resource_usage)
        
        await self.execution_repository.save(execution)
        
        # Update plugin statistics
        await self._update_plugin_stats(execution)
        
        # Update installation usage
        await self._update_installation_usage(execution)
        
        return execution

    async def fail_execution(
        self,
        execution_id: UUID,
        error: str,
        resource_usage: Optional[Dict[str, any]] = None
    ) -> PluginExecution:
        """
        Mark execution as failed.
        
        Args:
            execution_id: Execution identifier
            error: Error message
            resource_usage: Optional resource usage metrics
            
        Returns:
            Failed execution
        """
        execution = await self.execution_repository.get_by_id(execution_id)
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")
        
        execution.fail(error)
        
        if resource_usage:
            execution.update_resource_usage(resource_usage)
        
        await self.execution_repository.save(execution)
        
        # Update plugin statistics
        await self._update_plugin_stats(execution)
        
        return execution

    async def _update_plugin_stats(self, execution: PluginExecution) -> None:
        """Update plugin execution statistics"""
        plugin = await self.plugin_repository.get_by_id(execution.plugin_id)
        if plugin:
            success = execution.is_successful
            duration_ms = execution.duration_ms or 0.0
            plugin.record_execution(success, duration_ms)
            await self.plugin_repository.save(plugin)

    async def _update_installation_usage(self, execution: PluginExecution) -> None:
        """Update installation usage statistics"""
        installation = await self.installation_repository.get_by_plugin_and_tenant(
            execution.plugin_id,
            execution.tenant_id
        )
        if installation:
            installation.record_usage()
            await self.installation_repository.save(installation)


class PluginMarketplaceService:
    """
    Domain service for plugin marketplace operations.
    
    Handles plugin discovery, search, and installation.
    """

    def __init__(
        self,
        plugin_repository: PluginRepository,
        installation_repository: PluginInstallationRepository
    ):
        self.plugin_repository = plugin_repository
        self.installation_repository = installation_repository

    async def list_marketplace_plugins(
        self,
        limit: int = 50,
        offset: int = 0,
        min_trust_score: int = 50
    ) -> List[Plugin]:
        """
        List available marketplace plugins.
        
        Args:
            limit: Maximum number of results
            offset: Offset for pagination
            min_trust_score: Minimum trust score filter
            
        Returns:
            List of marketplace plugins
        """
        # Get approved plugins
        plugins = await self.plugin_repository.list_approved(limit, offset)
        
        # Filter by trust score
        if min_trust_score > 0:
            plugins = [
                p for p in plugins
                if p.trust_score and p.trust_score.score >= min_trust_score
            ]
        
        # Sort by popularity (install count)
        plugins.sort(key=lambda p: p.install_count, reverse=True)
        
        return plugins

    async def search_plugins(
        self,
        query: str,
        runtime: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_trust_score: int = 50,
        limit: int = 50
    ) -> List[Plugin]:
        """
        Search marketplace plugins.
        
        Args:
            query: Search query
            runtime: Optional runtime filter
            tags: Optional tags filter
            min_trust_score: Minimum trust score
            limit: Maximum results
            
        Returns:
            List of matching plugins
        """
        from .entities import PluginRuntime
        
        runtime_enum = None
        if runtime:
            runtime_enum = PluginRuntime(runtime)
        
        plugins = await self.plugin_repository.search(
            query=query,
            runtime=runtime_enum,
            min_trust_score=min_trust_score,
            tags=tags,
            limit=limit
        )
        
        return plugins

    async def install_plugin(
        self,
        plugin_id: PluginId,
        tenant_id: UUID,
        configuration: Optional[Dict[str, any]] = None
    ) -> PluginInstallation:
        """
        Install plugin for a tenant.
        
        Args:
            plugin_id: Plugin identifier
            tenant_id: Tenant identifier
            configuration: Optional plugin configuration
            
        Returns:
            PluginInstallation
            
        Raises:
            ValueError: If plugin not found or already installed
        """
        # Check if plugin exists and is active
        plugin = await self.plugin_repository.get_by_id(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin {plugin_id} not found")
        
        if not plugin.can_be_executed():
            raise ValueError(
                f"Plugin {plugin_id} is not available for installation"
            )
        
        # Check if already installed
        existing = await self.installation_repository.get_by_plugin_and_tenant(
            plugin_id,
            tenant_id
        )
        if existing:
            raise ValueError(
                f"Plugin {plugin_id} is already installed for tenant {tenant_id}"
            )
        
        # Check trust score
        if plugin.trust_score and not plugin.trust_score.is_trusted():
            raise ValueError(
                f"Plugin {plugin_id} has insufficient trust score for installation: "
                f"{plugin.trust_score.score} (minimum: 50)"
            )
        
        # Create installation
        installation = PluginInstallation.create(
            plugin_id=plugin_id,
            tenant_id=tenant_id,
            configuration=configuration
        )
        
        await self.installation_repository.save(installation)
        
        # Update plugin install count
        plugin.record_installation()
        await self.plugin_repository.save(plugin)
        
        return installation

    async def uninstall_plugin(
        self,
        plugin_id: PluginId,
        tenant_id: UUID
    ) -> bool:
        """
        Uninstall plugin for a tenant.
        
        Args:
            plugin_id: Plugin identifier
            tenant_id: Tenant identifier
            
        Returns:
            True if uninstalled, False if not found
        """
        installation = await self.installation_repository.get_by_plugin_and_tenant(
            plugin_id,
            tenant_id
        )
        
        if not installation:
            return False
        
        return await self.installation_repository.delete(installation.id)

    async def get_installed_plugins(
        self,
        tenant_id: UUID,
        enabled_only: bool = True
    ) -> List[Plugin]:
        """
        Get all plugins installed for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            enabled_only: Only return enabled plugins
            
        Returns:
            List of installed plugins
        """
        installations = await self.installation_repository.list_by_tenant(
            tenant_id,
            enabled_only
        )
        
        plugins = []
        for installation in installations:
            plugin = await self.plugin_repository.get_by_id(installation.plugin_id)
            if plugin:
                plugins.append(plugin)
        
        return plugins

    async def get_popular_plugins(
        self,
        limit: int = 20,
        min_trust_score: int = 50
    ) -> List[Plugin]:
        """
        Get most popular plugins.
        
        Args:
            limit: Maximum number of results
            min_trust_score: Minimum trust score
            
        Returns:
            List of popular plugins
        """
        return await self.plugin_repository.get_popular(limit, min_trust_score)
