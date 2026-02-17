"""
Plugin Repository Implementations

PostgreSQL implementations of plugin repositories.
"""

import logging
from datetime import datetime
from typing import Optional, List
from uuid import UUID

from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from core.domain.plugin.entities import (
    Plugin,
    PluginExecution,
    PluginId,
    PluginInstallation,
    PluginManifest,
    PluginRuntime,
    PluginStatus,
    ExecutionStatus,
    TrustLevel,
    VersionInfo,
    Permission,
    SandboxConfig,
    PluginTrustScore,
    TrustFactor,
)
from core.domain.plugin.repositories import (
    PluginRepository,
    PluginExecutionRepository,
    PluginInstallationRepository,
)
from infrastructure.persistence.plugin_models import (
    PluginModel,
    PluginExecutionModel,
    PluginInstallationModel,
)


logger = logging.getLogger(__name__)


class PostgreSQLPluginRepository(PluginRepository):
    """PostgreSQL implementation of PluginRepository"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, plugin: Plugin) -> None:
        """Persist plugin"""
        try:
            stmt = select(PluginModel).where(PluginModel.id == plugin.id.value)
            result = await self.session.execute(stmt)
            existing_model = result.scalar_one_or_none()
            
            if existing_model:
                # Update existing
                self._update_model_from_entity(existing_model, plugin)
                await self.session.flush()
                logger.info(f"Updated plugin: {plugin.id}")
            else:
                # Create new
                plugin_model = self._to_model(plugin)
                self.session.add(plugin_model)
                await self.session.flush()
                logger.info(f"Created plugin: {plugin.id}")
        except Exception as e:
            logger.error(f"Error saving plugin {plugin.id}: {e}")
            raise
    
    async def get_by_id(self, plugin_id: PluginId) -> Optional[Plugin]:
        """Retrieve plugin by ID"""
        try:
            stmt = select(PluginModel).where(PluginModel.id == plugin_id.value)
            result = await self.session.execute(stmt)
            plugin_model = result.scalar_one_or_none()
            
            if plugin_model is None:
                return None
            
            return self._to_entity(plugin_model)
        except Exception as e:
            logger.error(f"Error fetching plugin {plugin_id}: {e}")
            raise
    
    async def get_by_name(
        self,
        name: str,
        tenant_id: Optional[UUID] = None
    ) -> Optional[Plugin]:
        """Retrieve plugin by name"""
        try:
            conditions = [PluginModel.name == name]
            if tenant_id is not None:
                conditions.append(PluginModel.tenant_id == tenant_id)
            else:
                conditions.append(PluginModel.tenant_id.is_(None))
            
            stmt = select(PluginModel).where(and_(*conditions))
            result = await self.session.execute(stmt)
            plugin_model = result.scalar_one_or_none()
            
            if plugin_model is None:
                return None
            
            return self._to_entity(plugin_model)
        except Exception as e:
            logger.error(f"Error fetching plugin by name {name}: {e}")
            raise
    
    async def delete(self, plugin_id: PluginId) -> bool:
        """Delete plugin"""
        try:
            stmt = select(PluginModel).where(PluginModel.id == plugin_id.value)
            result = await self.session.execute(stmt)
            plugin_model = result.scalar_one_or_none()
            
            if plugin_model is None:
                return False
            
            await self.session.delete(plugin_model)
            await self.session.flush()
            
            logger.info(f"Deleted plugin: {plugin_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting plugin {plugin_id}: {e}")
            raise
    
    async def list_by_status(
        self,
        status: PluginStatus,
        limit: int = 100,
        offset: int = 0
    ) -> List[Plugin]:
        """List plugins by status"""
        try:
            stmt = (
                select(PluginModel)
                .where(PluginModel.status == status)
                .offset(offset)
                .limit(limit)
                .order_by(PluginModel.created_at.desc())
            )
            result = await self.session.execute(stmt)
            plugin_models = result.scalars().all()
            
            return [self._to_entity(model) for model in plugin_models]
        except Exception as e:
            logger.error(f"Error listing plugins by status {status}: {e}")
            raise
    
    async def list_by_tenant(
        self,
        tenant_id: Optional[UUID],
        limit: int = 100,
        offset: int = 0
    ) -> List[Plugin]:
        """List plugins by tenant"""
        try:
            if tenant_id is None:
                condition = PluginModel.tenant_id.is_(None)
            else:
                condition = PluginModel.tenant_id == tenant_id
            
            stmt = (
                select(PluginModel)
                .where(condition)
                .offset(offset)
                .limit(limit)
                .order_by(PluginModel.created_at.desc())
            )
            result = await self.session.execute(stmt)
            plugin_models = result.scalars().all()
            
            return [self._to_entity(model) for model in plugin_models]
        except Exception as e:
            logger.error(f"Error listing plugins by tenant {tenant_id}: {e}")
            raise
    
    async def list_approved(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Plugin]:
        """List approved (active) plugins available in marketplace"""
        try:
            stmt = (
                select(PluginModel)
                .where(
                    and_(
                        PluginModel.status == PluginStatus.ACTIVE,
                        PluginModel.tenant_id.is_(None)
                    )
                )
                .offset(offset)
                .limit(limit)
                .order_by(PluginModel.install_count.desc())
            )
            result = await self.session.execute(stmt)
            plugin_models = result.scalars().all()
            
            return [self._to_entity(model) for model in plugin_models]
        except Exception as e:
            logger.error(f"Error listing approved plugins: {e}")
            raise
    
    async def search(
        self,
        query: str,
        runtime: Optional[PluginRuntime] = None,
        min_trust_score: Optional[int] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Plugin]:
        """Search plugins by various criteria"""
        try:
            conditions = [PluginModel.status == PluginStatus.ACTIVE]
            
            # Text search in name, description, author
            if query:
                search_pattern = f"%{query}%"
                conditions.append(
                    or_(
                        PluginModel.name.ilike(search_pattern),
                        PluginModel.description.ilike(search_pattern),
                        PluginModel.author.ilike(search_pattern)
                    )
                )
            
            if runtime:
                conditions.append(PluginModel.runtime == runtime)
            
            if min_trust_score is not None:
                conditions.append(PluginModel.trust_score >= min_trust_score)
            
            if tags:
                # PostgreSQL array overlap operator
                for tag in tags:
                    conditions.append(PluginModel.tags.contains([tag]))
            
            stmt = (
                select(PluginModel)
                .where(and_(*conditions))
                .offset(offset)
                .limit(limit)
                .order_by(PluginModel.install_count.desc())
            )
            result = await self.session.execute(stmt)
            plugin_models = result.scalars().all()
            
            return [self._to_entity(model) for model in plugin_models]
        except Exception as e:
            logger.error(f"Error searching plugins: {e}")
            raise
    
    async def get_by_runtime(
        self,
        runtime: PluginRuntime,
        limit: int = 100,
        offset: int = 0
    ) -> List[Plugin]:
        """Get plugins by runtime type"""
        try:
            stmt = (
                select(PluginModel)
                .where(PluginModel.runtime == runtime)
                .offset(offset)
                .limit(limit)
                .order_by(PluginModel.created_at.desc())
            )
            result = await self.session.execute(stmt)
            plugin_models = result.scalars().all()
            
            return [self._to_entity(model) for model in plugin_models]
        except Exception as e:
            logger.error(f"Error getting plugins by runtime {runtime}: {e}")
            raise
    
    async def get_popular(
        self,
        limit: int = 20,
        min_trust_score: int = 50
    ) -> List[Plugin]:
        """Get most popular plugins (by install count)"""
        try:
            stmt = (
                select(PluginModel)
                .where(
                    and_(
                        PluginModel.status == PluginStatus.ACTIVE,
                        PluginModel.trust_score >= min_trust_score
                    )
                )
                .order_by(PluginModel.install_count.desc())
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            plugin_models = result.scalars().all()
            
            return [self._to_entity(model) for model in plugin_models]
        except Exception as e:
            logger.error(f"Error getting popular plugins: {e}")
            raise
    
    async def get_by_trust_level(
        self,
        trust_level: TrustLevel,
        limit: int = 100,
        offset: int = 0
    ) -> List[Plugin]:
        """Get plugins by trust level"""
        try:
            stmt = (
                select(PluginModel)
                .where(PluginModel.trust_level == trust_level.value)
                .offset(offset)
                .limit(limit)
                .order_by(PluginModel.created_at.desc())
            )
            result = await self.session.execute(stmt)
            plugin_models = result.scalars().all()
            
            return [self._to_entity(model) for model in plugin_models]
        except Exception as e:
            logger.error(f"Error getting plugins by trust level {trust_level}: {e}")
            raise
    
    async def count_by_status(self, status: PluginStatus) -> int:
        """Count plugins by status"""
        try:
            stmt = select(func.count()).select_from(PluginModel).where(PluginModel.status == status)
            result = await self.session.execute(stmt)
            count = result.scalar()
            
            return count or 0
        except Exception as e:
            logger.error(f"Error counting plugins by status {status}: {e}")
            raise
    
    def _to_entity(self, model: PluginModel) -> Plugin:
        """Convert model to entity"""
        # Build version info
        version = VersionInfo(
            major=model.version_major,
            minor=model.version_minor,
            patch=model.version_patch,
            prerelease=model.version_prerelease
        )
        
        # Build permissions
        permissions = [
            Permission(
                resource=p["resource"],
                action=p["action"],
                description=p["description"]
            )
            for p in (model.permissions or [])
        ]
        
        # Build sandbox config
        sandbox_data = model.sandbox_config or {}
        sandbox_config = SandboxConfig(
            network_access=sandbox_data.get("network_access", False),
            filesystem_access=sandbox_data.get("filesystem_access", False),
            max_memory_mb=sandbox_data.get("max_memory_mb", 512),
            max_cpu_percent=sandbox_data.get("max_cpu_percent", 50),
            timeout_seconds=sandbox_data.get("timeout_seconds", 300),
            allowed_paths=sandbox_data.get("allowed_paths", []),
            environment_variables=sandbox_data.get("environment_variables", {})
        )
        
        # Build manifest
        manifest = PluginManifest(
            name=model.name,
            version=version,
            author=model.author,
            description=model.description,
            entry_point=model.entry_point,
            runtime=model.runtime,
            permissions=permissions,
            dependencies=model.dependencies or {},
            sandbox_config=sandbox_config,
            homepage_url=model.homepage_url,
            documentation_url=model.documentation_url,
            source_url=model.source_url,
            tags=model.tags or []
        )
        
        # Build trust score if present
        trust_score = None
        if model.trust_score is not None:
            trust_factors = []
            for factor_data in (model.trust_factors or []):
                trust_factors.append(
                    TrustFactor(
                        name=factor_data["name"],
                        score=factor_data["score"],
                        weight=factor_data["weight"],
                        description=factor_data["description"]
                    )
                )
            
            trust_score = PluginTrustScore(
                plugin_id=PluginId(model.id),
                score=model.trust_score,
                factors=trust_factors,
                last_calculated=model.trust_last_calculated or datetime.utcnow(),
                calculation_metadata={}
            )
        
        # Build plugin entity
        return Plugin(
            id=PluginId(model.id),
            manifest=manifest,
            status=model.status,
            trust_score=trust_score,
            tenant_id=model.tenant_id,
            created_at=model.created_at,
            updated_at=model.updated_at,
            published_at=model.published_at,
            deprecated_at=model.deprecated_at,
            install_count=model.install_count,
            execution_count=model.execution_count,
            success_count=model.success_count,
            failure_count=model.failure_count,
            average_execution_time_ms=model.average_execution_time_ms,
            metadata=model.metadata or {}
        )
    
    def _to_model(self, entity: Plugin) -> PluginModel:
        """Convert entity to model"""
        # Convert permissions to JSON
        permissions = [
            {
                "resource": p.resource,
                "action": p.action,
                "description": p.description
            }
            for p in entity.manifest.permissions
        ]
        
        # Convert sandbox config to JSON
        sandbox_config = {
            "network_access": entity.manifest.sandbox_config.network_access,
            "filesystem_access": entity.manifest.sandbox_config.filesystem_access,
            "max_memory_mb": entity.manifest.sandbox_config.max_memory_mb,
            "max_cpu_percent": entity.manifest.sandbox_config.max_cpu_percent,
            "timeout_seconds": entity.manifest.sandbox_config.timeout_seconds,
            "allowed_paths": entity.manifest.sandbox_config.allowed_paths,
            "environment_variables": entity.manifest.sandbox_config.environment_variables
        }
        
        # Convert trust score to JSON
        trust_score = None
        trust_level = None
        trust_factors = None
        trust_last_calculated = None
        
        if entity.trust_score:
            trust_score = entity.trust_score.score
            trust_level = entity.trust_score.trust_level.value
            trust_factors = [
                {
                    "name": f.name,
                    "score": f.score,
                    "weight": f.weight,
                    "description": f.description
                }
                for f in entity.trust_score.factors
            ]
            trust_last_calculated = entity.trust_score.last_calculated
        
        return PluginModel(
            id=entity.id.value,
            name=entity.manifest.name,
            version_major=entity.manifest.version.major,
            version_minor=entity.manifest.version.minor,
            version_patch=entity.manifest.version.patch,
            version_prerelease=entity.manifest.version.prerelease,
            author=entity.manifest.author,
            description=entity.manifest.description,
            entry_point=entity.manifest.entry_point,
            runtime=entity.manifest.runtime,
            permissions=permissions,
            dependencies=entity.manifest.dependencies,
            sandbox_config=sandbox_config,
            homepage_url=entity.manifest.homepage_url,
            documentation_url=entity.manifest.documentation_url,
            source_url=entity.manifest.source_url,
            tags=entity.manifest.tags,
            status=entity.status,
            tenant_id=entity.tenant_id,
            trust_score=trust_score,
            trust_level=trust_level,
            trust_factors=trust_factors,
            trust_last_calculated=trust_last_calculated,
            install_count=entity.install_count,
            execution_count=entity.execution_count,
            success_count=entity.success_count,
            failure_count=entity.failure_count,
            average_execution_time_ms=entity.average_execution_time_ms,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            published_at=entity.published_at,
            deprecated_at=entity.deprecated_at,
            metadata=entity.metadata
        )
    
    def _update_model_from_entity(self, model: PluginModel, entity: Plugin) -> None:
        """Update model fields from entity"""
        # Convert permissions to JSON
        permissions = [
            {
                "resource": p.resource,
                "action": p.action,
                "description": p.description
            }
            for p in entity.manifest.permissions
        ]
        
        # Convert sandbox config to JSON
        sandbox_config = {
            "network_access": entity.manifest.sandbox_config.network_access,
            "filesystem_access": entity.manifest.sandbox_config.filesystem_access,
            "max_memory_mb": entity.manifest.sandbox_config.max_memory_mb,
            "max_cpu_percent": entity.manifest.sandbox_config.max_cpu_percent,
            "timeout_seconds": entity.manifest.sandbox_config.timeout_seconds,
            "allowed_paths": entity.manifest.sandbox_config.allowed_paths,
            "environment_variables": entity.manifest.sandbox_config.environment_variables
        }
        
        model.name = entity.manifest.name
        model.version_major = entity.manifest.version.major
        model.version_minor = entity.manifest.version.minor
        model.version_patch = entity.manifest.version.patch
        model.version_prerelease = entity.manifest.version.prerelease
        model.author = entity.manifest.author
        model.description = entity.manifest.description
        model.entry_point = entity.manifest.entry_point
        model.runtime = entity.manifest.runtime
        model.permissions = permissions
        model.dependencies = entity.manifest.dependencies
        model.sandbox_config = sandbox_config
        model.homepage_url = entity.manifest.homepage_url
        model.documentation_url = entity.manifest.documentation_url
        model.source_url = entity.manifest.source_url
        model.tags = entity.manifest.tags
        model.status = entity.status
        model.tenant_id = entity.tenant_id
        
        if entity.trust_score:
            model.trust_score = entity.trust_score.score
            model.trust_level = entity.trust_score.trust_level.value
            model.trust_factors = [
                {
                    "name": f.name,
                    "score": f.score,
                    "weight": f.weight,
                    "description": f.description
                }
                for f in entity.trust_score.factors
            ]
            model.trust_last_calculated = entity.trust_score.last_calculated
        
        model.install_count = entity.install_count
        model.execution_count = entity.execution_count
        model.success_count = entity.success_count
        model.failure_count = entity.failure_count
        model.average_execution_time_ms = entity.average_execution_time_ms
        model.updated_at = entity.updated_at
        model.published_at = entity.published_at
        model.deprecated_at = entity.deprecated_at
        model.metadata = entity.metadata


class PostgreSQLPluginExecutionRepository(PluginExecutionRepository):
    """PostgreSQL implementation of PluginExecutionRepository"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, execution: PluginExecution) -> None:
        """Persist plugin execution"""
        try:
            stmt = select(PluginExecutionModel).where(PluginExecutionModel.id == execution.id)
            result = await self.session.execute(stmt)
            existing_model = result.scalar_one_or_none()
            
            if existing_model:
                # Update existing
                self._update_model_from_entity(existing_model, execution)
                await self.session.flush()
            else:
                # Create new
                execution_model = self._to_model(execution)
                self.session.add(execution_model)
                await self.session.flush()
        except Exception as e:
            logger.error(f"Error saving plugin execution {execution.id}: {e}")
            raise
    
    async def get_by_id(self, execution_id: UUID) -> Optional[PluginExecution]:
        """Retrieve execution by ID"""
        try:
            stmt = select(PluginExecutionModel).where(PluginExecutionModel.id == execution_id)
            result = await self.session.execute(stmt)
            execution_model = result.scalar_one_or_none()
            
            if execution_model is None:
                return None
            
            return self._to_entity(execution_model)
        except Exception as e:
            logger.error(f"Error fetching plugin execution {execution_id}: {e}")
            raise
    
    async def list_by_plugin(
        self,
        plugin_id: PluginId,
        limit: int = 100,
        offset: int = 0
    ) -> List[PluginExecution]:
        """List executions for a plugin"""
        try:
            stmt = (
                select(PluginExecutionModel)
                .where(PluginExecutionModel.plugin_id == plugin_id.value)
                .order_by(PluginExecutionModel.start_time.desc())
                .offset(offset)
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            execution_models = result.scalars().all()
            
            return [self._to_entity(model) for model in execution_models]
        except Exception as e:
            logger.error(f"Error listing executions for plugin {plugin_id}: {e}")
            raise
    
    async def list_by_tenant(
        self,
        tenant_id: UUID,
        limit: int = 100,
        offset: int = 0
    ) -> List[PluginExecution]:
        """List executions for a tenant"""
        try:
            stmt = (
                select(PluginExecutionModel)
                .where(PluginExecutionModel.tenant_id == tenant_id)
                .order_by(PluginExecutionModel.start_time.desc())
                .offset(offset)
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            execution_models = result.scalars().all()
            
            return [self._to_entity(model) for model in execution_models]
        except Exception as e:
            logger.error(f"Error listing executions for tenant {tenant_id}: {e}")
            raise
    
    async def list_by_status(
        self,
        status: ExecutionStatus,
        limit: int = 100,
        offset: int = 0
    ) -> List[PluginExecution]:
        """List executions by status"""
        try:
            stmt = (
                select(PluginExecutionModel)
                .where(PluginExecutionModel.status == status)
                .order_by(PluginExecutionModel.start_time.desc())
                .offset(offset)
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            execution_models = result.scalars().all()
            
            return [self._to_entity(model) for model in execution_models]
        except Exception as e:
            logger.error(f"Error listing executions by status {status}: {e}")
            raise
    
    async def list_by_plugin_and_tenant(
        self,
        plugin_id: PluginId,
        tenant_id: UUID,
        limit: int = 100,
        offset: int = 0
    ) -> List[PluginExecution]:
        """List executions for a specific plugin and tenant"""
        try:
            stmt = (
                select(PluginExecutionModel)
                .where(
                    and_(
                        PluginExecutionModel.plugin_id == plugin_id.value,
                        PluginExecutionModel.tenant_id == tenant_id
                    )
                )
                .order_by(PluginExecutionModel.start_time.desc())
                .offset(offset)
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            execution_models = result.scalars().all()
            
            return [self._to_entity(model) for model in execution_models]
        except Exception as e:
            logger.error(f"Error listing executions for plugin {plugin_id} and tenant {tenant_id}: {e}")
            raise
    
    async def get_recent_by_tenant(
        self,
        tenant_id: UUID,
        limit: int = 50
    ) -> List[PluginExecution]:
        """Get recent executions for a tenant"""
        try:
            stmt = (
                select(PluginExecutionModel)
                .where(PluginExecutionModel.tenant_id == tenant_id)
                .order_by(PluginExecutionModel.start_time.desc())
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            execution_models = result.scalars().all()
            
            return [self._to_entity(model) for model in execution_models]
        except Exception as e:
            logger.error(f"Error getting recent executions for tenant {tenant_id}: {e}")
            raise
    
    async def get_running_executions(
        self,
        tenant_id: Optional[UUID] = None
    ) -> List[PluginExecution]:
        """Get currently running executions"""
        try:
            conditions = [PluginExecutionModel.status == ExecutionStatus.RUNNING]
            if tenant_id:
                conditions.append(PluginExecutionModel.tenant_id == tenant_id)
            
            stmt = select(PluginExecutionModel).where(and_(*conditions))
            result = await self.session.execute(stmt)
            execution_models = result.scalars().all()
            
            return [self._to_entity(model) for model in execution_models]
        except Exception as e:
            logger.error(f"Error getting running executions: {e}")
            raise
    
    async def count_by_plugin(
        self,
        plugin_id: PluginId,
        status: Optional[ExecutionStatus] = None
    ) -> int:
        """Count executions for a plugin"""
        try:
            conditions = [PluginExecutionModel.plugin_id == plugin_id.value]
            if status:
                conditions.append(PluginExecutionModel.status == status)
            
            stmt = select(func.count()).select_from(PluginExecutionModel).where(and_(*conditions))
            result = await self.session.execute(stmt)
            count = result.scalar()
            
            return count or 0
        except Exception as e:
            logger.error(f"Error counting executions for plugin {plugin_id}: {e}")
            raise
    
    async def get_execution_stats(
        self,
        plugin_id: PluginId,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> dict:
        """Get execution statistics for a plugin"""
        try:
            conditions = [PluginExecutionModel.plugin_id == plugin_id.value]
            if start_date:
                conditions.append(PluginExecutionModel.start_time >= start_date)
            if end_date:
                conditions.append(PluginExecutionModel.start_time <= end_date)
            
            # Get count by status
            stmt = (
                select(
                    PluginExecutionModel.status,
                    func.count().label('count')
                )
                .where(and_(*conditions))
                .group_by(PluginExecutionModel.status)
            )
            result = await self.session.execute(stmt)
            status_counts = {row.status: row.count for row in result}
            
            # Get average duration for completed executions
            stmt = (
                select(
                    func.avg(
                        func.extract('epoch', PluginExecutionModel.end_time - PluginExecutionModel.start_time) * 1000
                    )
                )
                .where(
                    and_(
                        *conditions,
                        PluginExecutionModel.status == ExecutionStatus.COMPLETED,
                        PluginExecutionModel.end_time.isnot(None)
                    )
                )
            )
            result = await self.session.execute(stmt)
            avg_duration = result.scalar()
            
            return {
                "total": sum(status_counts.values()),
                "completed": status_counts.get(ExecutionStatus.COMPLETED, 0),
                "failed": status_counts.get(ExecutionStatus.FAILED, 0),
                "running": status_counts.get(ExecutionStatus.RUNNING, 0),
                "pending": status_counts.get(ExecutionStatus.PENDING, 0),
                "timeout": status_counts.get(ExecutionStatus.TIMEOUT, 0),
                "cancelled": status_counts.get(ExecutionStatus.CANCELLED, 0),
                "avg_duration_ms": float(avg_duration) if avg_duration else 0.0
            }
        except Exception as e:
            logger.error(f"Error getting execution stats for plugin {plugin_id}: {e}")
            raise
    
    async def delete_old_executions(
        self,
        before_date: datetime,
        limit: int = 1000
    ) -> int:
        """Delete old execution records"""
        try:
            stmt = (
                select(PluginExecutionModel)
                .where(PluginExecutionModel.start_time < before_date)
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            execution_models = result.scalars().all()
            
            count = len(execution_models)
            for model in execution_models:
                await self.session.delete(model)
            
            await self.session.flush()
            
            logger.info(f"Deleted {count} old plugin executions before {before_date}")
            return count
        except Exception as e:
            logger.error(f"Error deleting old plugin executions: {e}")
            raise
    
    def _to_entity(self, model: PluginExecutionModel) -> PluginExecution:
        """Convert model to entity"""
        return PluginExecution(
            id=model.id,
            plugin_id=PluginId(model.plugin_id),
            tenant_id=model.tenant_id,
            status=model.status,
            input_data=model.input_data or {},
            start_time=model.start_time,
            end_time=model.end_time,
            result=model.result,
            error=model.error,
            execution_context=model.execution_context or {},
            resource_usage=model.resource_usage or {},
            logs=list(model.logs or [])
        )
    
    def _to_model(self, entity: PluginExecution) -> PluginExecutionModel:
        """Convert entity to model"""
        return PluginExecutionModel(
            id=entity.id,
            plugin_id=entity.plugin_id.value,
            tenant_id=entity.tenant_id,
            status=entity.status,
            input_data=entity.input_data,
            result=entity.result,
            error=entity.error,
            start_time=entity.start_time,
            end_time=entity.end_time,
            execution_context=entity.execution_context,
            resource_usage=entity.resource_usage,
            logs=entity.logs
        )
    
    def _update_model_from_entity(self, model: PluginExecutionModel, entity: PluginExecution) -> None:
        """Update model fields from entity"""
        model.status = entity.status
        model.input_data = entity.input_data
        model.result = entity.result
        model.error = entity.error
        model.end_time = entity.end_time
        model.execution_context = entity.execution_context
        model.resource_usage = entity.resource_usage
        model.logs = entity.logs


class PostgreSQLPluginInstallationRepository(PluginInstallationRepository):
    """PostgreSQL implementation of PluginInstallationRepository"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, installation: PluginInstallation) -> None:
        """Persist plugin installation"""
        try:
            stmt = select(PluginInstallationModel).where(PluginInstallationModel.id == installation.id)
            result = await self.session.execute(stmt)
            existing_model = result.scalar_one_or_none()
            
            if existing_model:
                # Update existing
                self._update_model_from_entity(existing_model, installation)
                await self.session.flush()
            else:
                # Create new
                installation_model = self._to_model(installation)
                self.session.add(installation_model)
                await self.session.flush()
        except Exception as e:
            logger.error(f"Error saving plugin installation {installation.id}: {e}")
            raise
    
    async def get_by_id(self, installation_id: UUID) -> Optional[PluginInstallation]:
        """Retrieve installation by ID"""
        try:
            stmt = select(PluginInstallationModel).where(PluginInstallationModel.id == installation_id)
            result = await self.session.execute(stmt)
            installation_model = result.scalar_one_or_none()
            
            if installation_model is None:
                return None
            
            return self._to_entity(installation_model)
        except Exception as e:
            logger.error(f"Error fetching plugin installation {installation_id}: {e}")
            raise
    
    async def get_by_plugin_and_tenant(
        self,
        plugin_id: PluginId,
        tenant_id: UUID
    ) -> Optional[PluginInstallation]:
        """Get installation for a specific plugin and tenant"""
        try:
            stmt = (
                select(PluginInstallationModel)
                .where(
                    and_(
                        PluginInstallationModel.plugin_id == plugin_id.value,
                        PluginInstallationModel.tenant_id == tenant_id
                    )
                )
            )
            result = await self.session.execute(stmt)
            installation_model = result.scalar_one_or_none()
            
            if installation_model is None:
                return None
            
            return self._to_entity(installation_model)
        except Exception as e:
            logger.error(f"Error fetching installation for plugin {plugin_id} and tenant {tenant_id}: {e}")
            raise
    
    async def list_by_tenant(
        self,
        tenant_id: UUID,
        enabled_only: bool = False
    ) -> List[PluginInstallation]:
        """List all plugin installations for a tenant"""
        try:
            conditions = [PluginInstallationModel.tenant_id == tenant_id]
            if enabled_only:
                conditions.append(PluginInstallationModel.enabled == True)
            
            stmt = (
                select(PluginInstallationModel)
                .where(and_(*conditions))
                .order_by(PluginInstallationModel.installed_at.desc())
            )
            result = await self.session.execute(stmt)
            installation_models = result.scalars().all()
            
            return [self._to_entity(model) for model in installation_models]
        except Exception as e:
            logger.error(f"Error listing installations for tenant {tenant_id}: {e}")
            raise
    
    async def list_by_plugin(
        self,
        plugin_id: PluginId,
        limit: int = 100,
        offset: int = 0
    ) -> List[PluginInstallation]:
        """List all installations of a specific plugin"""
        try:
            stmt = (
                select(PluginInstallationModel)
                .where(PluginInstallationModel.plugin_id == plugin_id.value)
                .order_by(PluginInstallationModel.installed_at.desc())
                .offset(offset)
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            installation_models = result.scalars().all()
            
            return [self._to_entity(model) for model in installation_models]
        except Exception as e:
            logger.error(f"Error listing installations for plugin {plugin_id}: {e}")
            raise
    
    async def delete(self, installation_id: UUID) -> bool:
        """Delete plugin installation"""
        try:
            stmt = select(PluginInstallationModel).where(PluginInstallationModel.id == installation_id)
            result = await self.session.execute(stmt)
            installation_model = result.scalar_one_or_none()
            
            if installation_model is None:
                return False
            
            await self.session.delete(installation_model)
            await self.session.flush()
            
            logger.info(f"Deleted plugin installation: {installation_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting plugin installation {installation_id}: {e}")
            raise
    
    async def count_by_plugin(self, plugin_id: PluginId) -> int:
        """Count installations for a plugin"""
        try:
            stmt = (
                select(func.count())
                .select_from(PluginInstallationModel)
                .where(PluginInstallationModel.plugin_id == plugin_id.value)
            )
            result = await self.session.execute(stmt)
            count = result.scalar()
            
            return count or 0
        except Exception as e:
            logger.error(f"Error counting installations for plugin {plugin_id}: {e}")
            raise
    
    async def is_installed(
        self,
        plugin_id: PluginId,
        tenant_id: UUID
    ) -> bool:
        """Check if plugin is installed for a tenant"""
        try:
            stmt = (
                select(PluginInstallationModel.id)
                .where(
                    and_(
                        PluginInstallationModel.plugin_id == plugin_id.value,
                        PluginInstallationModel.tenant_id == tenant_id
                    )
                )
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none() is not None
        except Exception as e:
            logger.error(f"Error checking installation for plugin {plugin_id} and tenant {tenant_id}: {e}")
            raise
    
    def _to_entity(self, model: PluginInstallationModel) -> PluginInstallation:
        """Convert model to entity"""
        return PluginInstallation(
            id=model.id,
            plugin_id=PluginId(model.plugin_id),
            tenant_id=model.tenant_id,
            installed_at=model.installed_at,
            configuration=model.configuration or {},
            enabled=model.enabled,
            last_used_at=model.last_used_at,
            usage_count=model.usage_count
        )
    
    def _to_model(self, entity: PluginInstallation) -> PluginInstallationModel:
        """Convert entity to model"""
        return PluginInstallationModel(
            id=entity.id,
            plugin_id=entity.plugin_id.value,
            tenant_id=entity.tenant_id,
            installed_at=entity.installed_at,
            configuration=entity.configuration,
            enabled=entity.enabled,
            last_used_at=entity.last_used_at,
            usage_count=entity.usage_count
        )
    
    def _update_model_from_entity(self, model: PluginInstallationModel, entity: PluginInstallation) -> None:
        """Update model fields from entity"""
        model.configuration = entity.configuration
        model.enabled = entity.enabled
        model.last_used_at = entity.last_used_at
        model.usage_count = entity.usage_count
