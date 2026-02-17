"""
Plugin API Routes

Provides REST endpoints for plugin marketplace and management.
"""

import sys
import os

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.src.auth.dependencies import CurrentUser, get_current_user, require_role
from services.api.src.dependencies.injection import get_db_session
from infrastructure.middleware.tenant_context import get_current_tenant
from core.domain.plugin.entities import Plugin, PluginId, PluginStatus, PluginRuntime, PluginExecution, ExecutionStatus


router = APIRouter(prefix="/api/v3/plugins", tags=["Plugins"])


# ==================== Request/Response Schemas ====================

class PluginConfigSchema(BaseModel):
    """Plugin configuration schema"""
    env_vars: dict = Field(default_factory=dict, description="Environment variables")
    secrets: dict = Field(default_factory=dict, description="Secret configuration")
    settings: dict = Field(default_factory=dict, description="Plugin-specific settings")


class RegisterPluginRequest(BaseModel):
    """Request to register a new plugin"""
    name: str = Field(..., min_length=1, max_length=255, description="Plugin name")
    description: str = Field(..., description="Plugin description")
    version: str = Field(..., pattern=r'^\d+\.\d+\.\d+', description="Semantic version")
    author: str = Field(..., description="Plugin author")
    runtime: str = Field(..., description="Runtime environment (python, javascript, docker)")
    entrypoint: str = Field(..., description="Entry point for execution")
    source_url: Optional[str] = Field(default=None, description="Source code URL")
    documentation_url: Optional[str] = Field(default=None, description="Documentation URL")
    tags: List[str] = Field(default_factory=list, description="Plugin tags")
    is_public: bool = Field(default=False, description="Available in marketplace")


class InstallPluginRequest(BaseModel):
    """Request to install a plugin"""
    config: Optional[PluginConfigSchema] = Field(default=None, description="Plugin configuration")
    enabled: bool = Field(default=True, description="Enable plugin after installation")


class ExecutePluginRequest(BaseModel):
    """Request to execute a plugin"""
    inputs: dict = Field(..., description="Plugin input parameters")
    timeout_seconds: Optional[int] = Field(default=300, description="Execution timeout")


class PluginResponse(BaseModel):
    """Plugin response"""
    id: str = Field(..., description="Plugin ID")
    name: str = Field(..., description="Plugin name")
    description: str = Field(..., description="Plugin description")
    version: str = Field(..., description="Plugin version")
    author: str = Field(..., description="Plugin author")
    status: str = Field(..., description="Plugin status")
    runtime: str = Field(..., description="Runtime environment")
    trust_score: int = Field(..., description="Trust score (0-100)")
    install_count: int = Field(..., description="Number of installations")
    tags: List[str] = Field(..., description="Plugin tags")
    source_url: Optional[str] = Field(default=None, description="Source code URL")
    documentation_url: Optional[str] = Field(default=None, description="Documentation URL")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    is_installed: bool = Field(default=False, description="Installed in current tenant")


class PluginListResponse(BaseModel):
    """Plugin list response"""
    plugins: List[PluginResponse] = Field(..., description="List of plugins")
    total: int = Field(..., description="Total plugin count")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Page size")


class InstalledPluginResponse(BaseModel):
    """Installed plugin response"""
    installation_id: str = Field(..., description="Installation ID")
    plugin: PluginResponse = Field(..., description="Plugin details")
    config: PluginConfigSchema = Field(..., description="Installation configuration")
    enabled: bool = Field(..., description="Is plugin enabled")
    installed_at: datetime = Field(..., description="Installation timestamp")
    last_used_at: Optional[datetime] = Field(default=None, description="Last execution timestamp")


class PluginExecutionResponse(BaseModel):
    """Plugin execution response"""
    execution_id: str = Field(..., description="Execution ID")
    plugin_id: str = Field(..., description="Plugin ID")
    plugin_name: str = Field(..., description="Plugin name")
    status: str = Field(..., description="Execution status")
    inputs: dict = Field(..., description="Execution inputs")
    outputs: Optional[dict] = Field(default=None, description="Execution outputs")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    started_at: datetime = Field(..., description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    duration_ms: Optional[int] = Field(default=None, description="Execution duration in milliseconds")


class SuccessResponse(BaseModel):
    """Generic success response"""
    success: bool = Field(default=True, description="Success flag")
    message: str = Field(..., description="Success message")


# ==================== Database Dependencies ====================

async def get_plugin_repository(session: AsyncSession = Depends(get_db_session)):
    """Get plugin repository"""
    from infrastructure.persistence.plugin_repository import PostgreSQLPluginRepository
    return PostgreSQLPluginRepository(session)


async def get_plugin_installation_repository(session: AsyncSession = Depends(get_db_session)):
    """Get plugin installation repository"""
    from infrastructure.persistence.plugin_repository import PostgreSQLPluginInstallationRepository
    return PostgreSQLPluginInstallationRepository(session)


async def get_plugin_execution_repository(session: AsyncSession = Depends(get_db_session)):
    """Get plugin execution repository"""
    from infrastructure.persistence.plugin_repository import PostgreSQLPluginExecutionRepository
    return PostgreSQLPluginExecutionRepository(session)


# ==================== Plugin Marketplace Endpoints ====================

@router.get(
    "",
    response_model=PluginListResponse,
    summary="List marketplace plugins",
    description="List all available plugins in the marketplace with filtering and pagination",
)
async def list_plugins(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search query"),
    runtime: Optional[str] = Query(None, description="Filter by runtime"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    min_trust_score: Optional[int] = Query(None, ge=0, le=100, description="Minimum trust score"),
    current_user: CurrentUser = Depends(get_current_user),
    plugin_repo = Depends(get_plugin_repository),
    installation_repo = Depends(get_plugin_installation_repository),
) -> PluginListResponse:
    """List marketplace plugins"""
    
    tenant = get_current_tenant()
    
    # Calculate offset for pagination
    offset = (page - 1) * page_size
    
    # Get plugins based on filters
    if search:
        plugins = await plugin_repo.search(
            query=search,
            runtime=PluginRuntime(runtime) if runtime else None,
            min_trust_score=min_trust_score,
            tags=tags,
            limit=page_size,
            offset=offset,
        )
    else:
        plugins = await plugin_repo.list_approved(
            limit=page_size,
            offset=offset,
        )
    
    # Check which plugins are installed for current tenant
    installed_plugin_ids = set()
    if tenant:
        installations = await installation_repo.list_by_tenant(tenant.id)
        installed_plugin_ids = {str(inst.plugin_id.value) for inst in installations}
    
    # Convert to response schemas
    plugin_responses = [
        PluginResponse(
            id=str(plugin.id.value),
            name=plugin.name,
            description=plugin.description,
            version=str(plugin.version),
            author=plugin.author,
            status=plugin.status.value,
            runtime=plugin.runtime.value,
            trust_score=plugin.trust_score,
            install_count=plugin.install_count,
            tags=plugin.tags,
            source_url=plugin.source_url,
            documentation_url=plugin.documentation_url,
            created_at=plugin.created_at,
            updated_at=plugin.updated_at,
            is_installed=str(plugin.id.value) in installed_plugin_ids,
        )
        for plugin in plugins
    ]
    
    # Get total count (simplified - in real implementation, use a count query)
    total = len(plugins)
    
    return PluginListResponse(
        plugins=plugin_responses,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{plugin_id}",
    response_model=PluginResponse,
    summary="Get plugin details",
    description="Retrieve detailed information about a specific plugin",
)
async def get_plugin(
    plugin_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    plugin_repo = Depends(get_plugin_repository),
    installation_repo = Depends(get_plugin_installation_repository),
) -> PluginResponse:
    """Get plugin details"""
    
    plugin = await plugin_repo.get_by_id(PluginId(UUID(plugin_id)))
    
    if not plugin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin {plugin_id} not found"
        )
    
    # Check if installed for current tenant
    is_installed = False
    tenant = get_current_tenant()
    if tenant:
        installation = await installation_repo.get_by_plugin_and_tenant(
            PluginId(UUID(plugin_id)),
            tenant.id
        )
        is_installed = installation is not None
    
    return PluginResponse(
        id=str(plugin.id.value),
        name=plugin.name,
        description=plugin.description,
        version=str(plugin.version),
        author=plugin.author,
        status=plugin.status.value,
        runtime=plugin.runtime.value,
        trust_score=plugin.trust_score,
        install_count=plugin.install_count,
        tags=plugin.tags,
        source_url=plugin.source_url,
        documentation_url=plugin.documentation_url,
        created_at=plugin.created_at,
        updated_at=plugin.updated_at,
        is_installed=is_installed,
    )


@router.post(
    "",
    response_model=PluginResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new plugin (admin)",
    description="Register a new plugin in the marketplace (admin only)",
)
async def register_plugin(
    request: RegisterPluginRequest,
    current_user: CurrentUser = Depends(require_role("admin")),
    plugin_repo = Depends(get_plugin_repository),
    session: AsyncSession = Depends(get_db_session),
) -> PluginResponse:
    """Register a new plugin"""
    
    # Check if plugin with same name already exists
    existing = await plugin_repo.get_by_name(request.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Plugin with name '{request.name}' already exists"
        )
    
    # Create plugin entity
    from core.domain.plugin.entities import VersionInfo
    
    plugin = Plugin(
        id=PluginId.generate(),
        name=request.name,
        description=request.description,
        version=VersionInfo.parse(request.version),
        author=request.author,
        status=PluginStatus.PENDING_REVIEW if request.is_public else PluginStatus.ACTIVE,
        runtime=PluginRuntime(request.runtime),
        entrypoint=request.entrypoint,
        schema={},  # Would be validated in real implementation
        trust_score=50,  # Initial trust score
        install_count=0,
        tags=request.tags,
        source_url=request.source_url,
        documentation_url=request.documentation_url,
        tenant_id=None if request.is_public else get_current_tenant().id if get_current_tenant() else None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    
    # Save to database
    await plugin_repo.save(plugin)
    await session.commit()
    
    return PluginResponse(
        id=str(plugin.id.value),
        name=plugin.name,
        description=plugin.description,
        version=str(plugin.version),
        author=plugin.author,
        status=plugin.status.value,
        runtime=plugin.runtime.value,
        trust_score=plugin.trust_score,
        install_count=plugin.install_count,
        tags=plugin.tags,
        source_url=plugin.source_url,
        documentation_url=plugin.documentation_url,
        created_at=plugin.created_at,
        updated_at=plugin.updated_at,
        is_installed=False,
    )


@router.post(
    "/{plugin_id}/install",
    response_model=InstalledPluginResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Install plugin",
    description="Install a plugin for the current tenant",
)
async def install_plugin(
    plugin_id: str,
    request: InstallPluginRequest,
    current_user: CurrentUser = Depends(get_current_user),
    plugin_repo = Depends(get_plugin_repository),
    installation_repo = Depends(get_plugin_installation_repository),
    session: AsyncSession = Depends(get_db_session),
) -> InstalledPluginResponse:
    """Install a plugin"""
    
    tenant = get_current_tenant()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context is required"
        )
    
    # Check if plugins are enabled for this tenant
    if not tenant.settings.enable_plugins:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Plugin marketplace is not enabled for this tenant"
        )
    
    # Get plugin
    plugin = await plugin_repo.get_by_id(PluginId(UUID(plugin_id)))
    if not plugin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin {plugin_id} not found"
        )
    
    # Check if plugin is active
    if plugin.status != PluginStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Plugin is not available (status: {plugin.status.value})"
        )
    
    # Check if already installed
    existing = await installation_repo.get_by_plugin_and_tenant(
        PluginId(UUID(plugin_id)),
        tenant.id
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Plugin is already installed"
        )
    
    # Create installation
    from core.domain.plugin.entities import PluginInstallation
    
    config = {}
    if request.config:
        config = {
            "env_vars": request.config.env_vars,
            "secrets": request.config.secrets,
            "settings": request.config.settings,
        }
    
    installation = PluginInstallation(
        id=uuid4(),
        plugin_id=plugin.id,
        tenant_id=tenant.id,
        config=config,
        enabled=request.enabled,
        installed_at=datetime.utcnow(),
        installed_by=UUID(current_user.user_id),
        last_used_at=None,
    )
    
    # Save installation
    await installation_repo.save(installation)
    
    # Update plugin install count
    plugin.install_count += 1
    await plugin_repo.save(plugin)
    
    await session.commit()
    
    return InstalledPluginResponse(
        installation_id=str(installation.id),
        plugin=PluginResponse(
            id=str(plugin.id.value),
            name=plugin.name,
            description=plugin.description,
            version=str(plugin.version),
            author=plugin.author,
            status=plugin.status.value,
            runtime=plugin.runtime.value,
            trust_score=plugin.trust_score,
            install_count=plugin.install_count,
            tags=plugin.tags,
            source_url=plugin.source_url,
            documentation_url=plugin.documentation_url,
            created_at=plugin.created_at,
            updated_at=plugin.updated_at,
            is_installed=True,
        ),
        config=PluginConfigSchema(
            env_vars=config.get("env_vars", {}),
            secrets=config.get("secrets", {}),
            settings=config.get("settings", {}),
        ),
        enabled=installation.enabled,
        installed_at=installation.installed_at,
        last_used_at=installation.last_used_at,
    )


@router.delete(
    "/{plugin_id}/install",
    response_model=SuccessResponse,
    summary="Uninstall plugin",
    description="Uninstall a plugin from the current tenant",
)
async def uninstall_plugin(
    plugin_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    installation_repo = Depends(get_plugin_installation_repository),
    session: AsyncSession = Depends(get_db_session),
) -> SuccessResponse:
    """Uninstall a plugin"""
    
    tenant = get_current_tenant()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context is required"
        )
    
    # Get installation
    installation = await installation_repo.get_by_plugin_and_tenant(
        PluginId(UUID(plugin_id)),
        tenant.id
    )
    
    if not installation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Plugin is not installed"
        )
    
    # Delete installation
    await installation_repo.delete(installation.id)
    await session.commit()
    
    return SuccessResponse(
        success=True,
        message="Plugin uninstalled successfully"
    )


@router.post(
    "/{plugin_id}/execute",
    response_model=PluginExecutionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Execute plugin",
    description="Execute a plugin with provided inputs",
)
async def execute_plugin(
    plugin_id: str,
    request: ExecutePluginRequest,
    current_user: CurrentUser = Depends(get_current_user),
    plugin_repo = Depends(get_plugin_repository),
    installation_repo = Depends(get_plugin_installation_repository),
    execution_repo = Depends(get_plugin_execution_repository),
    session: AsyncSession = Depends(get_db_session),
) -> PluginExecutionResponse:
    """Execute a plugin"""
    
    tenant = get_current_tenant()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context is required"
        )
    
    # Check if plugin is installed
    installation = await installation_repo.get_by_plugin_and_tenant(
        PluginId(UUID(plugin_id)),
        tenant.id
    )
    
    if not installation:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Plugin must be installed before execution"
        )
    
    if not installation.enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Plugin is disabled"
        )
    
    # Get plugin
    plugin = await plugin_repo.get_by_id(PluginId(UUID(plugin_id)))
    if not plugin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin {plugin_id} not found"
        )
    
    # Create execution record
    
    execution = PluginExecution(
        id=uuid4(),
        plugin_id=plugin.id,
        tenant_id=tenant.id,
        status=ExecutionStatus.PENDING,
        inputs=request.inputs,
        outputs=None,
        error=None,
        started_at=datetime.utcnow(),
        completed_at=None,
        executed_by=UUID(current_user.user_id),
    )
    
    # Save execution
    await execution_repo.save(execution)
    
    # Update last used timestamp
    installation.last_used_at = datetime.utcnow()
    await installation_repo.save(installation)
    
    await session.commit()
    
    # In real implementation, this would trigger async execution
    # For now, return pending status
    
    return PluginExecutionResponse(
        execution_id=str(execution.id),
        plugin_id=str(plugin.id.value),
        plugin_name=plugin.name,
        status=execution.status.value,
        inputs=execution.inputs,
        outputs=execution.outputs,
        error=execution.error,
        started_at=execution.started_at,
        completed_at=execution.completed_at,
        duration_ms=None,
    )


@router.get(
    "/installed",
    response_model=List[InstalledPluginResponse],
    summary="List installed plugins",
    description="List all plugins installed for the current tenant",
)
async def list_installed_plugins(
    current_user: CurrentUser = Depends(get_current_user),
    plugin_repo = Depends(get_plugin_repository),
    installation_repo = Depends(get_plugin_installation_repository),
) -> List[InstalledPluginResponse]:
    """List installed plugins"""
    
    tenant = get_current_tenant()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context is required"
        )
    
    # Get all installations for tenant
    installations = await installation_repo.list_by_tenant(tenant.id)
    
    # Build response with plugin details
    responses = []
    for installation in installations:
        plugin = await plugin_repo.get_by_id(installation.plugin_id)
        if plugin:
            responses.append(
                InstalledPluginResponse(
                    installation_id=str(installation.id),
                    plugin=PluginResponse(
                        id=str(plugin.id.value),
                        name=plugin.name,
                        description=plugin.description,
                        version=str(plugin.version),
                        author=plugin.author,
                        status=plugin.status.value,
                        runtime=plugin.runtime.value,
                        trust_score=plugin.trust_score,
                        install_count=plugin.install_count,
                        tags=plugin.tags,
                        source_url=plugin.source_url,
                        documentation_url=plugin.documentation_url,
                        created_at=plugin.created_at,
                        updated_at=plugin.updated_at,
                        is_installed=True,
                    ),
                    config=PluginConfigSchema(
                        env_vars=installation.config.get("env_vars", {}),
                        secrets=installation.config.get("secrets", {}),
                        settings=installation.config.get("settings", {}),
                    ),
                    enabled=installation.enabled,
                    installed_at=installation.installed_at,
                    last_used_at=installation.last_used_at,
                )
            )
    
    return responses
