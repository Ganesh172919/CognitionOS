"""
FastAPI Dependencies

Provides dependency injection for use cases, repositories, and infrastructure.
"""

import sys
import os
from typing import AsyncGenerator

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy.orm import sessionmaker

from core.config import get_config
from core.application.workflow.use_cases import (
    CreateWorkflowUseCase,
    ExecuteWorkflowUseCase,
    GetWorkflowExecutionStatusUseCase,
    ProcessWorkflowStepUseCase,
)
from core.application.agent.use_cases import (
    RegisterAgentDefinitionUseCase,
    CreateAgentUseCase,
    AssignTaskToAgentUseCase,
    CompleteAgentTaskUseCase,
)
from infrastructure.persistence.workflow_repository import (
    SQLAlchemyWorkflowRepository,
    SQLAlchemyWorkflowExecutionRepository,
)
from infrastructure.persistence.checkpoint_repository import PostgreSQLCheckpointRepository
from infrastructure.persistence.health_repository import (
    PostgreSQLAgentHealthRepository,
    PostgreSQLHealthIncidentRepository,
)
from infrastructure.persistence.cost_repository import (
    PostgreSQLWorkflowBudgetRepository,
    PostgreSQLCostTrackingRepository,
)
from core.domain.checkpoint.services import CheckpointService
from core.domain.health_monitoring.services import AgentHealthMonitoringService
from core.domain.cost_governance.services import CostGovernanceService
from infrastructure.events.event_bus import InMemoryEventBus


# Configuration
config = get_config()

# Database engine (singleton)
_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    """Get or create database engine"""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            config.database.url,
            pool_size=config.database.pool_size,
            max_overflow=config.database.max_overflow,
            pool_timeout=config.database.pool_timeout,
            echo=config.api.debug,
        )
    return _engine


# Session factory
async_session_factory = sessionmaker(
    bind=get_engine(),
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency"""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Event bus (singleton)
_event_bus: InMemoryEventBus | None = None


def get_event_bus() -> InMemoryEventBus:
    """Get event bus dependency"""
    global _event_bus
    if _event_bus is None:
        _event_bus = InMemoryEventBus()
    return _event_bus


# ==================== Repository Dependencies ====================

async def get_workflow_repository(
    session: AsyncSession = None
) -> SQLAlchemyWorkflowRepository:
    """Get workflow repository dependency"""
    if session is None:
        async with async_session_factory() as session:
            return SQLAlchemyWorkflowRepository(session)
    return SQLAlchemyWorkflowRepository(session)


async def get_workflow_execution_repository(
    session: AsyncSession = None
) -> SQLAlchemyWorkflowExecutionRepository:
    """Get workflow execution repository dependency"""
    if session is None:
        async with async_session_factory() as session:
            return SQLAlchemyWorkflowExecutionRepository(session)
    return SQLAlchemyWorkflowExecutionRepository(session)


# ==================== Use Case Dependencies ====================

def get_create_workflow_use_case(
    workflow_repository: SQLAlchemyWorkflowRepository | None = None,
    event_bus: InMemoryEventBus | None = None,
) -> CreateWorkflowUseCase:
    """Get create workflow use case dependency"""
    if workflow_repository is None:
        workflow_repository = SQLAlchemyWorkflowRepository(None)
    if event_bus is None:
        event_bus = get_event_bus()
    return CreateWorkflowUseCase(workflow_repository, event_bus)


def get_execute_workflow_use_case(
    workflow_repository: SQLAlchemyWorkflowRepository | None = None,
    execution_repository: SQLAlchemyWorkflowExecutionRepository | None = None,
    event_bus: InMemoryEventBus | None = None,
) -> ExecuteWorkflowUseCase:
    """Get execute workflow use case dependency"""
    if workflow_repository is None:
        workflow_repository = SQLAlchemyWorkflowRepository(None)
    if execution_repository is None:
        execution_repository = SQLAlchemyWorkflowExecutionRepository(None)
    if event_bus is None:
        event_bus = get_event_bus()
    return ExecuteWorkflowUseCase(
        workflow_repository,
        execution_repository,
        event_bus
    )


def get_workflow_status_use_case(
    execution_repository: SQLAlchemyWorkflowExecutionRepository | None = None,
) -> GetWorkflowExecutionStatusUseCase:
    """Get workflow status use case dependency"""
    if execution_repository is None:
        execution_repository = SQLAlchemyWorkflowExecutionRepository(None)
    return GetWorkflowExecutionStatusUseCase(execution_repository)


# ==================== Health Check Dependencies ====================

async def check_database_health() -> bool:
    """Check database connection health"""
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception:
        return False


async def check_redis_health() -> bool:
    """Check Redis connection health"""
    # TODO: Implement Redis health check
    return True


async def check_rabbitmq_health() -> bool:
    """Check RabbitMQ connection health"""
    # TODO: Implement RabbitMQ health check
    return True


# ==================== Phase 3: Checkpoint Dependencies ====================

async def get_checkpoint_repository(
    session: AsyncSession = None
) -> PostgreSQLCheckpointRepository:
    """Get checkpoint repository dependency"""
    if session is None:
        async with async_session_factory() as session:
            return PostgreSQLCheckpointRepository(session)
    return PostgreSQLCheckpointRepository(session)


async def get_checkpoint_service(
    session: AsyncSession = None
) -> CheckpointService:
    """Get checkpoint service dependency"""
    checkpoint_repo = await get_checkpoint_repository(session)
    return CheckpointService(checkpoint_repository=checkpoint_repo)


# ==================== Phase 3: Health Monitoring Dependencies ====================

async def get_health_repository(
    session: AsyncSession = None
) -> PostgreSQLAgentHealthRepository:
    """Get agent health repository dependency"""
    if session is None:
        async with async_session_factory() as session:
            return PostgreSQLAgentHealthRepository(session)
    return PostgreSQLAgentHealthRepository(session)


async def get_health_incident_repository(
    session: AsyncSession = None
) -> PostgreSQLHealthIncidentRepository:
    """Get health incident repository dependency"""
    if session is None:
        async with async_session_factory() as session:
            return PostgreSQLHealthIncidentRepository(session)
    return PostgreSQLHealthIncidentRepository(session)


async def get_health_monitoring_service(
    session: AsyncSession = None
) -> AgentHealthMonitoringService:
    """Get health monitoring service dependency"""
    health_repo = await get_health_repository(session)
    incident_repo = await get_health_incident_repository(session)
    return AgentHealthMonitoringService(
        health_repository=health_repo,
        incident_repository=incident_repo,
    )


# ==================== Phase 3: Cost Governance Dependencies ====================

async def get_budget_repository(
    session: AsyncSession = None
) -> PostgreSQLWorkflowBudgetRepository:
    """Get workflow budget repository dependency"""
    if session is None:
        async with async_session_factory() as session:
            return PostgreSQLWorkflowBudgetRepository(session)
    return PostgreSQLWorkflowBudgetRepository(session)


async def get_cost_tracking_repository(
    session: AsyncSession = None
) -> PostgreSQLCostTrackingRepository:
    """Get cost tracking repository dependency"""
    if session is None:
        async with async_session_factory() as session:
            return PostgreSQLCostTrackingRepository(session)
    return PostgreSQLCostTrackingRepository(session)


async def get_cost_governance_service(
    session: AsyncSession = None
) -> CostGovernanceService:
    """Get cost governance service dependency"""
    budget_repo = await get_budget_repository(session)
    cost_tracking_repo = await get_cost_tracking_repository(session)
    return CostGovernanceService(
        budget_repository=budget_repo,
        cost_tracking_repository=cost_tracking_repo,
    )
