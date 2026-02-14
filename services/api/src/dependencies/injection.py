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
