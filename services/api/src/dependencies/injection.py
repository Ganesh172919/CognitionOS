"""
FastAPI Dependencies

Provides dependency injection for use cases, repositories, and infrastructure.
"""

import os
from typing import AsyncGenerator
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
    PostgreSQLWorkflowRepository,
    PostgreSQLWorkflowExecutionRepository,
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
from infrastructure.persistence.tenant_repository import PostgreSQLTenantRepository
from infrastructure.persistence.billing_repository import (
    PostgreSQLSubscriptionRepository,
    PostgreSQLInvoiceRepository,
    PostgreSQLUsageRecordRepository,
)
from infrastructure.persistence.plugin_repository import (
    PostgreSQLPluginRepository,
    PostgreSQLPluginExecutionRepository,
    PostgreSQLPluginInstallationRepository,
)
from infrastructure.persistence.api_key_repository import PostgresAPIKeyRepository
from core.domain.checkpoint.services import CheckpointService
from core.domain.health_monitoring.services import AgentHealthMonitoringService
from core.domain.cost_governance.services import CostGovernanceService
from core.domain.billing.services import (
    BillingService,
    UsageMeteringService,
    EntitlementService,
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


async def close_db() -> None:
    """Close database engine and all connections"""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


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
) -> PostgreSQLWorkflowRepository:
    """Get workflow repository dependency"""
    if session is None:
        async with async_session_factory() as session:
            return PostgreSQLWorkflowRepository(session)
    return PostgreSQLWorkflowRepository(session)


async def get_workflow_execution_repository(
    session: AsyncSession = None
) -> PostgreSQLWorkflowExecutionRepository:
    """Get workflow execution repository dependency"""
    if session is None:
        async with async_session_factory() as session:
            return PostgreSQLWorkflowExecutionRepository(session)
    return PostgreSQLWorkflowExecutionRepository(session)


# ==================== Use Case Dependencies ====================

def get_create_workflow_use_case(
    workflow_repository: PostgreSQLWorkflowRepository | None = None,
    event_bus: InMemoryEventBus | None = None,
) -> CreateWorkflowUseCase:
    """Get create workflow use case dependency"""
    if workflow_repository is None:
        workflow_repository = PostgreSQLWorkflowRepository(None)
    if event_bus is None:
        event_bus = get_event_bus()
    return CreateWorkflowUseCase(workflow_repository, event_bus)


def get_execute_workflow_use_case(
    workflow_repository: PostgreSQLWorkflowRepository | None = None,
    execution_repository: PostgreSQLWorkflowExecutionRepository | None = None,
    event_bus: InMemoryEventBus | None = None,
) -> ExecuteWorkflowUseCase:
    """Get execute workflow use case dependency"""
    if workflow_repository is None:
        workflow_repository = PostgreSQLWorkflowRepository(None)
    if execution_repository is None:
        execution_repository = PostgreSQLWorkflowExecutionRepository(None)
    if event_bus is None:
        event_bus = get_event_bus()
    return ExecuteWorkflowUseCase(
        workflow_repository,
        execution_repository,
        event_bus
    )


def get_workflow_status_use_case(
    execution_repository: PostgreSQLWorkflowExecutionRepository | None = None,
) -> GetWorkflowExecutionStatusUseCase:
    """Get workflow status use case dependency"""
    if execution_repository is None:
        execution_repository = PostgreSQLWorkflowExecutionRepository(None)
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
    try:
        from infrastructure.persistence.redis_pool import check_redis_health as redis_health
        return await redis_health()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Redis health check failed: {e}")
        return False


async def check_rabbitmq_health() -> bool:
    """Check RabbitMQ connection health"""
    try:
        import aio_pika
        import logging
        logger = logging.getLogger(__name__)
        
        config = get_config()
        # Build RabbitMQ URL
        rabbitmq_url = (
            f"amqp://{config.rabbitmq.username}:{config.rabbitmq.password}"
            f"@{config.rabbitmq.host}:{config.rabbitmq.port}{config.rabbitmq.virtual_host}"
        )
        # Try to connect
        connection = await aio_pika.connect_robust(
            rabbitmq_url,
            timeout=5,
        )
        await connection.close()
        return True
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"RabbitMQ health check failed: {e}")
        return False


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


# ==================== Phase 3: Memory Hierarchy Dependencies ====================

async def get_working_memory_repository(
    session: AsyncSession = None
):
    """Get working memory repository dependency"""
    from infrastructure.persistence.memory_hierarchy_repository import PostgreSQLWorkingMemoryRepository
    if session is None:
        async with async_session_factory() as session:
            return PostgreSQLWorkingMemoryRepository(session)
    return PostgreSQLWorkingMemoryRepository(session)


async def get_episodic_memory_repository(
    session: AsyncSession = None
):
    """Get episodic memory repository dependency"""
    from infrastructure.persistence.memory_hierarchy_repository import PostgreSQLEpisodicMemoryRepository
    if session is None:
        async with async_session_factory() as session:
            return PostgreSQLEpisodicMemoryRepository(session)
    return PostgreSQLEpisodicMemoryRepository(session)


async def get_longterm_memory_repository(
    session: AsyncSession = None
):
    """Get long-term memory repository dependency"""
    from infrastructure.persistence.memory_hierarchy_repository import PostgreSQLLongTermMemoryRepository
    if session is None:
        async with async_session_factory() as session:
            return PostgreSQLLongTermMemoryRepository(session)
    return PostgreSQLLongTermMemoryRepository(session)


async def get_store_memory_use_case(
    session: AsyncSession = None
):
    """Get store working memory use case dependency"""
    from core.application.memory_hierarchy.use_cases import StoreWorkingMemoryUseCase
    l1_repo = await get_working_memory_repository(session)
    event_bus = get_event_bus()
    return StoreWorkingMemoryUseCase(
        l1_repository=l1_repo,
        event_publisher=event_bus,
    )


async def get_retrieve_memory_use_case(
    session: AsyncSession = None
):
    """Get retrieve working memory use case dependency"""
    from core.application.memory_hierarchy.use_cases import RetrieveWorkingMemoryUseCase
    l1_repo = await get_working_memory_repository(session)
    event_bus = get_event_bus()
    return RetrieveWorkingMemoryUseCase(
        l1_repository=l1_repo,
        event_publisher=event_bus,
    )


async def get_promote_l2_use_case(
    session: AsyncSession = None
):
    """Get promote L1 to L2 use case dependency"""
    from core.application.memory_hierarchy.use_cases import PromoteMemoriesToL2UseCase
    from core.domain.memory_hierarchy.services import MemoryTierManager, MemoryCompressionService
    
    l1_repo = await get_working_memory_repository(session)
    l2_repo = await get_episodic_memory_repository(session)
    l3_repo = await get_longterm_memory_repository(session)
    event_bus = get_event_bus()
    
    tier_manager = MemoryTierManager(
        l1_repository=l1_repo,
        l2_repository=l2_repo,
        l3_repository=l3_repo,
    )
    compression_service = MemoryCompressionService()
    
    return PromoteMemoriesToL2UseCase(
        tier_manager=tier_manager,
        compression_service=compression_service,
        l1_repository=l1_repo,
        event_publisher=event_bus,
    )


async def get_promote_l3_use_case(
    session: AsyncSession = None
):
    """Get promote L2 to L3 use case dependency"""
    from core.application.memory_hierarchy.use_cases import PromoteMemoriesToL3UseCase
    from core.domain.memory_hierarchy.services import MemoryTierManager
    
    l1_repo = await get_working_memory_repository(session)
    l2_repo = await get_episodic_memory_repository(session)
    l3_repo = await get_longterm_memory_repository(session)
    event_bus = get_event_bus()
    
    tier_manager = MemoryTierManager(
        l1_repository=l1_repo,
        l2_repository=l2_repo,
        l3_repository=l3_repo,
    )
    
    return PromoteMemoriesToL3UseCase(
        tier_manager=tier_manager,
        l2_repository=l2_repo,
        event_publisher=event_bus,
    )


async def get_evict_memories_use_case(
    session: AsyncSession = None
):
    """Get evict memories use case dependency"""
    from core.application.memory_hierarchy.use_cases import EvictLowPriorityMemoriesUseCase
    from core.domain.memory_hierarchy.services import MemoryTierManager
    
    l1_repo = await get_working_memory_repository(session)
    l2_repo = await get_episodic_memory_repository(session)
    l3_repo = await get_longterm_memory_repository(session)
    event_bus = get_event_bus()
    
    tier_manager = MemoryTierManager(
        l1_repository=l1_repo,
        l2_repository=l2_repo,
        l3_repository=l3_repo,
    )
    
    return EvictLowPriorityMemoriesUseCase(
        tier_manager=tier_manager,
        event_publisher=event_bus,
    )


async def get_memory_statistics_use_case(
    session: AsyncSession = None
):
    """Get memory statistics use case dependency"""
    from core.application.memory_hierarchy.use_cases import GetMemoryStatisticsUseCase
    
    l1_repo = await get_working_memory_repository(session)
    l2_repo = await get_episodic_memory_repository(session)
    l3_repo = await get_longterm_memory_repository(session)
    
    return GetMemoryStatisticsUseCase(
        l1_repository=l1_repo,
        l2_repository=l2_repo,
        l3_repository=l3_repo,
    )


async def get_search_memories_use_case(
    session: AsyncSession = None
):
    """Get search memories use case dependency"""
    from core.application.memory_hierarchy.use_cases import SearchMemoriesAcrossTiersUseCase
    from core.domain.memory_hierarchy.services import MemoryCompressionService
    
    l1_repo = await get_working_memory_repository(session)
    l2_repo = await get_episodic_memory_repository(session)
    l3_repo = await get_longterm_memory_repository(session)
    compression_service = MemoryCompressionService()
    
    return SearchMemoriesAcrossTiersUseCase(
        l1_repository=l1_repo,
        l2_repository=l2_repo,
        l3_repository=l3_repo,
        compression_service=compression_service,
    )


async def get_update_importance_use_case(
    session: AsyncSession = None
):
    """Get update importance use case dependency"""
    from core.application.memory_hierarchy.use_cases import UpdateMemoryImportanceUseCase
    from core.domain.memory_hierarchy.services import MemoryImportanceScorer
    
    l1_repo = await get_working_memory_repository(session)
    importance_scorer = MemoryImportanceScorer()
    
    return UpdateMemoryImportanceUseCase(
        importance_scorer=importance_scorer,
        l1_repository=l1_repo,
    )


# ==================== Multi-Tenancy Dependencies ====================

async def get_tenant_repository(
    session: AsyncSession = None
) -> PostgreSQLTenantRepository:
    """Get tenant repository dependency"""
    if session is None:
        async with async_session_factory() as session:
            return PostgreSQLTenantRepository(session)
    return PostgreSQLTenantRepository(session)


async def get_api_key_repository(
    session: AsyncSession = None
) -> PostgresAPIKeyRepository:
    """Get API key repository dependency"""
    if session is None:
        async with async_session_factory() as session:
            return PostgresAPIKeyRepository(session)
    return PostgresAPIKeyRepository(session)


# ==================== Billing Dependencies ====================

async def get_subscription_repository(
    session: AsyncSession = None
) -> PostgreSQLSubscriptionRepository:
    """Get subscription repository dependency"""
    if session is None:
        async with async_session_factory() as session:
            return PostgreSQLSubscriptionRepository(session)
    return PostgreSQLSubscriptionRepository(session)


async def get_invoice_repository(
    session: AsyncSession = None
) -> PostgreSQLInvoiceRepository:
    """Get invoice repository dependency"""
    if session is None:
        async with async_session_factory() as session:
            return PostgreSQLInvoiceRepository(session)
    return PostgreSQLInvoiceRepository(session)


async def get_usage_record_repository(
    session: AsyncSession = None
) -> PostgreSQLUsageRecordRepository:
    """Get usage record repository dependency"""
    if session is None:
        async with async_session_factory() as session:
            return PostgreSQLUsageRecordRepository(session)
    return PostgreSQLUsageRecordRepository(session)


async def get_entitlement_service(
    session: AsyncSession = None
) -> EntitlementService:
    """Get entitlement service dependency"""
    subscription_repo = await get_subscription_repository(session)
    usage_repo = await get_usage_record_repository(session)
    return EntitlementService(
        subscription_repository=subscription_repo,
        usage_repository=usage_repo,
    )


async def get_usage_metering_service(
    session: AsyncSession = None
) -> UsageMeteringService:
    """Get usage metering service dependency"""
    usage_repo = await get_usage_record_repository(session)
    return UsageMeteringService(usage_repository=usage_repo)


async def get_billing_service(
    session: AsyncSession = None
) -> BillingService:
    """Get billing service dependency"""
    import os
    from infrastructure.billing.provider import (
        StripeBillingProvider,
        MockBillingProvider,
    )
    
    # Use environment variable to determine billing provider
    billing_mode = os.getenv("BILLING_PROVIDER", "mock").lower()
    
    if billing_mode == "stripe":
        stripe_api_key = os.getenv("STRIPE_API_KEY")
        if not stripe_api_key:
            raise ValueError("STRIPE_API_KEY must be set when using Stripe billing provider")
        billing_provider = StripeBillingProvider(api_key=stripe_api_key)
    elif billing_mode == "mock":
        billing_provider = MockBillingProvider()
    else:
        raise ValueError(f"Unknown billing provider: {billing_mode}. Use 'stripe' or 'mock'")
    
    subscription_repo = await get_subscription_repository(session)
    invoice_repo = await get_invoice_repository(session)
    
    return BillingService(
        subscription_repository=subscription_repo,
        invoice_repository=invoice_repo,
        billing_provider=billing_provider,
    )


# ==================== Plugin Dependencies ====================

async def get_plugin_repository(
    session: AsyncSession = None
) -> PostgreSQLPluginRepository:
    """Get plugin repository dependency"""
    if session is None:
        async with async_session_factory() as session:
            return PostgreSQLPluginRepository(session)
    return PostgreSQLPluginRepository(session)


async def get_plugin_execution_repository(
    session: AsyncSession = None
) -> PostgreSQLPluginExecutionRepository:
    """Get plugin execution repository dependency"""
    if session is None:
        async with async_session_factory() as session:
            return PostgreSQLPluginExecutionRepository(session)
    return PostgreSQLPluginExecutionRepository(session)


async def get_plugin_installation_repository(
    session: AsyncSession = None
) -> PostgreSQLPluginInstallationRepository:
    """Get plugin installation repository dependency"""
    if session is None:
        async with async_session_factory() as session:
            return PostgreSQLPluginInstallationRepository(session)
    return PostgreSQLPluginInstallationRepository(session)
