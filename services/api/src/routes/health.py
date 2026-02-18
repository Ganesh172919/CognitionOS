"""
Health Monitoring API Routes

Provides REST endpoints for agent health monitoring, incident management,
and system health checks.
"""

import os
from typing import List, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.application.health_monitoring.use_cases import (
    RecordHeartbeatCommand,
    AgentHealthStatusQuery,
    CreateIncidentCommand,
    TriggerRecoveryCommand,
    RecordHeartbeatUseCase,
    DetectHealthFailuresUseCase,
    GetAgentHealthStatusUseCase,
    CreateHealthIncidentUseCase,
    TriggerRecoveryUseCase,
)
from services.api.src.schemas.phase3 import (
    RecordHeartbeatRequest,
    DetectFailuresRequest,
    CreateHealthIncidentRequest,
    HealthStatusResponse,
    HealthIncidentResponse,
    HealthStatusListResponse,
)
from services.api.src.dependencies.injection import (
    get_db_session,
    get_health_monitoring_service,
    get_health_repository,
    get_event_bus,
)
from infrastructure.health.checks import SystemHealthAggregator


router = APIRouter(prefix="/api/v3/health", tags=["Health Monitoring"])


@router.post(
    "/heartbeat",
    response_model=HealthStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Record agent heartbeat",
    description="Record agent heartbeat with resource, cost, and task metrics",
)
async def record_heartbeat(
    request: RecordHeartbeatRequest,
    session: AsyncSession = Depends(get_db_session),
) -> HealthStatusResponse:
    """Record agent heartbeat"""
    try:
        # Get health monitoring service and use case
        health_monitoring_service = await get_health_monitoring_service(session)
        event_bus = get_event_bus()
        use_case = RecordHeartbeatUseCase(
            health_monitoring_service=health_monitoring_service,
            event_publisher=event_bus,
        )
        
        # Convert request to command
        command = RecordHeartbeatCommand(
            agent_id=request.agent_id,
            workflow_execution_id=request.workflow_execution_id,
            memory_usage_mb=request.memory_usage_mb,
            cpu_usage_percent=request.cpu_usage_percent,
            working_memory_count=request.working_memory_count,
            episodic_memory_count=request.episodic_memory_count,
            cost_consumed=request.cost_consumed,
            budget_remaining=request.budget_remaining,
            active_tasks_count=request.active_tasks_count,
            completed_tasks_count=request.completed_tasks_count,
            failed_tasks_count=request.failed_tasks_count,
        )
        
        # Execute use case
        result = await use_case.execute(command)
        
        return HealthStatusResponse(
            agent_id=result.agent_id,
            workflow_execution_id=result.workflow_execution_id,
            status=result.status.value,
            health_score=result.health_score,
            last_heartbeat=result.last_heartbeat,
            memory_usage_mb=result.memory_usage_mb,
            cpu_usage_percent=result.cpu_usage_percent,
            cost_consumed=result.cost_consumed,
            budget_remaining=result.budget_remaining,
            active_tasks_count=result.active_tasks_count,
            completed_tasks_count=result.completed_tasks_count,
            failed_tasks_count=result.failed_tasks_count,
            error_message=result.error_message,
            recovery_attempts=result.recovery_attempts,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record heartbeat: {str(e)}"
        )


@router.get(
    "/agent/{agent_id}",
    response_model=HealthStatusResponse,
    summary="Get agent health status",
    description="Retrieve current health status for a specific agent",
)
async def get_agent_health_status(
    agent_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> HealthStatusResponse:
    """Get agent health status"""
    try:
        # Get health repository and use case
        health_repository = await get_health_repository(session)
        use_case = GetAgentHealthStatusUseCase(health_repository=health_repository)
        
        # Create query
        query = AgentHealthStatusQuery(agent_id=agent_id)
        
        # Execute use case
        result = await use_case.execute(query)
        
        return HealthStatusResponse(
            agent_id=result.agent_id,
            workflow_execution_id=result.workflow_execution_id,
            status=result.status.value,
            health_score=result.health_score,
            last_heartbeat=result.last_heartbeat,
            memory_usage_mb=result.memory_usage_mb,
            cpu_usage_percent=result.cpu_usage_percent,
            cost_consumed=result.cost_consumed,
            budget_remaining=result.budget_remaining,
            active_tasks_count=result.active_tasks_count,
            completed_tasks_count=result.completed_tasks_count,
            failed_tasks_count=result.failed_tasks_count,
            error_message=result.error_message,
            recovery_attempts=result.recovery_attempts,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve agent health status: {str(e)}"
        )


@router.post(
    "/detect-failures",
    response_model=HealthStatusListResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect health failures",
    description="Detect failed agents based on stale heartbeats",
)
async def detect_failures(
    request: DetectFailuresRequest,
    session: AsyncSession = Depends(get_db_session),
) -> HealthStatusListResponse:
    """Detect health failures"""
    try:
        # Get health monitoring service and use case
        health_monitoring_service = await get_health_monitoring_service(session)
        event_bus = get_event_bus()
        use_case = DetectHealthFailuresUseCase(
            health_monitoring_service=health_monitoring_service,
            event_publisher=event_bus,
        )
        
        # Execute use case
        results = await use_case.execute(threshold_seconds=request.threshold_seconds)
        
        return HealthStatusListResponse(
            health_statuses=[
                HealthStatusResponse(
                    agent_id=r.agent_id,
                    workflow_execution_id=r.workflow_execution_id,
                    status=r.status.value,
                    health_score=r.health_score,
                    last_heartbeat=r.last_heartbeat,
                    memory_usage_mb=r.memory_usage_mb,
                    cpu_usage_percent=r.cpu_usage_percent,
                    cost_consumed=r.cost_consumed,
                    budget_remaining=r.budget_remaining,
                    active_tasks_count=r.active_tasks_count,
                    completed_tasks_count=r.completed_tasks_count,
                    failed_tasks_count=r.failed_tasks_count,
                    error_message=r.error_message,
                    recovery_attempts=r.recovery_attempts,
                )
                for r in results
            ],
            total=len(results),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect health failures: {str(e)}"
        )


@router.post(
    "/incidents",
    response_model=HealthIncidentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create health incident",
    description="Create a health incident for an agent",
)
async def create_incident(
    request: CreateHealthIncidentRequest,
    session: AsyncSession = Depends(get_db_session),
) -> HealthIncidentResponse:
    """Create health incident"""
    try:
        # Get health monitoring service and use case
        health_monitoring_service = await get_health_monitoring_service(session)
        event_bus = get_event_bus()
        use_case = CreateHealthIncidentUseCase(
            health_monitoring_service=health_monitoring_service,
            event_publisher=event_bus,
        )
        
        # Convert request to command
        command = CreateIncidentCommand(
            agent_id=request.agent_id,
            workflow_execution_id=request.workflow_execution_id,
            severity=request.severity,
            title=request.title,
            description=request.description,
            error_message=request.error_message,
            health_score=request.health_score,
            failure_rate=request.failure_rate,
        )
        
        # Execute use case
        result = await use_case.execute(command)
        
        return HealthIncidentResponse(
            incident_id=result.incident_id,
            agent_id=result.agent_id,
            workflow_execution_id=result.workflow_execution_id,
            severity=result.severity.value,
            status=result.status.value,
            title=result.title,
            description=result.description,
            error_message=result.error_message,
            health_score=result.health_score,
            failure_rate=result.failure_rate,
            created_at=result.created_at,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create incident: {str(e)}"
        )


@router.post(
    "/recover/{agent_id}",
    response_model=HealthStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger agent recovery",
    description="Trigger recovery for a failed agent",
)
async def trigger_recovery(
    agent_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> HealthStatusResponse:
    """Trigger agent recovery"""
    try:
        # Get health monitoring service and use case
        health_monitoring_service = await get_health_monitoring_service(session)
        event_bus = get_event_bus()
        use_case = TriggerRecoveryUseCase(
            health_monitoring_service=health_monitoring_service,
            event_publisher=event_bus,
        )
        
        # Convert request to command
        command = TriggerRecoveryCommand(agent_id=agent_id)
        
        # Execute use case
        result = await use_case.execute(command)
        
        return HealthStatusResponse(
            agent_id=result.agent_id,
            workflow_execution_id=result.workflow_execution_id,
            status=result.status.value,
            health_score=result.health_score,
            last_heartbeat=result.last_heartbeat,
            memory_usage_mb=result.memory_usage_mb,
            cpu_usage_percent=result.cpu_usage_percent,
            cost_consumed=result.cost_consumed,
            budget_remaining=result.budget_remaining,
            active_tasks_count=result.active_tasks_count,
            completed_tasks_count=result.completed_tasks_count,
            failed_tasks_count=result.failed_tasks_count,
            error_message=result.error_message,
            recovery_attempts=result.recovery_attempts,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger recovery: {str(e)}"
        )


# ==================== System Health Check Endpoints ====================

@router.get(
    "/system",
    summary="System health check",
    description="Comprehensive health check for all system dependencies (Redis, RabbitMQ, Database)",
    response_model=Dict[str, Any],
)
async def system_health(
    session: AsyncSession = Depends(get_db_session),
) -> Dict[str, Any]:
    """
    Perform comprehensive system health check.
    
    Checks:
    - Redis connectivity and performance
    - RabbitMQ connectivity and messaging
    - Database connectivity and query performance
    
    Returns overall status (healthy/degraded/unhealthy) with details.
    """
    try:
        from core.config import get_config
        config = get_config()
        
        # Get Redis and RabbitMQ URLs from config
        redis_url = config.redis.url if hasattr(config, 'redis') else "redis://localhost:6379/0"
        rabbitmq_url = config.rabbitmq.url if hasattr(config, 'rabbitmq') else "amqp://guest:guest@localhost:5672/"
        
        # Create health aggregator
        health_aggregator = SystemHealthAggregator(
            redis_url=redis_url,
            rabbitmq_url=rabbitmq_url,
        )
        
        # Perform all health checks
        result = await health_aggregator.check_all(session)
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform system health check: {str(e)}"
        )


@router.get(
    "/ready",
    summary="Readiness probe",
    description="Kubernetes readiness probe - checks if system is ready to accept traffic",
    status_code=status.HTTP_200_OK,
)
async def readiness_probe(
    session: AsyncSession = Depends(get_db_session),
) -> Dict[str, Any]:
    """
    Readiness probe for Kubernetes.
    
    Returns 200 if system is ready to accept traffic, 503 otherwise.
    """
    try:
        from core.config import get_config
        config = get_config()
        
        redis_url = config.redis.url if hasattr(config, 'redis') else "redis://localhost:6379/0"
        rabbitmq_url = config.rabbitmq.url if hasattr(config, 'rabbitmq') else "amqp://guest:guest@localhost:5672/"
        
        health_aggregator = SystemHealthAggregator(
            redis_url=redis_url,
            rabbitmq_url=rabbitmq_url,
        )
        
        result = await health_aggregator.check_all(session)
        
        # Service is ready if not unhealthy
        if result["status"] == "unhealthy":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="System is not ready"
            )
        
        return {
            "status": "ready",
            "timestamp": result["timestamp"],
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Readiness check failed: {str(e)}"
        )


@router.get(
    "/live",
    summary="Liveness probe",
    description="Kubernetes liveness probe - checks if application is alive",
    status_code=status.HTTP_200_OK,
)
async def liveness_probe() -> Dict[str, str]:
    """
    Liveness probe for Kubernetes.
    
    Simple check to verify the application is alive and responding.
    Returns 200 if alive, indicating Kubernetes should not restart the pod.
    """
    from datetime import datetime, timezone
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
