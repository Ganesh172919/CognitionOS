"""
Health Monitoring Application - Use Cases

Application layer use cases for Health Monitoring bounded context.
Orchestrates domain entities and coordinates with infrastructure.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import UUID

from core.domain.health_monitoring import (
    AgentHealthStatus,
    AgentHealthIncident,
    HealthStatus,
    IncidentSeverity,
    IncidentStatus,
    ResourceMetrics,
    CostMetrics,
    TaskMetrics,
    AgentHealthRepository,
    HealthIncidentRepository,
    AgentHealthMonitoringService,
    HeartbeatReceived,
    HealthDegraded,
    HealthFailed,
    HealthRecovered,
    IncidentCreated,
    IncidentResolved,
)


# ==================== DTOs (Data Transfer Objects) ====================

@dataclass
class RecordHeartbeatCommand:
    """Command to record agent heartbeat"""
    agent_id: UUID
    workflow_execution_id: UUID
    memory_usage_mb: float
    cpu_usage_percent: float
    working_memory_count: int
    episodic_memory_count: int
    cost_consumed: float
    budget_remaining: float
    active_tasks_count: int
    completed_tasks_count: int
    failed_tasks_count: int


@dataclass
class AgentHealthStatusQuery:
    """Query to get agent health status"""
    agent_id: UUID


@dataclass
class CreateIncidentCommand:
    """Command to create health incident"""
    agent_id: UUID
    workflow_execution_id: UUID
    severity: str
    title: str
    description: str
    error_message: Optional[str] = None
    health_score: float = 0.0
    failure_rate: float = 0.0


@dataclass
class TriggerRecoveryCommand:
    """Command to trigger agent recovery"""
    agent_id: UUID


@dataclass
class HealthStatusResult:
    """Result of health status operation"""
    agent_id: UUID
    workflow_execution_id: UUID
    status: HealthStatus
    health_score: float
    last_heartbeat: str
    memory_usage_mb: float
    cpu_usage_percent: float
    cost_consumed: float
    budget_remaining: float
    active_tasks_count: int
    completed_tasks_count: int
    failed_tasks_count: int
    error_message: Optional[str] = None
    recovery_attempts: int = 0


@dataclass
class IncidentResult:
    """Result of incident operation"""
    incident_id: UUID
    agent_id: UUID
    workflow_execution_id: UUID
    severity: IncidentSeverity
    status: IncidentStatus
    title: str
    description: str
    error_message: Optional[str] = None
    health_score: float = 0.0
    failure_rate: float = 0.0
    created_at: str = None


# ==================== Use Cases ====================

class RecordHeartbeatUseCase:
    """
    Use Case: Record agent heartbeat.

    Orchestrates:
    1. Build metrics from command
    2. Record heartbeat via domain service
    3. Update health status
    4. Publish domain event
    """

    def __init__(
        self,
        health_monitoring_service: AgentHealthMonitoringService,
        event_publisher: Optional[Any] = None
    ):
        self.health_monitoring_service = health_monitoring_service
        self.event_publisher = event_publisher

    async def execute(self, command: RecordHeartbeatCommand) -> HealthStatusResult:
        """
        Record agent heartbeat.

        Args:
            command: Record heartbeat command

        Returns:
            HealthStatusResult with updated health status

        Raises:
            ValueError: If heartbeat recording fails
        """
        # Build metrics
        resource_metrics = ResourceMetrics(
            memory_usage_mb=command.memory_usage_mb,
            cpu_usage_percent=command.cpu_usage_percent,
            working_memory_count=command.working_memory_count,
            episodic_memory_count=command.episodic_memory_count,
        )

        cost_metrics = CostMetrics(
            cost_consumed=command.cost_consumed,
            budget_remaining=command.budget_remaining,
        )

        task_metrics = TaskMetrics(
            active_tasks_count=command.active_tasks_count,
            completed_tasks_count=command.completed_tasks_count,
            failed_tasks_count=command.failed_tasks_count,
        )

        # Record heartbeat via domain service
        health_status = await self.health_monitoring_service.record_heartbeat(
            agent_id=command.agent_id,
            workflow_execution_id=command.workflow_execution_id,
            resource_metrics=resource_metrics,
            cost_metrics=cost_metrics,
            task_metrics=task_metrics,
        )

        # Publish event
        if self.event_publisher:
            event = HeartbeatReceived.create(
                agent_id=command.agent_id,
                workflow_execution_id=command.workflow_execution_id,
                health_score=health_status.health_score,
                status=health_status.status.value,
            )
            await self.event_publisher.publish(event)

        return HealthStatusResult(
            agent_id=health_status.agent_id,
            workflow_execution_id=health_status.workflow_execution_id,
            status=health_status.status,
            health_score=health_status.health_score,
            last_heartbeat=health_status.last_heartbeat.isoformat(),
            memory_usage_mb=health_status.resource_metrics.memory_usage_mb,
            cpu_usage_percent=health_status.resource_metrics.cpu_usage_percent,
            cost_consumed=health_status.cost_metrics.cost_consumed,
            budget_remaining=health_status.cost_metrics.budget_remaining,
            active_tasks_count=health_status.task_metrics.active_tasks_count,
            completed_tasks_count=health_status.task_metrics.completed_tasks_count,
            failed_tasks_count=health_status.task_metrics.failed_tasks_count,
            error_message=health_status.error_message,
            recovery_attempts=health_status.recovery_attempts,
        )


class DetectHealthFailuresUseCase:
    """
    Use Case: Detect failed agents based on stale heartbeats.

    Orchestrates:
    1. Detect failures via domain service (30s threshold)
    2. Mark agents as failed
    3. Create incidents
    4. Publish domain events
    """

    def __init__(
        self,
        health_monitoring_service: AgentHealthMonitoringService,
        event_publisher: Optional[Any] = None
    ):
        self.health_monitoring_service = health_monitoring_service
        self.event_publisher = event_publisher

    async def execute(self, threshold_seconds: int = 30) -> list[HealthStatusResult]:
        """
        Detect health failures.

        Args:
            threshold_seconds: Heartbeat staleness threshold (default: 30s)

        Returns:
            List of failed agent health status results
        """
        # Detect failures via domain service
        failed_agents = await self.health_monitoring_service.detect_failures(
            threshold_seconds=threshold_seconds
        )

        # Convert to results and publish events
        results = []
        for health_status in failed_agents:
            # Publish event
            if self.event_publisher:
                stale_seconds = 0.0
                if health_status.last_heartbeat:
                    from datetime import datetime
                    stale_seconds = (datetime.utcnow() - health_status.last_heartbeat).total_seconds()

                event = HealthFailed.create(
                    agent_id=health_status.agent_id,
                    workflow_execution_id=health_status.workflow_execution_id,
                    error_message=health_status.error_message or "Heartbeat timeout",
                    health_score=health_status.health_score,
                    stale_heartbeat_seconds=stale_seconds,
                )
                await self.event_publisher.publish(event)

            result = HealthStatusResult(
                agent_id=health_status.agent_id,
                workflow_execution_id=health_status.workflow_execution_id,
                status=health_status.status,
                health_score=health_status.health_score,
                last_heartbeat=health_status.last_heartbeat.isoformat(),
                memory_usage_mb=health_status.resource_metrics.memory_usage_mb,
                cpu_usage_percent=health_status.resource_metrics.cpu_usage_percent,
                cost_consumed=health_status.cost_metrics.cost_consumed,
                budget_remaining=health_status.cost_metrics.budget_remaining,
                active_tasks_count=health_status.task_metrics.active_tasks_count,
                completed_tasks_count=health_status.task_metrics.completed_tasks_count,
                failed_tasks_count=health_status.task_metrics.failed_tasks_count,
                error_message=health_status.error_message,
                recovery_attempts=health_status.recovery_attempts,
            )
            results.append(result)

        return results


class GetAgentHealthStatusUseCase:
    """
    Use Case: Get health status for an agent.

    Retrieves current health status for a specific agent.
    """

    def __init__(
        self,
        health_repository: AgentHealthRepository
    ):
        self.health_repository = health_repository

    async def execute(self, query: AgentHealthStatusQuery) -> HealthStatusResult:
        """
        Get agent health status.

        Args:
            query: Agent health status query

        Returns:
            HealthStatusResult with current health status

        Raises:
            ValueError: If agent health status not found
        """
        # Find health status
        health_status = await self.health_repository.find_by_agent(query.agent_id)
        if not health_status:
            raise ValueError(f"Agent health status not found: {query.agent_id}")

        return HealthStatusResult(
            agent_id=health_status.agent_id,
            workflow_execution_id=health_status.workflow_execution_id,
            status=health_status.status,
            health_score=health_status.health_score,
            last_heartbeat=health_status.last_heartbeat.isoformat(),
            memory_usage_mb=health_status.resource_metrics.memory_usage_mb,
            cpu_usage_percent=health_status.resource_metrics.cpu_usage_percent,
            cost_consumed=health_status.cost_metrics.cost_consumed,
            budget_remaining=health_status.cost_metrics.budget_remaining,
            active_tasks_count=health_status.task_metrics.active_tasks_count,
            completed_tasks_count=health_status.task_metrics.completed_tasks_count,
            failed_tasks_count=health_status.task_metrics.failed_tasks_count,
            error_message=health_status.error_message,
            recovery_attempts=health_status.recovery_attempts,
        )


class CreateHealthIncidentUseCase:
    """
    Use Case: Create health incident.

    Orchestrates:
    1. Create incident via domain service
    2. Publish domain event
    """

    def __init__(
        self,
        health_monitoring_service: AgentHealthMonitoringService,
        event_publisher: Optional[Any] = None
    ):
        self.health_monitoring_service = health_monitoring_service
        self.event_publisher = event_publisher

    async def execute(self, command: CreateIncidentCommand) -> IncidentResult:
        """
        Create health incident.

        Args:
            command: Create incident command

        Returns:
            IncidentResult with created incident details

        Raises:
            ValueError: If incident creation fails
        """
        # Parse severity
        severity = IncidentSeverity(command.severity)

        # Create incident via domain service
        incident = await self.health_monitoring_service.create_incident(
            agent_id=command.agent_id,
            workflow_execution_id=command.workflow_execution_id,
            severity=severity,
            title=command.title,
            description=command.description,
            error_message=command.error_message,
            health_score=command.health_score,
            failure_rate=command.failure_rate,
        )

        # Publish event
        if self.event_publisher:
            event = IncidentCreated.create(
                agent_id=command.agent_id,
                workflow_execution_id=command.workflow_execution_id,
                incident_id=incident.id,
                severity=severity.value,
                title=command.title,
                description=command.description,
            )
            await self.event_publisher.publish(event)

        return IncidentResult(
            incident_id=incident.id,
            agent_id=incident.agent_id,
            workflow_execution_id=incident.workflow_execution_id,
            severity=incident.severity,
            status=incident.status,
            title=incident.title,
            description=incident.description,
            error_message=incident.error_message,
            health_score=incident.health_score,
            failure_rate=incident.failure_rate,
            created_at=incident.created_at.isoformat(),
        )


class TriggerRecoveryUseCase:
    """
    Use Case: Trigger agent recovery.

    Orchestrates:
    1. Trigger recovery via domain service
    2. Mark agent as recovering
    3. Publish domain event
    """

    def __init__(
        self,
        health_monitoring_service: AgentHealthMonitoringService,
        event_publisher: Optional[Any] = None
    ):
        self.health_monitoring_service = health_monitoring_service
        self.event_publisher = event_publisher

    async def execute(self, command: TriggerRecoveryCommand) -> HealthStatusResult:
        """
        Trigger agent recovery.

        Args:
            command: Trigger recovery command

        Returns:
            HealthStatusResult with recovering health status

        Raises:
            ValueError: If recovery trigger fails
        """
        # Trigger recovery via domain service
        health_status = await self.health_monitoring_service.trigger_recovery(
            agent_id=command.agent_id
        )

        # Publish event
        if self.event_publisher:
            event = HealthRecovered.create(
                agent_id=health_status.agent_id,
                workflow_execution_id=health_status.workflow_execution_id,
                recovery_attempt=health_status.recovery_attempts,
                health_score=health_status.health_score,
                previous_status=HealthStatus.FAILED.value,
            )
            await self.event_publisher.publish(event)

        return HealthStatusResult(
            agent_id=health_status.agent_id,
            workflow_execution_id=health_status.workflow_execution_id,
            status=health_status.status,
            health_score=health_status.health_score,
            last_heartbeat=health_status.last_heartbeat.isoformat(),
            memory_usage_mb=health_status.resource_metrics.memory_usage_mb,
            cpu_usage_percent=health_status.resource_metrics.cpu_usage_percent,
            cost_consumed=health_status.cost_metrics.cost_consumed,
            budget_remaining=health_status.cost_metrics.budget_remaining,
            active_tasks_count=health_status.task_metrics.active_tasks_count,
            completed_tasks_count=health_status.task_metrics.completed_tasks_count,
            failed_tasks_count=health_status.task_metrics.failed_tasks_count,
            error_message=health_status.error_message,
            recovery_attempts=health_status.recovery_attempts,
        )
