"""
Health Monitoring Infrastructure - PostgreSQL Repository Implementation

Concrete implementation of AgentHealthRepository and HealthIncidentRepository using PostgreSQL.
"""

from typing import List, Optional
from uuid import UUID
from datetime import datetime, timedelta

from sqlalchemy import select, delete, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from core.domain.health_monitoring.entities import (
    AgentHealthStatus,
    AgentHealthIncident,
    HealthStatus,
    IncidentSeverity,
    IncidentStatus,
    ResourceMetrics,
    CostMetrics,
    TaskMetrics
)
from core.domain.health_monitoring.repositories import (
    AgentHealthRepository,
    HealthIncidentRepository
)

from infrastructure.persistence.health_models import (
    AgentHealthStatusModel,
    AgentHealthIncidentModel,
    HealthStatusEnum,
    IncidentSeverityEnum,
    IncidentStatusEnum
)


class PostgreSQLAgentHealthRepository(AgentHealthRepository):
    """
    PostgreSQL implementation of AgentHealthRepository.

    Maps between domain entities and SQLAlchemy models.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, health_status: AgentHealthStatus) -> None:
        """Persist agent health status to database"""
        model = self._to_model(health_status)
        self.session.add(model)
        await self.session.flush()

    async def find_by_id(self, health_status_id: UUID) -> Optional[AgentHealthStatus]:
        """Retrieve health status by ID"""
        stmt = select(AgentHealthStatusModel).where(AgentHealthStatusModel.id == health_status_id)
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._to_entity(model)

    async def find_by_agent(self, agent_id: UUID) -> Optional[AgentHealthStatus]:
        """Find current health status for an agent"""
        stmt = (
            select(AgentHealthStatusModel)
            .where(AgentHealthStatusModel.agent_id == agent_id)
            .order_by(AgentHealthStatusModel.updated_at.desc())
            .limit(1)
        )
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._to_entity(model)

    async def find_by_workflow_execution(
        self,
        workflow_execution_id: UUID,
    ) -> List[AgentHealthStatus]:
        """Find all health statuses for a workflow execution"""
        stmt = (
            select(AgentHealthStatusModel)
            .where(AgentHealthStatusModel.workflow_execution_id == workflow_execution_id)
            .order_by(AgentHealthStatusModel.updated_at.desc())
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]

    async def find_failing_agents(
        self,
        workflow_execution_id: Optional[UUID] = None,
    ) -> List[AgentHealthStatus]:
        """Find agents with failing health status"""
        stmt = select(AgentHealthStatusModel).where(
            AgentHealthStatusModel.status.in_([
                HealthStatusEnum.DEGRADED,
                HealthStatusEnum.FAILED
            ])
        )
        
        if workflow_execution_id:
            stmt = stmt.where(AgentHealthStatusModel.workflow_execution_id == workflow_execution_id)
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]

    async def find_stale_heartbeats(
        self,
        threshold_seconds: int = 30,
    ) -> List[AgentHealthStatus]:
        """Find agents with stale heartbeats"""
        threshold_time = datetime.utcnow() - timedelta(seconds=threshold_seconds)
        
        stmt = select(AgentHealthStatusModel).where(
            AgentHealthStatusModel.last_heartbeat < threshold_time
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]

    async def delete(self, health_status_id: UUID) -> bool:
        """Delete a health status"""
        stmt = select(AgentHealthStatusModel).where(AgentHealthStatusModel.id == health_status_id)
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return False
        
        await self.session.delete(model)
        await self.session.flush()
        return True

    def _to_model(self, health_status: AgentHealthStatus) -> AgentHealthStatusModel:
        """Convert domain entity to ORM model"""
        return AgentHealthStatusModel(
            id=health_status.id,
            agent_id=health_status.agent_id,
            workflow_execution_id=health_status.workflow_execution_id,
            status=HealthStatusEnum(health_status.status.value),
            last_heartbeat=health_status.last_heartbeat,
            memory_usage_mb=health_status.resource_metrics.memory_usage_mb,
            cpu_usage_percent=health_status.resource_metrics.cpu_usage_percent,
            working_memory_count=health_status.resource_metrics.working_memory_count,
            episodic_memory_count=health_status.resource_metrics.episodic_memory_count,
            cost_consumed=health_status.cost_metrics.cost_consumed,
            budget_remaining=health_status.cost_metrics.budget_remaining,
            active_tasks_count=health_status.task_metrics.active_tasks_count,
            completed_tasks_count=health_status.task_metrics.completed_tasks_count,
            failed_tasks_count=health_status.task_metrics.failed_tasks_count,
            health_score=health_status.health_score,
            error_message=health_status.error_message,
            recovery_attempts=health_status.recovery_attempts,
            metadata=health_status.metadata,
            created_at=health_status.created_at,
            updated_at=health_status.updated_at
        )

    def _to_entity(self, model: AgentHealthStatusModel) -> AgentHealthStatus:
        """Convert ORM model to domain entity"""
        return AgentHealthStatus(
            id=model.id,
            agent_id=model.agent_id,
            workflow_execution_id=model.workflow_execution_id,
            status=HealthStatus(model.status.value),
            last_heartbeat=model.last_heartbeat,
            resource_metrics=ResourceMetrics(
                memory_usage_mb=model.memory_usage_mb or 0.0,
                cpu_usage_percent=model.cpu_usage_percent or 0.0,
                working_memory_count=model.working_memory_count or 0,
                episodic_memory_count=model.episodic_memory_count or 0
            ),
            cost_metrics=CostMetrics(
                cost_consumed=model.cost_consumed or 0.0,
                budget_remaining=model.budget_remaining or 0.0
            ),
            task_metrics=TaskMetrics(
                active_tasks_count=model.active_tasks_count or 0,
                completed_tasks_count=model.completed_tasks_count or 0,
                failed_tasks_count=model.failed_tasks_count or 0
            ),
            health_score=model.health_score or 0.0,
            error_message=model.error_message,
            recovery_attempts=model.recovery_attempts or 0,
            metadata=model.metadata or {},
            created_at=model.created_at,
            updated_at=model.updated_at
        )


class PostgreSQLHealthIncidentRepository(HealthIncidentRepository):
    """
    PostgreSQL implementation of HealthIncidentRepository.

    Maps between domain entities and SQLAlchemy models.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, incident: AgentHealthIncident) -> None:
        """Persist health incident to database"""
        model = self._to_model(incident)
        self.session.add(model)
        await self.session.flush()

    async def find_by_id(self, incident_id: UUID) -> Optional[AgentHealthIncident]:
        """Retrieve incident by ID"""
        stmt = select(AgentHealthIncidentModel).where(AgentHealthIncidentModel.id == incident_id)
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._to_entity(model)

    async def find_by_agent(
        self,
        agent_id: UUID,
        limit: Optional[int] = None,
    ) -> List[AgentHealthIncident]:
        """Find incidents for an agent"""
        stmt = (
            select(AgentHealthIncidentModel)
            .where(AgentHealthIncidentModel.agent_id == agent_id)
            .order_by(AgentHealthIncidentModel.created_at.desc())
        )
        
        if limit:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]

    async def find_by_workflow_execution(
        self,
        workflow_execution_id: UUID,
        limit: Optional[int] = None,
    ) -> List[AgentHealthIncident]:
        """Find incidents for a workflow execution"""
        stmt = (
            select(AgentHealthIncidentModel)
            .where(AgentHealthIncidentModel.workflow_execution_id == workflow_execution_id)
            .order_by(AgentHealthIncidentModel.created_at.desc())
        )
        
        if limit:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]

    async def find_open_incidents(
        self,
        workflow_execution_id: Optional[UUID] = None,
    ) -> List[AgentHealthIncident]:
        """Find open (unresolved) incidents"""
        stmt = select(AgentHealthIncidentModel).where(
            AgentHealthIncidentModel.status.in_([
                IncidentStatusEnum.OPEN,
                IncidentStatusEnum.INVESTIGATING
            ])
        )
        
        if workflow_execution_id:
            stmt = stmt.where(AgentHealthIncidentModel.workflow_execution_id == workflow_execution_id)
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]

    async def find_critical_incidents(
        self,
        workflow_execution_id: Optional[UUID] = None,
    ) -> List[AgentHealthIncident]:
        """Find critical severity incidents"""
        stmt = select(AgentHealthIncidentModel).where(
            AgentHealthIncidentModel.severity == IncidentSeverityEnum.CRITICAL
        )
        
        if workflow_execution_id:
            stmt = stmt.where(AgentHealthIncidentModel.workflow_execution_id == workflow_execution_id)
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]

    async def get_incident_count(
        self,
        agent_id: UUID,
    ) -> int:
        """Get count of incidents for an agent"""
        stmt = (
            select(func.count())
            .select_from(AgentHealthIncidentModel)
            .where(AgentHealthIncidentModel.agent_id == agent_id)
        )
        
        result = await self.session.execute(stmt)
        count = result.scalar_one()
        
        return count

    async def delete(self, incident_id: UUID) -> bool:
        """Delete an incident"""
        stmt = select(AgentHealthIncidentModel).where(AgentHealthIncidentModel.id == incident_id)
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return False
        
        await self.session.delete(model)
        await self.session.flush()
        return True

    def _to_model(self, incident: AgentHealthIncident) -> AgentHealthIncidentModel:
        """Convert domain entity to ORM model"""
        # Create metrics snapshot
        metrics_snapshot = {
            "health_score": incident.health_score,
            "failure_rate": incident.failure_rate
        }
        
        return AgentHealthIncidentModel(
            id=incident.id,
            agent_id=incident.agent_id,
            workflow_execution_id=incident.workflow_execution_id,
            incident_type=incident.title,
            severity=IncidentSeverityEnum(incident.severity.value),
            description=incident.description,
            status=IncidentStatusEnum(incident.status.value),
            resolved_at=incident.resolved_at,
            resolution_notes=incident.resolution_notes,
            recovery_action=None,
            recovery_successful=None,
            created_at=incident.created_at,
            metrics_snapshot=metrics_snapshot,
            metadata=incident.metadata
        )

    def _to_entity(self, model: AgentHealthIncidentModel) -> AgentHealthIncident:
        """Convert ORM model to domain entity"""
        metrics = model.metrics_snapshot or {}
        
        return AgentHealthIncident(
            id=model.id,
            agent_id=model.agent_id,
            workflow_execution_id=model.workflow_execution_id,
            severity=IncidentSeverity(model.severity.value),
            status=IncidentStatus(model.status.value),
            title=model.incident_type,
            description=model.description,
            error_message=None,
            health_score=metrics.get("health_score", 0.0),
            failure_rate=metrics.get("failure_rate", 0.0),
            resolution_notes=model.resolution_notes,
            resolved_at=model.resolved_at,
            metadata=model.metadata or {},
            created_at=model.created_at,
            updated_at=model.created_at
        )
