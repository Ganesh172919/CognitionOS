"""
Health Monitoring Domain - Events

Domain events for health monitoring operations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID


@dataclass(frozen=True)
class HealthMonitoringEvent:
    """Base health monitoring domain event"""
    agent_id: UUID
    workflow_execution_id: UUID
    occurred_at: datetime
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class HeartbeatReceived(HealthMonitoringEvent):
    """Event raised when agent heartbeat is received"""
    health_score: float
    status: str

    @classmethod
    def create(
        cls,
        agent_id: UUID,
        workflow_execution_id: UUID,
        health_score: float,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "HeartbeatReceived":
        """Factory method to create event"""
        return cls(
            agent_id=agent_id,
            workflow_execution_id=workflow_execution_id,
            health_score=health_score,
            status=status,
            occurred_at=datetime.utcnow(),
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class HealthDegraded(HealthMonitoringEvent):
    """Event raised when agent health degrades"""
    health_score: float
    reason: str
    previous_status: str

    @classmethod
    def create(
        cls,
        agent_id: UUID,
        workflow_execution_id: UUID,
        health_score: float,
        reason: str,
        previous_status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "HealthDegraded":
        """Factory method to create event"""
        return cls(
            agent_id=agent_id,
            workflow_execution_id=workflow_execution_id,
            health_score=health_score,
            reason=reason,
            previous_status=previous_status,
            occurred_at=datetime.utcnow(),
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class HealthFailed(HealthMonitoringEvent):
    """Event raised when agent health fails"""
    error_message: str
    health_score: float
    stale_heartbeat_seconds: Optional[float] = None

    @classmethod
    def create(
        cls,
        agent_id: UUID,
        workflow_execution_id: UUID,
        error_message: str,
        health_score: float,
        stale_heartbeat_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "HealthFailed":
        """Factory method to create event"""
        return cls(
            agent_id=agent_id,
            workflow_execution_id=workflow_execution_id,
            error_message=error_message,
            health_score=health_score,
            stale_heartbeat_seconds=stale_heartbeat_seconds,
            occurred_at=datetime.utcnow(),
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class HealthRecovered(HealthMonitoringEvent):
    """Event raised when agent health recovers"""
    recovery_attempt: int
    health_score: float
    previous_status: str

    @classmethod
    def create(
        cls,
        agent_id: UUID,
        workflow_execution_id: UUID,
        recovery_attempt: int,
        health_score: float,
        previous_status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "HealthRecovered":
        """Factory method to create event"""
        return cls(
            agent_id=agent_id,
            workflow_execution_id=workflow_execution_id,
            recovery_attempt=recovery_attempt,
            health_score=health_score,
            previous_status=previous_status,
            occurred_at=datetime.utcnow(),
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class IncidentCreated(HealthMonitoringEvent):
    """Event raised when a health incident is created"""
    incident_id: UUID
    severity: str
    title: str
    description: str

    @classmethod
    def create(
        cls,
        agent_id: UUID,
        workflow_execution_id: UUID,
        incident_id: UUID,
        severity: str,
        title: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "IncidentCreated":
        """Factory method to create event"""
        return cls(
            agent_id=agent_id,
            workflow_execution_id=workflow_execution_id,
            incident_id=incident_id,
            severity=severity,
            title=title,
            description=description,
            occurred_at=datetime.utcnow(),
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class IncidentResolved(HealthMonitoringEvent):
    """Event raised when a health incident is resolved"""
    incident_id: UUID
    resolution_notes: str
    time_to_resolution_seconds: float

    @classmethod
    def create(
        cls,
        agent_id: UUID,
        workflow_execution_id: UUID,
        incident_id: UUID,
        resolution_notes: str,
        time_to_resolution_seconds: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "IncidentResolved":
        """Factory method to create event"""
        return cls(
            agent_id=agent_id,
            workflow_execution_id=workflow_execution_id,
            incident_id=incident_id,
            resolution_notes=resolution_notes,
            time_to_resolution_seconds=time_to_resolution_seconds,
            occurred_at=datetime.utcnow(),
            metadata=metadata or {},
        )
