"""
Health Monitoring Domain - Services

Domain services for health monitoring orchestration and business logic.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from .entities import (
    AgentHealthStatus,
    AgentHealthIncident,
    HealthStatus,
    IncidentSeverity,
    ResourceMetrics,
    CostMetrics,
    TaskMetrics,
)
from .repositories import AgentHealthRepository, HealthIncidentRepository


class AgentHealthMonitoringService:
    """
    Domain service for agent health monitoring operations.
    
    Orchestrates health tracking, failure detection, and recovery.
    """

    def __init__(
        self,
        health_repository: AgentHealthRepository,
        incident_repository: HealthIncidentRepository,
    ):
        """
        Initialize health monitoring service.
        
        Args:
            health_repository: Agent health repository
            incident_repository: Health incident repository
        """
        self.health_repository = health_repository
        self.incident_repository = incident_repository

    async def record_heartbeat(
        self,
        agent_id: UUID,
        workflow_execution_id: UUID,
        resource_metrics: ResourceMetrics,
        cost_metrics: CostMetrics,
        task_metrics: TaskMetrics,
    ) -> AgentHealthStatus:
        """
        Record agent heartbeat and update health status.
        
        Args:
            agent_id: Agent ID
            workflow_execution_id: Workflow execution ID
            resource_metrics: Resource usage metrics
            cost_metrics: Cost metrics
            task_metrics: Task metrics
            
        Returns:
            Updated health status
        """
        # Find or create health status
        health_status = await self.health_repository.find_by_agent(agent_id)
        
        if not health_status:
            # Create new health status
            health_score = await self.calculate_health_score(
                resource_metrics=resource_metrics,
                cost_metrics=cost_metrics,
                task_metrics=task_metrics,
            )
            
            health_status = AgentHealthStatus.create(
                agent_id=agent_id,
                workflow_execution_id=workflow_execution_id,
                resource_metrics=resource_metrics,
                cost_metrics=cost_metrics,
                task_metrics=task_metrics,
                health_score=health_score,
            )
        else:
            # Update existing health status
            health_status.update_heartbeat()
            health_status.update_metrics(
                resource_metrics=resource_metrics,
                cost_metrics=cost_metrics,
                task_metrics=task_metrics,
            )
            
            # Recalculate health score
            health_score = await self.calculate_health_score(
                resource_metrics=resource_metrics,
                cost_metrics=cost_metrics,
                task_metrics=task_metrics,
            )
            health_status.update_health_score(health_score)
            
            # Update status based on health score
            if health_score >= 0.8:
                if health_status.status != HealthStatus.HEALTHY:
                    health_status.mark_as_healthy()
            elif health_score >= 0.5:
                if health_status.status == HealthStatus.HEALTHY:
                    health_status.mark_as_degraded(
                        reason=f"Health score degraded to {health_score:.2f}"
                    )
        
        # Save health status
        await self.health_repository.save(health_status)
        
        return health_status

    async def detect_failures(
        self,
        threshold_seconds: int = 30,
    ) -> List[AgentHealthStatus]:
        """
        Detect failed agents based on stale heartbeats.
        
        Args:
            threshold_seconds: Heartbeat staleness threshold (default: 30s)
            
        Returns:
            List of failed agent health statuses
        """
        # Find agents with stale heartbeats
        stale_agents = await self.health_repository.find_stale_heartbeats(
            threshold_seconds=threshold_seconds
        )
        
        failed_agents = []
        
        for health_status in stale_agents:
            if health_status.status != HealthStatus.FAILED:
                # Calculate stale duration
                stale_seconds = (datetime.utcnow() - health_status.last_heartbeat).total_seconds()
                
                # Mark as failed
                health_status.mark_as_failed(
                    error=f"No heartbeat for {stale_seconds:.0f} seconds (threshold: {threshold_seconds}s)"
                )
                
                # Update health score to 0
                health_status.update_health_score(0.0)
                
                # Save updated status
                await self.health_repository.save(health_status)
                
                # Create incident for failure
                await self.create_incident(
                    agent_id=health_status.agent_id,
                    workflow_execution_id=health_status.workflow_execution_id,
                    severity=IncidentSeverity.CRITICAL,
                    title="Agent Heartbeat Failure",
                    description=f"Agent has not sent heartbeat for {stale_seconds:.0f} seconds",
                    error_message=health_status.error_message,
                    health_score=health_status.health_score,
                    failure_rate=health_status.get_failure_rate(),
                )
                
                failed_agents.append(health_status)
        
        return failed_agents

    async def calculate_health_score(
        self,
        resource_metrics: ResourceMetrics,
        cost_metrics: CostMetrics,
        task_metrics: TaskMetrics,
    ) -> float:
        """
        Calculate composite health score (0-1).
        
        Algorithm:
        - Resource health: 40% weight
        - Cost health: 30% weight
        - Task success: 30% weight
        
        Args:
            resource_metrics: Resource metrics
            cost_metrics: Cost metrics
            task_metrics: Task metrics
            
        Returns:
            Health score between 0.0 and 1.0
        """
        # Resource health score (memory and CPU)
        memory_score = max(0.0, 1.0 - (resource_metrics.memory_usage_mb / 2048.0))  # Normalize to 2GB
        cpu_score = max(0.0, 1.0 - (resource_metrics.cpu_usage_percent / 100.0))
        resource_score = (memory_score + cpu_score) / 2.0
        
        # Cost health score (budget remaining)
        total_budget = cost_metrics.cost_consumed + cost_metrics.budget_remaining
        cost_score = (
            cost_metrics.budget_remaining / total_budget
            if total_budget > 0
            else 1.0
        )
        
        # Task success score
        total_tasks = task_metrics.completed_tasks_count + task_metrics.failed_tasks_count
        task_score = (
            task_metrics.completed_tasks_count / total_tasks
            if total_tasks > 0
            else 1.0
        )
        
        # Composite score with weights
        health_score = (
            (resource_score * 0.4) +
            (cost_score * 0.3) +
            (task_score * 0.3)
        )
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, health_score))

    async def create_incident(
        self,
        agent_id: UUID,
        workflow_execution_id: UUID,
        severity: IncidentSeverity,
        title: str,
        description: str,
        error_message: Optional[str] = None,
        health_score: float = 0.0,
        failure_rate: float = 0.0,
    ) -> AgentHealthIncident:
        """
        Create a health incident.
        
        Args:
            agent_id: Agent ID
            workflow_execution_id: Workflow execution ID
            severity: Incident severity
            title: Incident title
            description: Incident description
            error_message: Optional error message
            health_score: Health score at incident time
            failure_rate: Task failure rate at incident time
            
        Returns:
            Created incident
        """
        incident = AgentHealthIncident.create(
            agent_id=agent_id,
            workflow_execution_id=workflow_execution_id,
            severity=severity,
            title=title,
            description=description,
            error_message=error_message,
            health_score=health_score,
            failure_rate=failure_rate,
        )
        
        await self.incident_repository.save(incident)
        
        return incident

    async def trigger_recovery(
        self,
        agent_id: UUID,
    ) -> AgentHealthStatus:
        """
        Trigger recovery for a failed agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Health status marked as recovering
            
        Raises:
            ValueError: If agent not found or not in failed state
        """
        # Find current health status
        health_status = await self.health_repository.find_by_agent(agent_id)
        if not health_status:
            raise ValueError(f"Agent health status not found: {agent_id}")
        
        # Verify agent is in failed or degraded state
        if not health_status.is_failing():
            raise ValueError(
                f"Cannot trigger recovery for agent in status: {health_status.status}"
            )
        
        # Mark as recovering
        health_status.mark_as_recovering()
        
        # Save updated status
        await self.health_repository.save(health_status)
        
        return health_status

    async def resolve_incident(
        self,
        incident_id: UUID,
        resolution: str,
    ) -> AgentHealthIncident:
        """
        Resolve a health incident.
        
        Args:
            incident_id: Incident ID
            resolution: Resolution notes
            
        Returns:
            Resolved incident
            
        Raises:
            ValueError: If incident not found
        """
        # Find incident
        incident = await self.incident_repository.find_by_id(incident_id)
        if not incident:
            raise ValueError(f"Incident not found: {incident_id}")
        
        # Mark as resolved
        incident.mark_as_resolved(resolution)
        
        # Save updated incident
        await self.incident_repository.save(incident)
        
        return incident

    async def get_agent_health_summary(
        self,
        workflow_execution_id: UUID,
    ) -> dict:
        """
        Get health summary for all agents in a workflow execution.
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            Dict with health summary metrics
        """
        # Get all health statuses
        health_statuses = await self.health_repository.find_by_workflow_execution(
            workflow_execution_id
        )
        
        # Calculate summary metrics
        total_agents = len(health_statuses)
        healthy_count = sum(1 for h in health_statuses if h.status == HealthStatus.HEALTHY)
        degraded_count = sum(1 for h in health_statuses if h.status == HealthStatus.DEGRADED)
        failed_count = sum(1 for h in health_statuses if h.status == HealthStatus.FAILED)
        recovering_count = sum(1 for h in health_statuses if h.status == HealthStatus.RECOVERING)
        
        # Calculate average health score
        avg_health_score = (
            sum(h.health_score for h in health_statuses) / total_agents
            if total_agents > 0
            else 0.0
        )
        
        # Get open incidents
        open_incidents = await self.incident_repository.find_open_incidents(
            workflow_execution_id
        )
        
        return {
            "total_agents": total_agents,
            "healthy_count": healthy_count,
            "degraded_count": degraded_count,
            "failed_count": failed_count,
            "recovering_count": recovering_count,
            "average_health_score": avg_health_score,
            "open_incidents_count": len(open_incidents),
            "health_percentage": (healthy_count / total_agents * 100) if total_agents > 0 else 0.0,
        }

    def validate_health_status_invariants(
        self,
        health_status: AgentHealthStatus,
    ) -> List[str]:
        """
        Validate health status business invariants.
        
        Args:
            health_status: Health status to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Validate health score
        if health_status.health_score < 0.0 or health_status.health_score > 1.0:
            errors.append("Health score must be between 0.0 and 1.0")

        # Validate metrics
        if health_status.resource_metrics.memory_usage_mb < 0:
            errors.append("Memory usage cannot be negative")

        if health_status.resource_metrics.cpu_usage_percent < 0 or health_status.resource_metrics.cpu_usage_percent > 100:
            errors.append("CPU usage must be between 0 and 100")

        if health_status.cost_metrics.cost_consumed < 0:
            errors.append("Cost consumed cannot be negative")

        if health_status.task_metrics.active_tasks_count < 0:
            errors.append("Active tasks count cannot be negative")

        # Validate recovery attempts
        if health_status.recovery_attempts < 0:
            errors.append("Recovery attempts cannot be negative")

        # Validate heartbeat
        if health_status.last_heartbeat > datetime.utcnow():
            errors.append("Last heartbeat cannot be in the future")

        return errors
