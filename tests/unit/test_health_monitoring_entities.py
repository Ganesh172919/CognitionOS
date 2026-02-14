"""
Unit Tests for Health Monitoring Domain Entities

Tests for health monitoring domain entities and value objects.
"""

import pytest
from datetime import datetime
from uuid import uuid4

from core.domain.health_monitoring.entities import (
    AgentHealthStatus,
    AgentHealthIncident,
    HealthStatus,
    IncidentSeverity,
    IncidentStatus,
    ResourceMetrics,
    CostMetrics,
    TaskMetrics,
)


class TestResourceMetrics:
    """Tests for ResourceMetrics value object"""
    
    def test_resource_metrics_creation(self):
        """Test creating resource metrics"""
        metrics = ResourceMetrics(
            memory_usage_mb=512.5,
            cpu_usage_percent=45.2,
            working_memory_count=150,
            episodic_memory_count=500
        )
        assert metrics.memory_usage_mb == 512.5
        assert metrics.cpu_usage_percent == 45.2
        assert metrics.working_memory_count == 150
    
    def test_resource_metrics_to_dict(self):
        """Test converting resource metrics to dictionary"""
        metrics = ResourceMetrics(
            memory_usage_mb=1024.0,
            cpu_usage_percent=75.0,
            working_memory_count=200,
            episodic_memory_count=1000
        )
        data = metrics.to_dict()
        
        assert data["memory_usage_mb"] == 1024.0
        assert data["cpu_usage_percent"] == 75.0
    
    def test_resource_metrics_from_dict(self):
        """Test creating resource metrics from dictionary"""
        data = {
            "memory_usage_mb": 256.0,
            "cpu_usage_percent": 30.0,
            "working_memory_count": 50,
            "episodic_memory_count": 200
        }
        metrics = ResourceMetrics.from_dict(data)
        
        assert metrics.memory_usage_mb == 256.0
        assert metrics.working_memory_count == 50


class TestCostMetrics:
    """Tests for CostMetrics value object"""
    
    def test_cost_metrics_creation(self):
        """Test creating cost metrics"""
        metrics = CostMetrics(
            cost_consumed=25.50,
            budget_remaining=74.50
        )
        assert metrics.cost_consumed == 25.50
        assert metrics.budget_remaining == 74.50
    
    def test_cost_metrics_to_dict(self):
        """Test converting cost metrics to dictionary"""
        metrics = CostMetrics(
            cost_consumed=10.0,
            budget_remaining=90.0
        )
        data = metrics.to_dict()
        
        assert data["cost_consumed"] == 10.0
        assert data["budget_remaining"] == 90.0


class TestTaskMetrics:
    """Tests for TaskMetrics value object"""
    
    def test_task_metrics_creation(self):
        """Test creating task metrics"""
        metrics = TaskMetrics(
            active_tasks_count=5,
            completed_tasks_count=10,
            failed_tasks_count=2
        )
        assert metrics.active_tasks_count == 5
        assert metrics.completed_tasks_count == 10
        assert metrics.failed_tasks_count == 2
    
    def test_task_metrics_get_total_tasks(self):
        """Test getting total tasks count"""
        metrics = TaskMetrics(
            active_tasks_count=3,
            completed_tasks_count=7,
            failed_tasks_count=1
        )
        assert metrics.get_total_tasks() == 11
    
    def test_task_metrics_get_failure_rate(self):
        """Test calculating failure rate"""
        metrics = TaskMetrics(
            active_tasks_count=5,
            completed_tasks_count=85,
            failed_tasks_count=10
        )
        # Failed / Total = 10 / 100 = 0.1
        assert metrics.get_failure_rate() == 0.1
    
    def test_task_metrics_get_failure_rate_no_tasks(self):
        """Test failure rate with no tasks"""
        metrics = TaskMetrics(
            active_tasks_count=0,
            completed_tasks_count=0,
            failed_tasks_count=0
        )
        assert metrics.get_failure_rate() == 0.0


class TestAgentHealthStatus:
    """Tests for AgentHealthStatus entity"""
    
    def test_agent_health_status_creation(self):
        """Test creating agent health status"""
        agent_id = uuid4()
        status = AgentHealthStatus.create(
            agent_id=agent_id,
            resource_metrics=ResourceMetrics(
                memory_usage_mb=512.0,
                cpu_usage_percent=50.0,
                working_memory_count=100,
                episodic_memory_count=500
            ),
            cost_metrics=CostMetrics(
                cost_consumed=10.0,
                budget_remaining=90.0
            ),
            task_metrics=TaskMetrics(
                active_tasks_count=5,
                completed_tasks_count=10,
                failed_tasks_count=1
            )
        )
        
        assert status.agent_id == agent_id
        assert status.status == HealthStatus.HEALTHY
        assert status.resource_metrics.memory_usage_mb == 512.0
    
    def test_agent_health_status_update_heartbeat(self):
        """Test updating heartbeat timestamp"""
        agent_id = uuid4()
        status = AgentHealthStatus.create(
            agent_id=agent_id,
            resource_metrics=ResourceMetrics(
                memory_usage_mb=256.0,
                cpu_usage_percent=25.0,
                working_memory_count=50,
                episodic_memory_count=200
            ),
            cost_metrics=CostMetrics(
                cost_consumed=5.0,
                budget_remaining=95.0
            ),
            task_metrics=TaskMetrics(
                active_tasks_count=2,
                completed_tasks_count=5,
                failed_tasks_count=0
            )
        )
        
        original_heartbeat = status.last_heartbeat
        status.update_heartbeat()
        
        assert status.last_heartbeat > original_heartbeat
    
    def test_agent_health_status_mark_as_degraded(self):
        """Test marking agent as degraded"""
        agent_id = uuid4()
        status = AgentHealthStatus.create(
            agent_id=agent_id,
            resource_metrics=ResourceMetrics(
                memory_usage_mb=256.0,
                cpu_usage_percent=25.0,
                working_memory_count=50,
                episodic_memory_count=200
            ),
            cost_metrics=CostMetrics(
                cost_consumed=5.0,
                budget_remaining=95.0
            ),
            task_metrics=TaskMetrics(
                active_tasks_count=2,
                completed_tasks_count=5,
                failed_tasks_count=0
            )
        )
        
        status.mark_as_degraded("High memory usage")
        
        assert status.status == HealthStatus.DEGRADED
        assert status.error_message == "High memory usage"
    
    def test_agent_health_status_mark_as_failed(self):
        """Test marking agent as failed"""
        agent_id = uuid4()
        status = AgentHealthStatus.create(
            agent_id=agent_id,
            resource_metrics=ResourceMetrics(
                memory_usage_mb=256.0,
                cpu_usage_percent=25.0,
                working_memory_count=50,
                episodic_memory_count=200
            ),
            cost_metrics=CostMetrics(
                cost_consumed=5.0,
                budget_remaining=95.0
            ),
            task_metrics=TaskMetrics(
                active_tasks_count=2,
                completed_tasks_count=5,
                failed_tasks_count=0
            )
        )
        
        status.mark_as_failed("Critical failure")
        
        assert status.status == HealthStatus.FAILED
        assert status.error_message == "Critical failure"
    
    def test_agent_health_status_calculate_health_score(self):
        """Test calculating health score"""
        agent_id = uuid4()
        status = AgentHealthStatus.create(
            agent_id=agent_id,
            resource_metrics=ResourceMetrics(
                memory_usage_mb=512.0,  # Low usage
                cpu_usage_percent=25.0,  # Low usage
                working_memory_count=100,
                episodic_memory_count=500
            ),
            cost_metrics=CostMetrics(
                cost_consumed=20.0,  # 20% used
                budget_remaining=80.0
            ),
            task_metrics=TaskMetrics(
                active_tasks_count=10,
                completed_tasks_count=90,
                failed_tasks_count=5  # 5% failure rate
            )
        )
        
        score = status.calculate_health_score()
        
        # Should be high (close to 1.0) since metrics are healthy
        assert 0.0 <= score <= 1.0
        assert score > 0.7  # Expect healthy score


class TestAgentHealthIncident:
    """Tests for AgentHealthIncident entity"""
    
    def test_incident_creation(self):
        """Test creating health incident"""
        agent_id = uuid4()
        incident = AgentHealthIncident.create(
            agent_id=agent_id,
            incident_type="heartbeat_failure",
            severity=IncidentSeverity.HIGH,
            description="Agent failed to send heartbeat for 60 seconds"
        )
        
        assert incident.agent_id == agent_id
        assert incident.incident_type == "heartbeat_failure"
        assert incident.severity == IncidentSeverity.HIGH
        assert incident.status == IncidentStatus.OPEN
    
    def test_incident_resolve(self):
        """Test resolving incident"""
        agent_id = uuid4()
        incident = AgentHealthIncident.create(
            agent_id=agent_id,
            incident_type="memory_overflow",
            severity=IncidentSeverity.MEDIUM,
            description="Memory usage exceeded threshold"
        )
        
        incident.resolve("Memory cleared successfully")
        
        assert incident.status == IncidentStatus.RESOLVED
        assert incident.resolution_notes == "Memory cleared successfully"
        assert incident.resolved_at is not None
    
    def test_incident_mark_investigating(self):
        """Test marking incident as investigating"""
        agent_id = uuid4()
        incident = AgentHealthIncident.create(
            agent_id=agent_id,
            incident_type="task_failure",
            severity=IncidentSeverity.CRITICAL,
            description="Multiple tasks failed"
        )
        
        incident.mark_investigating()
        
        assert incident.status == IncidentStatus.INVESTIGATING
    
    def test_incident_ignore(self):
        """Test ignoring incident"""
        agent_id = uuid4()
        incident = AgentHealthIncident.create(
            agent_id=agent_id,
            incident_type="minor_issue",
            severity=IncidentSeverity.LOW,
            description="Temporary spike in CPU"
        )
        
        incident.ignore("False positive")
        
        assert incident.status == IncidentStatus.IGNORED
        assert incident.resolution_notes == "False positive"
    
    def test_incident_record_recovery_attempt(self):
        """Test recording recovery attempt"""
        agent_id = uuid4()
        incident = AgentHealthIncident.create(
            agent_id=agent_id,
            incident_type="agent_crash",
            severity=IncidentSeverity.CRITICAL,
            description="Agent crashed"
        )
        
        incident.record_recovery_attempt("restart", True)
        
        assert incident.recovery_action == "restart"
        assert incident.recovery_successful is True
    
    def test_incident_is_critical(self):
        """Test checking if incident is critical"""
        agent_id = uuid4()
        
        critical_incident = AgentHealthIncident.create(
            agent_id=agent_id,
            incident_type="system_failure",
            severity=IncidentSeverity.CRITICAL,
            description="System failure"
        )
        
        low_incident = AgentHealthIncident.create(
            agent_id=agent_id,
            incident_type="warning",
            severity=IncidentSeverity.LOW,
            description="Minor warning"
        )
        
        assert critical_incident.is_critical() is True
        assert low_incident.is_critical() is False
