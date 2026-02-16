"""
Unit Tests for Health Monitoring Domain Services

Tests for AgentHealthMonitoringService business logic and orchestration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from datetime import datetime, timedelta

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
from core.domain.health_monitoring.services import AgentHealthMonitoringService


class TestAgentHealthMonitoringService:
    """Tests for AgentHealthMonitoringService"""
    
    @pytest.fixture
    def mock_health_repository(self):
        """Create mock health repository"""
        repository = AsyncMock()
        repository.find_by_agent = AsyncMock(return_value=None)
        repository.save = AsyncMock()
        repository.find_stale_heartbeats = AsyncMock(return_value=[])
        repository.find_by_workflow_execution = AsyncMock(return_value=[])
        return repository
    
    @pytest.fixture
    def mock_incident_repository(self):
        """Create mock incident repository"""
        repository = AsyncMock()
        repository.save = AsyncMock()
        repository.find_by_id = AsyncMock()
        repository.find_open_incidents = AsyncMock(return_value=[])
        return repository
    
    @pytest.fixture
    def health_service(self, mock_health_repository, mock_incident_repository):
        """Create health monitoring service instance"""
        return AgentHealthMonitoringService(
            health_repository=mock_health_repository,
            incident_repository=mock_incident_repository
        )
    
    @pytest.fixture
    def sample_resource_metrics(self):
        """Create sample resource metrics"""
        return ResourceMetrics(
            memory_usage_mb=512.0,
            cpu_usage_percent=45.0,
            disk_usage_percent=60.0,
            network_rx_mb=100.0,
            network_tx_mb=50.0
        )
    
    @pytest.fixture
    def sample_cost_metrics(self):
        """Create sample cost metrics"""
        return CostMetrics(
            cost_consumed=5.50,
            budget_remaining=14.50
        )
    
    @pytest.fixture
    def sample_task_metrics(self):
        """Create sample task metrics"""
        return TaskMetrics(
            active_tasks_count=3,
            completed_tasks_count=45,
            failed_tasks_count=5,
            avg_task_duration_seconds=12.5
        )
    
    @pytest.mark.asyncio
    async def test_record_heartbeat_creates_new_health_status(
        self, health_service, mock_health_repository,
        sample_resource_metrics, sample_cost_metrics, sample_task_metrics
    ):
        """Test recording heartbeat creates new health status for new agent"""
        agent_id = uuid4()
        workflow_execution_id = uuid4()
        
        # Repository returns None (agent not found)
        mock_health_repository.find_by_agent.return_value = None
        
        # Record heartbeat
        result = await health_service.record_heartbeat(
            agent_id=agent_id,
            workflow_execution_id=workflow_execution_id,
            resource_metrics=sample_resource_metrics,
            cost_metrics=sample_cost_metrics,
            task_metrics=sample_task_metrics
        )
        
        # Verify health status was created
        assert isinstance(result, AgentHealthStatus)
        assert result.agent_id == agent_id
        assert result.workflow_execution_id == workflow_execution_id
        assert result.status == HealthStatus.HEALTHY
        assert 0.0 <= result.health_score <= 1.0
        
        # Verify save was called
        mock_health_repository.save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_record_heartbeat_updates_existing_health_status(
        self, health_service, mock_health_repository,
        sample_resource_metrics, sample_cost_metrics, sample_task_metrics
    ):
        """Test recording heartbeat updates existing health status"""
        agent_id = uuid4()
        workflow_execution_id = uuid4()
        
        # Create existing health status
        existing_health = AgentHealthStatus.create(
            agent_id=agent_id,
            workflow_execution_id=workflow_execution_id,
            resource_metrics=sample_resource_metrics,
            cost_metrics=sample_cost_metrics,
            task_metrics=sample_task_metrics,
            health_score=0.85
        )
        mock_health_repository.find_by_agent.return_value = existing_health
        
        # Update metrics (worse performance)
        degraded_resource = ResourceMetrics(
            memory_usage_mb=1800.0,  # High memory
            cpu_usage_percent=95.0,   # High CPU
            disk_usage_percent=85.0,
            network_rx_mb=200.0,
            network_tx_mb=100.0
        )
        
        # Record heartbeat with degraded metrics
        result = await health_service.record_heartbeat(
            agent_id=agent_id,
            workflow_execution_id=workflow_execution_id,
            resource_metrics=degraded_resource,
            cost_metrics=sample_cost_metrics,
            task_metrics=sample_task_metrics
        )
        
        # Verify status was updated
        assert result.agent_id == agent_id
        mock_health_repository.save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_calculate_health_score_high_performance(
        self, health_service,
        sample_resource_metrics, sample_cost_metrics, sample_task_metrics
    ):
        """Test health score calculation with good metrics"""
        score = await health_service.calculate_health_score(
            resource_metrics=sample_resource_metrics,
            cost_metrics=sample_cost_metrics,
            task_metrics=sample_task_metrics
        )
        
        # Should be high score for good metrics
        assert 0.7 <= score <= 1.0
        assert isinstance(score, float)
    
    @pytest.mark.asyncio
    async def test_calculate_health_score_poor_performance(self, health_service):
        """Test health score calculation with poor metrics"""
        poor_resource = ResourceMetrics(
            memory_usage_mb=2000.0,   # Very high
            cpu_usage_percent=98.0,   # Very high
            disk_usage_percent=95.0,
            network_rx_mb=500.0,
            network_tx_mb=300.0
        )
        poor_cost = CostMetrics(
            cost_consumed=19.0,
            budget_remaining=1.0  # Almost exhausted
        )
        poor_tasks = TaskMetrics(
            active_tasks_count=5,
            completed_tasks_count=10,
            failed_tasks_count=40,  # High failure rate
            avg_task_duration_seconds=60.0
        )
        
        score = await health_service.calculate_health_score(
            resource_metrics=poor_resource,
            cost_metrics=poor_cost,
            task_metrics=poor_tasks
        )
        
        # Should be low score for poor metrics
        assert 0.0 <= score < 0.5
    
    @pytest.mark.asyncio
    async def test_detect_failures_identifies_stale_heartbeats(
        self, health_service, mock_health_repository, mock_incident_repository
    ):
        """Test failure detection identifies agents with stale heartbeats"""
        agent_id = uuid4()
        workflow_execution_id = uuid4()
        
        # Create stale health status (old heartbeat)
        stale_health = AgentHealthStatus.create(
            agent_id=agent_id,
            workflow_execution_id=workflow_execution_id,
            resource_metrics=ResourceMetrics(
                memory_usage_mb=100.0, cpu_usage_percent=10.0,
                disk_usage_percent=20.0, network_rx_mb=10.0, network_tx_mb=5.0
            ),
            cost_metrics=CostMetrics(cost_consumed=1.0, budget_remaining=19.0),
            task_metrics=TaskMetrics(
                active_tasks_count=0, completed_tasks_count=10,
                failed_tasks_count=0, avg_task_duration_seconds=5.0
            ),
            health_score=0.9
        )
        stale_health.last_heartbeat = datetime.utcnow() - timedelta(seconds=60)
        
        mock_health_repository.find_stale_heartbeats.return_value = [stale_health]
        
        # Detect failures
        failed_agents = await health_service.detect_failures(threshold_seconds=30)
        
        # Verify failure was detected
        assert len(failed_agents) == 1
        assert failed_agents[0].agent_id == agent_id
        assert failed_agents[0].status == HealthStatus.FAILED
        assert failed_agents[0].health_score == 0.0
        
        # Verify incident was created
        mock_incident_repository.save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_detect_failures_skips_already_failed_agents(
        self, health_service, mock_health_repository
    ):
        """Test failure detection skips agents already marked as failed"""
        agent_id = uuid4()
        workflow_execution_id = uuid4()
        
        # Create already-failed health status
        failed_health = AgentHealthStatus.create(
            agent_id=agent_id,
            workflow_execution_id=workflow_execution_id,
            resource_metrics=ResourceMetrics(
                memory_usage_mb=100.0, cpu_usage_percent=10.0,
                disk_usage_percent=20.0, network_rx_mb=10.0, network_tx_mb=5.0
            ),
            cost_metrics=CostMetrics(cost_consumed=1.0, budget_remaining=19.0),
            task_metrics=TaskMetrics(
                active_tasks_count=0, completed_tasks_count=10,
                failed_tasks_count=0, avg_task_duration_seconds=5.0
            ),
            health_score=0.0
        )
        failed_health.mark_as_failed("Already failed")
        
        mock_health_repository.find_stale_heartbeats.return_value = [failed_health]
        
        # Detect failures
        failed_agents = await health_service.detect_failures(threshold_seconds=30)
        
        # Should not re-process already failed agents
        assert len(failed_agents) == 0
    
    @pytest.mark.asyncio
    async def test_create_incident(
        self, health_service, mock_incident_repository
    ):
        """Test creating health incident"""
        agent_id = uuid4()
        workflow_execution_id = uuid4()
        
        incident = await health_service.create_incident(
            agent_id=agent_id,
            workflow_execution_id=workflow_execution_id,
            severity=IncidentSeverity.CRITICAL,
            title="Test Incident",
            description="Test incident description",
            error_message="Test error",
            health_score=0.3,
            failure_rate=0.2
        )
        
        assert isinstance(incident, AgentHealthIncident)
        assert incident.agent_id == agent_id
        assert incident.severity == IncidentSeverity.CRITICAL
        assert incident.title == "Test Incident"
        assert incident.status == IncidentStatus.OPEN
        
        mock_incident_repository.save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trigger_recovery_for_failed_agent(
        self, health_service, mock_health_repository
    ):
        """Test triggering recovery for failed agent"""
        agent_id = uuid4()
        workflow_execution_id = uuid4()
        
        # Create failed health status
        failed_health = AgentHealthStatus.create(
            agent_id=agent_id,
            workflow_execution_id=workflow_execution_id,
            resource_metrics=ResourceMetrics(
                memory_usage_mb=100.0, cpu_usage_percent=10.0,
                disk_usage_percent=20.0, network_rx_mb=10.0, network_tx_mb=5.0
            ),
            cost_metrics=CostMetrics(cost_consumed=1.0, budget_remaining=19.0),
            task_metrics=TaskMetrics(
                active_tasks_count=0, completed_tasks_count=10,
                failed_tasks_count=0, avg_task_duration_seconds=5.0
            ),
            health_score=0.0
        )
        failed_health.mark_as_failed("Test failure")
        
        mock_health_repository.find_by_agent.return_value = failed_health
        
        # Trigger recovery
        result = await health_service.trigger_recovery(agent_id=agent_id)
        
        assert result.status == HealthStatus.RECOVERING
        assert result.recovery_attempts == 1
        mock_health_repository.save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trigger_recovery_raises_for_healthy_agent(
        self, health_service, mock_health_repository
    ):
        """Test triggering recovery raises error for healthy agent"""
        agent_id = uuid4()
        workflow_execution_id = uuid4()
        
        # Create healthy status
        healthy_status = AgentHealthStatus.create(
            agent_id=agent_id,
            workflow_execution_id=workflow_execution_id,
            resource_metrics=ResourceMetrics(
                memory_usage_mb=100.0, cpu_usage_percent=10.0,
                disk_usage_percent=20.0, network_rx_mb=10.0, network_tx_mb=5.0
            ),
            cost_metrics=CostMetrics(cost_consumed=1.0, budget_remaining=19.0),
            task_metrics=TaskMetrics(
                active_tasks_count=5, completed_tasks_count=50,
                failed_tasks_count=0, avg_task_duration_seconds=5.0
            ),
            health_score=0.95
        )
        
        mock_health_repository.find_by_agent.return_value = healthy_status
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Cannot trigger recovery"):
            await health_service.trigger_recovery(agent_id=agent_id)
    
    @pytest.mark.asyncio
    async def test_resolve_incident(
        self, health_service, mock_incident_repository
    ):
        """Test resolving health incident"""
        incident_id = uuid4()
        
        # Create open incident
        incident = AgentHealthIncident.create(
            agent_id=uuid4(),
            workflow_execution_id=uuid4(),
            severity=IncidentSeverity.WARNING,
            title="Test Incident",
            description="Test description",
            health_score=0.6,
            failure_rate=0.1
        )
        
        mock_incident_repository.find_by_id.return_value = incident
        
        # Resolve incident
        result = await health_service.resolve_incident(
            incident_id=incident_id,
            resolution="Issue resolved through manual intervention"
        )
        
        assert result.status == IncidentStatus.RESOLVED
        assert result.resolution == "Issue resolved through manual intervention"
        assert result.resolved_at is not None
        
        mock_incident_repository.save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_agent_health_summary(
        self, health_service, mock_health_repository, mock_incident_repository
    ):
        """Test getting health summary for workflow execution"""
        workflow_execution_id = uuid4()
        
        # Create mix of health statuses
        healthy = AgentHealthStatus.create(
            agent_id=uuid4(), workflow_execution_id=workflow_execution_id,
            resource_metrics=ResourceMetrics(100.0, 10.0, 20.0, 10.0, 5.0),
            cost_metrics=CostMetrics(1.0, 19.0),
            task_metrics=TaskMetrics(5, 50, 0, 5.0),
            health_score=0.95
        )
        
        degraded = AgentHealthStatus.create(
            agent_id=uuid4(), workflow_execution_id=workflow_execution_id,
            resource_metrics=ResourceMetrics(500.0, 60.0, 70.0, 50.0, 25.0),
            cost_metrics=CostMetrics(10.0, 10.0),
            task_metrics=TaskMetrics(3, 30, 5, 10.0),
            health_score=0.65
        )
        degraded.mark_as_degraded("High resource usage")
        
        failed = AgentHealthStatus.create(
            agent_id=uuid4(), workflow_execution_id=workflow_execution_id,
            resource_metrics=ResourceMetrics(100.0, 10.0, 20.0, 10.0, 5.0),
            cost_metrics=CostMetrics(1.0, 19.0),
            task_metrics=TaskMetrics(0, 10, 10, 5.0),
            health_score=0.0
        )
        failed.mark_as_failed("Heartbeat timeout")
        
        mock_health_repository.find_by_workflow_execution.return_value = [
            healthy, degraded, failed
        ]
        
        # Get summary
        summary = await health_service.get_agent_health_summary(workflow_execution_id)
        
        assert summary["total_agents"] == 3
        assert summary["healthy_count"] == 1
        assert summary["degraded_count"] == 1
        assert summary["failed_count"] == 1
        assert summary["recovering_count"] == 0
        assert 0.0 <= summary["average_health_score"] <= 1.0
        assert summary["health_percentage"] == pytest.approx(33.33, rel=0.1)
    
    def test_validate_health_status_invariants_valid(
        self, health_service
    ):
        """Test validation passes for valid health status"""
        health_status = AgentHealthStatus.create(
            agent_id=uuid4(),
            workflow_execution_id=uuid4(),
            resource_metrics=ResourceMetrics(100.0, 50.0, 60.0, 10.0, 5.0),
            cost_metrics=CostMetrics(5.0, 15.0),
            task_metrics=TaskMetrics(3, 45, 5, 10.0),
            health_score=0.85
        )
        
        errors = health_service.validate_health_status_invariants(health_status)
        
        assert len(errors) == 0
    
    def test_validate_health_status_invariants_invalid_health_score(
        self, health_service
    ):
        """Test validation fails for invalid health score"""
        health_status = AgentHealthStatus.create(
            agent_id=uuid4(),
            workflow_execution_id=uuid4(),
            resource_metrics=ResourceMetrics(100.0, 50.0, 60.0, 10.0, 5.0),
            cost_metrics=CostMetrics(5.0, 15.0),
            task_metrics=TaskMetrics(3, 45, 5, 10.0),
            health_score=1.5  # Invalid
        )
        
        errors = health_service.validate_health_status_invariants(health_status)
        
        assert len(errors) > 0
        assert any("Health score" in error for error in errors)
    
    def test_validate_health_status_invariants_invalid_metrics(
        self, health_service
    ):
        """Test validation fails for invalid metrics"""
        health_status = AgentHealthStatus.create(
            agent_id=uuid4(),
            workflow_execution_id=uuid4(),
            resource_metrics=ResourceMetrics(-100.0, 150.0, 60.0, 10.0, 5.0),  # Invalid
            cost_metrics=CostMetrics(-5.0, 15.0),  # Invalid
            task_metrics=TaskMetrics(-3, 45, 5, 10.0),  # Invalid
            health_score=0.85
        )
        
        errors = health_service.validate_health_status_invariants(health_status)
        
        assert len(errors) > 0
        assert any("Memory usage" in error or "CPU usage" in error for error in errors)
