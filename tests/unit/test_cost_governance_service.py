"""
Unit Tests for Cost Governance Domain Services

Tests for CostGovernanceService business logic and orchestration.
"""

import pytest
from unittest.mock import AsyncMock
from uuid import uuid4
from decimal import Decimal

from core.domain.cost_governance.entities import (
    BudgetStatus,
    CostEntry,
    OperationType,
    WorkflowBudget,
)
from core.domain.cost_governance.services import CostGovernanceService


class TestCostGovernanceService:
    """Tests for CostGovernanceService"""
    
    @pytest.fixture
    def mock_budget_repository(self):
        """Create mock budget repository"""
        repository = AsyncMock()
        repository.find_by_workflow_execution = AsyncMock(return_value=None)
        repository.save = AsyncMock()
        repository.update = AsyncMock()
        return repository
    
    @pytest.fixture
    def mock_cost_tracking_repository(self):
        """Create mock cost tracking repository"""
        repository = AsyncMock()
        repository.save = AsyncMock()
        repository.get_total_cost = AsyncMock(return_value=0.0)
        repository.get_cost_by_operation_type = AsyncMock(return_value={})
        repository.get_cost_by_agent = AsyncMock(return_value={})
        return repository
    
    @pytest.fixture
    def cost_service(self, mock_budget_repository, mock_cost_tracking_repository):
        """Create cost governance service instance"""
        return CostGovernanceService(
            budget_repository=mock_budget_repository,
            cost_tracking_repository=mock_cost_tracking_repository
        )
    
    @pytest.mark.asyncio
    async def test_create_budget(
        self, cost_service, mock_budget_repository
    ):
        """Test creating a new workflow budget"""
        workflow_execution_id = uuid4()
        allocated_budget = 100.0
        
        # Create budget
        budget, event = await cost_service.create_budget(
            workflow_execution_id=workflow_execution_id,
            allocated_budget=allocated_budget,
            currency="USD",
            warning_threshold=0.8,
            critical_threshold=0.95
        )
        
        # Verify budget was created
        assert isinstance(budget, WorkflowBudget)
        assert budget.workflow_execution_id == workflow_execution_id
        assert budget.allocated_budget == allocated_budget
        assert budget.consumed_budget == 0.0
        assert budget.status == BudgetStatus.ACTIVE
        
        # Verify event was created
        assert event.workflow_execution_id == workflow_execution_id
        assert event.allocated_budget == allocated_budget
        
        # Verify repository calls
        mock_budget_repository.save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_budget_raises_if_exists(
        self, cost_service, mock_budget_repository
    ):
        """Test creating budget raises error if already exists"""
        workflow_execution_id = uuid4()
        
        # Existing budget
        existing_budget = WorkflowBudget.create(
            workflow_execution_id=workflow_execution_id,
            allocated_budget=50.0
        )
        mock_budget_repository.find_by_workflow_execution.return_value = existing_budget
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Budget already exists"):
            await cost_service.create_budget(
                workflow_execution_id=workflow_execution_id,
                allocated_budget=100.0
            )
    
    @pytest.mark.asyncio
    async def test_record_cost(
        self, cost_service, mock_budget_repository, mock_cost_tracking_repository
    ):
        """Test recording a cost entry"""
        workflow_execution_id = uuid4()
        
        # Record cost
        cost_entry, event = await cost_service.record_cost(
            workflow_execution_id=workflow_execution_id,
            operation_type=OperationType.LLM_CALL,
            provider="OpenAI",
            cost=2.50,
            model="gpt-4",
            tokens_used=1000
        )
        
        # Verify cost entry was created
        assert isinstance(cost_entry, CostEntry)
        assert cost_entry.workflow_execution_id == workflow_execution_id
        assert cost_entry.operation_type == OperationType.LLM_CALL
        assert cost_entry.provider == "OpenAI"
        assert cost_entry.cost == 2.50
        assert cost_entry.model == "gpt-4"
        
        # Verify event was created
        assert event.workflow_execution_id == workflow_execution_id
        assert event.cost == 2.50
        
        # Verify repository calls
        mock_cost_tracking_repository.save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_record_cost_updates_budget(
        self, cost_service, mock_budget_repository, mock_cost_tracking_repository
    ):
        """Test recording cost updates budget if exists"""
        workflow_execution_id = uuid4()
        
        # Create budget
        budget = WorkflowBudget.create(
            workflow_execution_id=workflow_execution_id,
            allocated_budget=100.0
        )
        mock_budget_repository.find_by_workflow_execution.return_value = budget
        
        # Record cost
        cost_entry, event = await cost_service.record_cost(
            workflow_execution_id=workflow_execution_id,
            operation_type=OperationType.LLM_CALL,
            provider="OpenAI",
            cost=10.0
        )
        
        # Verify budget was updated
        mock_budget_repository.update.assert_called_once()
        
        # Verify event includes budget info
        assert event.consumed_budget == 10.0
        assert event.remaining_budget == 90.0
        assert event.usage_percentage == 10.0
    
    @pytest.mark.asyncio
    async def test_check_and_enforce_budget_warning_threshold(
        self, cost_service, mock_budget_repository
    ):
        """Test budget enforcement at warning threshold"""
        workflow_execution_id = uuid4()
        
        # Create budget at 81% usage (warning threshold is 80%)
        budget = WorkflowBudget.create(
            workflow_execution_id=workflow_execution_id,
            allocated_budget=100.0,
            warning_threshold=0.8,
            critical_threshold=0.95
        )
        budget.consume_budget(81.0)  # 81% usage
        
        mock_budget_repository.find_by_workflow_execution.return_value = budget
        
        # Check and enforce
        status, events = await cost_service.check_and_enforce_budget(
            workflow_execution_id
        )
        
        # Should be in WARNING status
        assert status == BudgetStatus.WARNING
        assert len(events) == 1
        assert "WarningThresholdReached" in events[0].__class__.__name__
    
    @pytest.mark.asyncio
    async def test_check_and_enforce_budget_critical_threshold(
        self, cost_service, mock_budget_repository
    ):
        """Test budget enforcement at critical threshold"""
        workflow_execution_id = uuid4()
        
        # Create budget at 96% usage (critical threshold is 95%)
        budget = WorkflowBudget.create(
            workflow_execution_id=workflow_execution_id,
            allocated_budget=100.0,
            warning_threshold=0.8,
            critical_threshold=0.95
        )
        budget.consume_budget(96.0)  # 96% usage
        
        mock_budget_repository.find_by_workflow_execution.return_value = budget
        
        # Check and enforce
        status, events = await cost_service.check_and_enforce_budget(
            workflow_execution_id
        )
        
        # Should be in CRITICAL status
        assert status == BudgetStatus.CRITICAL
        assert len(events) >= 1
        assert any("CriticalThresholdReached" in e.__class__.__name__ for e in events)
    
    @pytest.mark.asyncio
    async def test_check_and_enforce_budget_exhausted(
        self, cost_service, mock_budget_repository
    ):
        """Test budget enforcement when exhausted"""
        workflow_execution_id = uuid4()
        
        # Create budget at 100% usage
        budget = WorkflowBudget.create(
            workflow_execution_id=workflow_execution_id,
            allocated_budget=100.0
        )
        budget.consume_budget(100.0)  # 100% usage
        
        mock_budget_repository.find_by_workflow_execution.return_value = budget
        
        # Check and enforce
        status, events = await cost_service.check_and_enforce_budget(
            workflow_execution_id
        )
        
        # Should be EXHAUSTED
        assert status == BudgetStatus.EXHAUSTED
        assert len(events) >= 1
        assert any("BudgetExhausted" in e.__class__.__name__ for e in events)
    
    @pytest.mark.asyncio
    async def test_suspend_budget(
        self, cost_service, mock_budget_repository
    ):
        """Test suspending a budget"""
        workflow_execution_id = uuid4()
        
        # Create active budget
        budget = WorkflowBudget.create(
            workflow_execution_id=workflow_execution_id,
            allocated_budget=100.0
        )
        
        mock_budget_repository.find_by_workflow_execution.return_value = budget
        
        # Suspend budget
        suspended_budget, event = await cost_service.suspend_budget(
            workflow_execution_id=workflow_execution_id,
            reason="Manual suspension for review"
        )
        
        # Verify budget was suspended
        assert suspended_budget.status == BudgetStatus.SUSPENDED
        
        # Verify event was created
        assert "BudgetSuspended" in event.__class__.__name__
        assert event.suspension_reason == "Manual suspension for review"
        
        # Verify repository calls
        mock_budget_repository.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_cost_summary_with_budget(
        self, cost_service, mock_budget_repository, mock_cost_tracking_repository
    ):
        """Test getting cost summary with budget"""
        workflow_execution_id = uuid4()
        
        # Create budget with some consumption
        budget = WorkflowBudget.create(
            workflow_execution_id=workflow_execution_id,
            allocated_budget=100.0
        )
        budget.consume_budget(45.0)
        
        mock_budget_repository.find_by_workflow_execution.return_value = budget
        mock_cost_tracking_repository.get_total_cost.return_value = 45.0
        
        # Get summary
        summary = await cost_service.get_cost_summary(workflow_execution_id)
        
        # Verify summary contents
        assert summary["total_cost"] == 45.0
        assert summary["budget_exists"] is True
        assert summary["allocated_budget"] == 100.0
        assert summary["consumed_budget"] == 45.0
        assert summary["remaining_budget"] == 55.0
        assert summary["usage_percentage"] == 45.0
        assert summary["status"] == BudgetStatus.ACTIVE.value
        assert summary["is_exhausted"] is False
        assert summary["is_suspended"] is False
    
    @pytest.mark.asyncio
    async def test_get_cost_summary_without_budget(
        self, cost_service, mock_budget_repository, mock_cost_tracking_repository
    ):
        """Test getting cost summary without budget"""
        workflow_execution_id = uuid4()
        
        # No budget exists
        mock_budget_repository.find_by_workflow_execution.return_value = None
        mock_cost_tracking_repository.get_total_cost.return_value = 25.0
        
        # Get summary
        summary = await cost_service.get_cost_summary(workflow_execution_id)
        
        # Verify summary contents
        assert summary["total_cost"] == 25.0
        assert summary["budget_exists"] is False
        assert "allocated_budget" not in summary
    
    @pytest.mark.asyncio
    async def test_calculate_projected_cost(
        self, cost_service, mock_cost_tracking_repository
    ):
        """Test calculating projected total cost"""
        workflow_execution_id = uuid4()
        
        # Current cost is $30 at 30% completion
        mock_cost_tracking_repository.get_total_cost.return_value = 30.0
        
        # Calculate projection
        projected = await cost_service.calculate_projected_cost(
            workflow_execution_id=workflow_execution_id,
            completion_percentage=30.0
        )
        
        # Should project to $100 total
        assert projected == pytest.approx(100.0, rel=0.01)
    
    @pytest.mark.asyncio
    async def test_calculate_projected_cost_invalid_percentage(
        self, cost_service
    ):
        """Test projected cost raises error for invalid percentage"""
        workflow_execution_id = uuid4()
        
        # Should raise ValueError for invalid percentages
        with pytest.raises(ValueError, match="Completion percentage"):
            await cost_service.calculate_projected_cost(
                workflow_execution_id=workflow_execution_id,
                completion_percentage=0.0
            )
        
        with pytest.raises(ValueError, match="Completion percentage"):
            await cost_service.calculate_projected_cost(
                workflow_execution_id=workflow_execution_id,
                completion_percentage=150.0
            )
    
    @pytest.mark.asyncio
    async def test_get_cost_breakdown_by_operation_type(
        self, cost_service, mock_cost_tracking_repository
    ):
        """Test getting cost breakdown by operation type"""
        workflow_execution_id = uuid4()
        
        # Mock breakdown
        mock_cost_tracking_repository.get_cost_by_operation_type.return_value = {
            OperationType.LLM_CALL: 45.0,
            OperationType.AGENT_EXECUTION: 30.0,
            OperationType.STORAGE: 5.0
        }
        
        # Get breakdown
        breakdown = await cost_service.get_cost_breakdown_by_operation_type(
            workflow_execution_id
        )
        
        # Verify breakdown (keys should be strings)
        assert isinstance(breakdown, dict)
        assert breakdown[OperationType.LLM_CALL.value] == 45.0
        assert breakdown[OperationType.AGENT_EXECUTION.value] == 30.0
        assert breakdown[OperationType.STORAGE.value] == 5.0
    
    @pytest.mark.asyncio
    async def test_get_cost_breakdown_by_agent(
        self, cost_service, mock_cost_tracking_repository
    ):
        """Test getting cost breakdown by agent"""
        workflow_execution_id = uuid4()
        agent_id_1 = uuid4()
        agent_id_2 = uuid4()
        
        # Mock breakdown
        mock_cost_tracking_repository.get_cost_by_agent.return_value = {
            agent_id_1: 60.0,
            agent_id_2: 40.0
        }
        
        # Get breakdown
        breakdown = await cost_service.get_cost_breakdown_by_agent(
            workflow_execution_id
        )
        
        # Verify breakdown (keys should be strings)
        assert isinstance(breakdown, dict)
        assert breakdown[str(agent_id_1)] == 60.0
        assert breakdown[str(agent_id_2)] == 40.0
    
    @pytest.mark.asyncio
    async def test_get_budget_status_exists(
        self, cost_service, mock_budget_repository
    ):
        """Test getting budget status when budget exists"""
        workflow_execution_id = uuid4()
        
        # Create budget
        budget = WorkflowBudget.create(
            workflow_execution_id=workflow_execution_id,
            allocated_budget=100.0
        )
        budget.consume_budget(85.0)  # Put in WARNING status
        budget.check_thresholds()
        
        mock_budget_repository.find_by_workflow_execution.return_value = budget
        
        # Get status
        status = await cost_service.get_budget_status(workflow_execution_id)
        
        # Verify status
        assert status == BudgetStatus.WARNING
    
    @pytest.mark.asyncio
    async def test_get_budget_status_not_exists(
        self, cost_service, mock_budget_repository
    ):
        """Test getting budget status when budget doesn't exist"""
        workflow_execution_id = uuid4()
        
        # No budget
        mock_budget_repository.find_by_workflow_execution.return_value = None
        
        # Get status
        status = await cost_service.get_budget_status(workflow_execution_id)
        
        # Should return None
        assert status is None
