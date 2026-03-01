"""
Integration tests for collaborative multi-agent system.

Tests the integration between:
- Autonomous Agent Orchestrator
- Multi-Agent Coordinator
- Collaborative Agent Orchestrator
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from uuid import uuid4
from datetime import datetime

from core.application.collaborative_agent_orchestrator import (
    CollaborativeAgentOrchestrator,
)
from core.application.autonomous_agent_orchestrator import (
    AutonomousAgentOrchestrator,
    AgentContext,
    ExecutionPlan,
    PlanStep,
    PlanningStrategy,
    ExecutionMode,
)
from infrastructure.multi_agent.coordinator import (
    MultiAgentCoordinator,
    AgentRole,
    AgentStatus,
    ConsensusAlgorithm,
    DelegationStrategy,
)
from core.domain.agent.entities import BudgetLimits


class TestCollaborativeAgentOrchestrator:
    """Test collaborative agent orchestration."""

    @pytest.fixture
    def mock_memory_service(self):
        """Create mock memory service."""
        mock = AsyncMock()
        mock.retrieve.return_value = []
        mock.store.return_value = {"id": str(uuid4())}
        return mock

    @pytest.fixture
    def mock_learning_service(self):
        """Create mock learning service."""
        mock = AsyncMock()
        mock.record_collaboration.return_value = {"status": "recorded"}
        return mock

    @pytest.fixture
    def mock_planning_agent(self):
        """Create mock planning agent."""
        mock = AsyncMock()
        mock.plan.return_value = {
            "goal": "test goal",
            "steps": [
                {
                    "action": "analyze data",
                    "reasoning": "need to understand data first",
                    "inputs": {"data": "test.csv"},
                    "expected_output": "data analysis",
                    "validation_criteria": ["data loaded", "analysis complete"],
                    "dependencies": [],
                    "confidence": 0.75,
                }
            ],
            "strategy": "sequential",
            "confidence": 0.8,
        }
        return mock

    @pytest.fixture
    def mock_execution_agent(self):
        """Create mock execution agent."""
        mock = AsyncMock()
        mock.execute.return_value = {
            "output": "execution result",
            "cost": 0.01,
            "tokens": 100,
        }
        return mock

    @pytest.fixture
    def mock_validation_agent(self):
        """Create mock validation agent."""
        mock = AsyncMock()
        mock.validate.return_value = {
            "confidence": 0.9,
            "constraints_met": 2,
            "hallucination_detected": False,
        }
        return mock

    @pytest.fixture
    def mock_tool_registry(self):
        """Create mock tool registry."""
        return Mock()

    @pytest.fixture
    def autonomous_orchestrator(
        self,
        mock_planning_agent,
        mock_execution_agent,
        mock_validation_agent,
        mock_memory_service,
        mock_tool_registry,
    ):
        """Create autonomous orchestrator."""
        return AutonomousAgentOrchestrator(
            planning_agent=mock_planning_agent,
            execution_agent=mock_execution_agent,
            validation_agent=mock_validation_agent,
            memory_service=mock_memory_service,
            tool_registry=mock_tool_registry,
        )

    @pytest.fixture
    def multi_agent_coordinator(self):
        """Create multi-agent coordinator."""
        return MultiAgentCoordinator()

    @pytest.fixture
    def collaborative_orchestrator(
        self,
        autonomous_orchestrator,
        multi_agent_coordinator,
        mock_memory_service,
        mock_learning_service,
    ):
        """Create collaborative orchestrator."""
        return CollaborativeAgentOrchestrator(
            autonomous_orchestrator=autonomous_orchestrator,
            multi_agent_coordinator=multi_agent_coordinator,
            memory_service=mock_memory_service,
            learning_service=mock_learning_service,
        )

    @pytest.mark.asyncio
    async def test_collaborative_execution_basic(self, collaborative_orchestrator):
        """Test basic collaborative execution."""
        result = await collaborative_orchestrator.execute_goal_collaboratively(
            goal="Analyze sales data and create report",
            constraints=["Use only secure data sources"],
            mode=ExecutionMode.DETERMINISTIC,
            enable_consensus=False,
            enable_sub_agents=False,
            max_iterations=1,
        )

        assert result["status"] == "success"
        assert "collaboration" in result
        assert "session_id" in result["collaboration"]
        assert "main_agent_id" in result["collaboration"]

    @pytest.mark.asyncio
    async def test_collaborative_execution_with_sub_agents(
        self, collaborative_orchestrator
    ):
        """Test execution that spawns sub-agents."""
        result = await collaborative_orchestrator.execute_goal_collaboratively(
            goal="Complex multi-step data processing pipeline",
            constraints=[],
            mode=ExecutionMode.DETERMINISTIC,
            enable_consensus=False,
            enable_sub_agents=True,
            max_iterations=1,
        )

        assert result["status"] == "success"
        collaboration = result["collaboration"]

        # Should have spawned at least one sub-agent for complex task
        assert "sub_agents_spawned" in collaboration
        assert "messages_exchanged" in collaboration

        # Check that sub-agent was properly tracked
        if collaboration["sub_agents_spawned"] > 0:
            assert collaboration["messages_exchanged"] >= 2  # At least assign + result

    @pytest.mark.asyncio
    async def test_collaborative_execution_with_consensus(
        self, collaborative_orchestrator, mock_validation_agent
    ):
        """Test execution using consensus validation."""
        # Set low confidence to trigger consensus
        mock_validation_agent.validate.return_value = {
            "confidence": 0.7,  # Low confidence triggers consensus
            "constraints_met": 1,
            "hallucination_detected": False,
        }

        result = await collaborative_orchestrator.execute_goal_collaboratively(
            goal="Critical decision requiring validation",
            constraints=["Must be 100% accurate"],
            mode=ExecutionMode.CONSERVATIVE,
            enable_consensus=True,
            enable_sub_agents=False,
            max_iterations=1,
        )

        assert result["status"] == "success"
        collaboration = result["collaboration"]

        # Should have run consensus rounds for low confidence steps
        assert "consensus_rounds" in collaboration

    @pytest.mark.asyncio
    async def test_agent_to_agent_messaging(
        self, collaborative_orchestrator, multi_agent_coordinator
    ):
        """Test inter-agent messaging during execution."""
        # Register two agents
        agent1_id = multi_agent_coordinator.register_agent(
            name="Agent 1",
            role=AgentRole.EXECUTOR,
            capabilities=[
                {
                    "name": "data_processing",
                    "description": "Process data",
                    "input_schema": {},
                    "output_schema": {},
                    "avg_latency_ms": 100.0,
                    "success_rate": 0.9,
                    "cost_per_call": 0.01,
                }
            ],
        )

        agent2_id = multi_agent_coordinator.register_agent(
            name="Agent 2",
            role=AgentRole.VALIDATOR,
            capabilities=[
                {
                    "name": "validation",
                    "description": "Validate results",
                    "input_schema": {},
                    "output_schema": {},
                    "avg_latency_ms": 50.0,
                    "success_rate": 0.95,
                    "cost_per_call": 0.005,
                }
            ],
        )

        # Delegate task from agent1 to agent2
        task_result = await multi_agent_coordinator.delegate_task(
            description="Validate data processing result",
            required_capabilities=["validation"],
            payload={"data": "processed_data"},
            delegated_by=agent1_id,
            strategy=DelegationStrategy.CAPABILITY_MATCH,
        )

        assert task_result["success"] is True
        assert task_result["assigned_to"] == agent2_id

        # Check message bus
        bus_stats = multi_agent_coordinator.message_bus.get_bus_stats()
        assert bus_stats["registered_agents"] >= 2

    @pytest.mark.asyncio
    async def test_consensus_decision_making(self, multi_agent_coordinator):
        """Test consensus-based decision making."""
        # Register multiple agents
        agent_ids = []
        for i in range(5):
            agent_id = multi_agent_coordinator.register_agent(
                name=f"Validator {i}",
                role=AgentRole.VALIDATOR,
                capabilities=[{"name": "validation"}],
            )
            agent_ids.append(agent_id)

        # Run consensus round
        proposal = {
            "question": "Is this result correct?",
            "result": {"output": "test result"},
        }

        consensus = await multi_agent_coordinator.run_consensus(
            proposal=proposal,
            proposer_id=agent_ids[0],
            algorithm=ConsensusAlgorithm.MAJORITY_VOTE,
            auto_vote=True,
        )

        assert "result" in consensus
        assert consensus["status"] == "resolved"
        assert "accepted" in consensus["result"]

    @pytest.mark.asyncio
    async def test_learning_from_collaboration(
        self, collaborative_orchestrator, mock_learning_service
    ):
        """Test that collaboration patterns are learned."""
        result = await collaborative_orchestrator.execute_goal_collaboratively(
            goal="Test learning integration",
            constraints=[],
            mode=ExecutionMode.DETERMINISTIC,
            enable_consensus=False,
            enable_sub_agents=True,
            max_iterations=1,
        )

        assert result["status"] == "success"

        # Verify learning service was called
        assert mock_learning_service.record_collaboration.called
        call_args = mock_learning_service.record_collaboration.call_args
        learning_data = call_args[0][0]

        assert "session_id" in learning_data
        assert "success" in learning_data
        assert "goal" in learning_data
        assert "sub_agents_used" in learning_data

    @pytest.mark.asyncio
    async def test_collaboration_metrics(self, collaborative_orchestrator):
        """Test collaboration metrics tracking."""
        # Execute multiple collaborations
        for i in range(3):
            await collaborative_orchestrator.execute_goal_collaboratively(
                goal=f"Test goal {i}",
                constraints=[],
                mode=ExecutionMode.DETERMINISTIC,
                enable_consensus=False,
                enable_sub_agents=False,
                max_iterations=1,
            )

        metrics = collaborative_orchestrator.get_collaboration_metrics()

        assert metrics["total_collaborations"] >= 3
        assert metrics["successful_collaborations"] >= 0
        assert "success_rate" in metrics
        assert "avg_collaboration_time" in metrics
        assert "coordinator_health" in metrics

    @pytest.mark.asyncio
    async def test_complex_multi_agent_workflow(
        self, collaborative_orchestrator, mock_planning_agent
    ):
        """Test complex workflow with multiple agents collaborating."""
        # Set up complex plan with multiple steps
        mock_planning_agent.plan.return_value = {
            "goal": "complex workflow",
            "steps": [
                {
                    "action": "fetch data from API",
                    "reasoning": "need data",
                    "inputs": {"api": "https://api.example.com"},
                    "expected_output": "raw data",
                    "validation_criteria": ["data retrieved"],
                    "dependencies": [],
                    "confidence": 0.9,
                },
                {
                    "action": "process and clean data",
                    "reasoning": "data needs cleaning",
                    "inputs": {"data": "raw_data"},
                    "expected_output": "clean data",
                    "validation_criteria": ["no missing values"],
                    "dependencies": ["step_1"],
                    "confidence": 0.6,  # Low confidence = sub-agent
                },
                {
                    "action": "analyze data patterns",
                    "reasoning": "find insights",
                    "inputs": {"data": "clean_data"},
                    "expected_output": "analysis results",
                    "validation_criteria": ["patterns identified"],
                    "dependencies": ["step_2"],
                    "confidence": 0.5,  # Low confidence = sub-agent
                },
            ],
            "strategy": "sequential",
            "confidence": 0.7,
        }

        result = await collaborative_orchestrator.execute_goal_collaboratively(
            goal="Complex workflow with data pipeline",
            constraints=["Must handle errors gracefully"],
            mode=ExecutionMode.ADAPTIVE,
            enable_consensus=True,
            enable_sub_agents=True,
            max_iterations=1,
        )

        assert result["status"] == "success"
        collaboration = result["collaboration"]

        # Should have spawned sub-agents for low confidence steps
        assert collaboration["sub_agents_spawned"] >= 1
        # Should have exchanged messages
        assert collaboration["messages_exchanged"] >= 1

    @pytest.mark.asyncio
    async def test_agent_load_balancing(self, multi_agent_coordinator):
        """Test that tasks are balanced across available agents."""
        # Register multiple executor agents
        for i in range(3):
            multi_agent_coordinator.register_agent(
                name=f"Executor {i}",
                role=AgentRole.EXECUTOR,
                capabilities=[{"name": "execution"}],
                max_load=2.0,
            )

        # Delegate multiple tasks
        tasks = []
        for i in range(5):
            result = await multi_agent_coordinator.delegate_task(
                description=f"Task {i}",
                required_capabilities=["execution"],
                payload={"task_id": i},
                delegated_by="system",
                strategy=DelegationStrategy.LOAD_BALANCED,
            )
            if result["success"]:
                tasks.append(result)

        # All tasks should be successfully delegated
        assert len(tasks) == 5

        # Tasks should be distributed across agents
        assigned_agents = {t["assigned_to"] for t in tasks}
        assert len(assigned_agents) >= 2  # At least 2 agents used

    @pytest.mark.asyncio
    async def test_agent_capability_matching(self, multi_agent_coordinator):
        """Test that agents are selected based on capabilities."""
        # Register specialist agents
        code_agent = multi_agent_coordinator.register_agent(
            name="Code Specialist",
            role=AgentRole.SPECIALIST,
            capabilities=[{"name": "code_execution"}],
        )

        analysis_agent = multi_agent_coordinator.register_agent(
            name="Analysis Specialist",
            role=AgentRole.SPECIALIST,
            capabilities=[{"name": "data_analysis"}],
        )

        # Delegate code execution task
        code_task = await multi_agent_coordinator.delegate_task(
            description="Execute Python code",
            required_capabilities=["code_execution"],
            payload={"code": "print('hello')"},
            delegated_by="system",
            strategy=DelegationStrategy.CAPABILITY_MATCH,
        )

        assert code_task["success"] is True
        assert code_task["assigned_to"] == code_agent

        # Delegate analysis task
        analysis_task = await multi_agent_coordinator.delegate_task(
            description="Analyze dataset",
            required_capabilities=["data_analysis"],
            payload={"dataset": "sales.csv"},
            delegated_by="system",
            strategy=DelegationStrategy.CAPABILITY_MATCH,
        )

        assert analysis_task["success"] is True
        assert analysis_task["assigned_to"] == analysis_agent

    @pytest.mark.asyncio
    async def test_session_cleanup(self, collaborative_orchestrator):
        """Test that sessions are properly cleaned up."""
        result = await collaborative_orchestrator.execute_goal_collaboratively(
            goal="Test cleanup",
            constraints=[],
            mode=ExecutionMode.DETERMINISTIC,
            enable_consensus=False,
            enable_sub_agents=False,
            max_iterations=1,
        )

        session_id = result["collaboration"]["session_id"]

        # Session should be cleaned up after execution
        assert session_id not in collaborative_orchestrator.active_sessions

        # Agents should be deregistered
        coordinator = collaborative_orchestrator.coordinator
        main_agent_id = result["collaboration"]["main_agent_id"]
        assert coordinator.registry.get_agent(main_agent_id) is None

    def test_step_complexity_assessment(self, collaborative_orchestrator):
        """Test step complexity assessment logic."""
        # Simple step
        simple_step = PlanStep(
            step_id="step1",
            action="print hello",
            reasoning="simple output",
            inputs={},
            expected_output="hello",
            validation_criteria=["output matches"],
            dependencies=[],
            confidence=0.95,
            estimated_duration=5.0,
        )

        complexity = collaborative_orchestrator._assess_step_complexity(simple_step)
        assert complexity < 0.3  # Should be low complexity

        # Complex step
        complex_step = PlanStep(
            step_id="step2",
            action="multi-stage data pipeline",
            reasoning="complex processing",
            inputs={"data1": {}, "data2": {}, "config": {}, "params": {}},
            expected_output="processed data",
            validation_criteria=["data validated", "quality checks passed"],
            dependencies=["step1", "step_a", "step_b"],
            confidence=0.4,  # Low confidence
            estimated_duration=120.0,  # Long duration
        )

        complexity = collaborative_orchestrator._assess_step_complexity(complex_step)
        assert complexity > 0.7  # Should be high complexity

    def test_capability_extraction(self, collaborative_orchestrator):
        """Test capability extraction from steps."""
        # Code execution step
        code_step = PlanStep(
            step_id="s1",
            action="Execute Python code to process data",
            reasoning="",
            inputs={},
            expected_output="",
            validation_criteria=[],
        )

        capabilities = collaborative_orchestrator._extract_capabilities_from_step(code_step)
        assert "code_execution" in capabilities

        # Web search step
        search_step = PlanStep(
            step_id="s2",
            action="Search for latest research papers",
            reasoning="",
            inputs={},
            expected_output="",
            validation_criteria=[],
        )

        capabilities = collaborative_orchestrator._extract_capabilities_from_step(
            search_step
        )
        assert "web_search" in capabilities

        # File operation step
        file_step = PlanStep(
            step_id="s3",
            action="Write results to file",
            reasoning="",
            inputs={},
            expected_output="",
            validation_criteria=[],
        )

        capabilities = collaborative_orchestrator._extract_capabilities_from_step(
            file_step
        )
        assert "file_operations" in capabilities


class TestMultiAgentCoordinator:
    """Test multi-agent coordinator independently."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator."""
        return MultiAgentCoordinator()

    def test_agent_registration(self, coordinator):
        """Test agent registration."""
        agent_id = coordinator.register_agent(
            name="Test Agent",
            role=AgentRole.EXECUTOR,
            capabilities=[{"name": "test_capability"}],
            max_load=1.0,
        )

        assert agent_id is not None
        agent = coordinator.get_agent(agent_id)
        assert agent is not None
        assert agent["name"] == "Test Agent"
        assert agent["role"] == AgentRole.EXECUTOR.value

    def test_agent_deregistration(self, coordinator):
        """Test agent deregistration."""
        agent_id = coordinator.register_agent(
            name="Test Agent",
            role=AgentRole.EXECUTOR,
            capabilities=[],
        )

        success = coordinator.deregister_agent(agent_id)
        assert success is True

        agent = coordinator.get_agent(agent_id)
        assert agent is None

    @pytest.mark.asyncio
    async def test_task_delegation(self, coordinator):
        """Test task delegation."""
        # Register agent
        agent_id = coordinator.register_agent(
            name="Worker",
            role=AgentRole.EXECUTOR,
            capabilities=[{"name": "work"}],
        )

        # Delegate task
        result = await coordinator.delegate_task(
            description="Do work",
            required_capabilities=["work"],
            payload={"data": "test"},
            delegated_by="system",
        )

        assert result["success"] is True
        assert result["assigned_to"] == agent_id

    def test_system_health(self, coordinator):
        """Test system health reporting."""
        # Register some agents
        for i in range(3):
            coordinator.register_agent(
                name=f"Agent {i}",
                role=AgentRole.EXECUTOR,
                capabilities=[],
            )

        health = coordinator.get_system_health()

        assert "registry" in health
        assert "message_bus" in health
        assert "delegation" in health
        assert health["registry"]["total_agents"] == 3
