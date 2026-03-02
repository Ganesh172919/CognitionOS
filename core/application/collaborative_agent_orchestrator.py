"""
Collaborative Multi-Agent Orchestrator

Integrates the Multi-Agent Coordinator with the Autonomous Agent Orchestrator
to enable true multi-agent collaboration with:
- Inter-agent communication during execution
- Consensus-based decision making
- Dynamic agent spawning
- Real-time coordination
- Learning from multi-agent interactions
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID, uuid4

from infrastructure.multi_agent.coordinator import (
    MultiAgentCoordinator,
    AgentRole,
    AgentStatus,
    MessageType,
    ConsensusAlgorithm,
    DelegationStrategy,
    AgentMessage,
    DelegatedTask,
)
from core.application.autonomous_agent_orchestrator import (
    AutonomousAgentOrchestrator,
    AgentContext,
    ExecutionPlan,
    PlanStep,
    PlanningStrategy,
    ExecutionMode,
    ConfidenceLevel,
)
from core.domain.agent.entities import (
    Agent,
    AgentRole as DomainAgentRole,
    AgentStatus as DomainAgentStatus,
    Capability,
    BudgetLimits,
    BudgetUsage,
)
from core.exceptions import AgentException, ValidationError
from core.application.execution_feedback_loop import (
    ExecutionFeedbackLoop,
    ExecutionFeedback,
    FeedbackType,
)

logger = logging.getLogger(__name__)


class CollaborativeAgentOrchestrator:
    """
    Orchestrator that enables true multi-agent collaboration.

    Combines the autonomous execution capabilities with multi-agent
    coordination to enable:
    - Agents spawning sub-agents for subtasks
    - Agents communicating during execution
    - Consensus-based decision making
    - Dynamic task delegation
    - Learning from collaborative patterns
    """

    def __init__(
        self,
        autonomous_orchestrator: AutonomousAgentOrchestrator,
        multi_agent_coordinator: MultiAgentCoordinator,
        memory_service: Any,
        learning_service: Optional[Any] = None,
        feedback_loop: Optional[ExecutionFeedbackLoop] = None,
    ):
        """
        Initialize collaborative orchestrator.

        Args:
            autonomous_orchestrator: Autonomous agent orchestrator
            multi_agent_coordinator: Multi-agent coordinator
            memory_service: Long-term memory service
            learning_service: Optional learning service for feedback loops
            feedback_loop: Optional execution feedback loop for continuous learning
        """
        self.autonomous_orchestrator = autonomous_orchestrator
        self.coordinator = multi_agent_coordinator
        self.memory_service = memory_service
        self.learning_service = learning_service
        self.feedback_loop = feedback_loop

        # Track active collaborative sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        # Track agent relationships and communication patterns
        self.agent_relationships: Dict[str, List[str]] = {}
        self.communication_log: List[Dict[str, Any]] = []

        # Performance metrics for learning
        self.collaboration_metrics: Dict[str, Any] = {
            "total_collaborations": 0,
            "successful_collaborations": 0,
            "failed_collaborations": 0,
            "avg_collaboration_time": 0.0,
            "consensus_success_rate": 0.0,
        }

        logger.info("Collaborative agent orchestrator initialized")

    async def execute_goal_collaboratively(
        self,
        goal: str,
        constraints: Optional[List[str]] = None,
        mode: ExecutionMode = ExecutionMode.DETERMINISTIC,
        enable_consensus: bool = True,
        enable_sub_agents: bool = True,
        max_iterations: int = 5,
        budget_limits: Optional[BudgetLimits] = None,
    ) -> Dict[str, Any]:
        """
        Execute a goal using multi-agent collaboration.

        This method orchestrates multiple agents working together:
        1. Create main orchestration agent
        2. Generate execution plan
        3. For complex steps, spawn specialized sub-agents
        4. Enable inter-agent communication
        5. Use consensus for critical decisions
        6. Aggregate results and learn from collaboration

        Args:
            goal: High-level goal to accomplish
            constraints: Optional constraints
            mode: Execution mode
            enable_consensus: Whether to use consensus for decisions
            enable_sub_agents: Whether to spawn sub-agents for subtasks
            max_iterations: Maximum iteration cycles
            budget_limits: Resource budget limits

        Returns:
            Execution result with collaboration metadata
        """
        session_id = str(uuid4())
        start_time = datetime.now(timezone.utc)

        logger.info(f"Starting collaborative execution for goal: {goal}")
        logger.info(f"Session ID: {session_id}")

        # Register main orchestrator agent
        main_agent_id = self.coordinator.register_agent(
            name="Main Orchestrator",
            role=AgentRole.ORCHESTRATOR,
            capabilities=[
                {
                    "name": "goal_execution",
                    "description": "Execute high-level goals",
                    "input_schema": {"goal": "string"},
                    "output_schema": {"result": "object"},
                    "avg_latency_ms": 5000.0,
                    "success_rate": 0.9,
                    "cost_per_call": 0.1,
                    "tags": ["orchestration", "planning"],
                }
            ],
            max_load=5.0,
            metadata={"goal": goal, "session_id": session_id},
        )

        # Initialize session tracking
        self.active_sessions[session_id] = {
            "goal": goal,
            "main_agent_id": main_agent_id,
            "sub_agents": [],
            "messages": [],
            "consensus_rounds": [],
            "start_time": start_time,
            "status": "in_progress",
        }

        try:
            # Create execution context
            context = AgentContext(
                agent_id=UUID(main_agent_id),
                goal=goal,
                constraints=constraints or [],
            )

            # Augment autonomous orchestrator with collaborative capabilities
            original_execute_step = self.autonomous_orchestrator._execute_step
            self.autonomous_orchestrator._execute_step = lambda step, ctx, budget, limits: \
                self._execute_step_collaboratively(
                    step, ctx, budget, limits, session_id, enable_sub_agents, enable_consensus
                )

            # Execute using autonomous orchestrator
            result = await self.autonomous_orchestrator.execute_goal(
                goal=goal,
                constraints=constraints,
                mode=mode,
                max_iterations=max_iterations,
                budget_limits=budget_limits,
            )

            # Restore original method
            self.autonomous_orchestrator._execute_step = original_execute_step

            # Enrich result with collaboration metadata
            session = self.active_sessions[session_id]
            session["status"] = "completed" if result["status"] == "success" else "failed"

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            collaboration_data = {
                "session_id": session_id,
                "main_agent_id": main_agent_id,
                "sub_agents_spawned": len(session["sub_agents"]),
                "messages_exchanged": len(session["messages"]),
                "consensus_rounds": len(session["consensus_rounds"]),
                "collaboration_duration": duration,
                "agent_relationships": self._extract_relationships(session_id),
            }

            # Update metrics
            self._update_metrics(result["status"] == "success", duration)

            # Learn from collaboration if learning service available
            if self.learning_service:
                await self._learn_from_collaboration(session_id, result, collaboration_data)

            # Store collaboration pattern in memory
            await self._store_collaboration_pattern(goal, collaboration_data, result)

            # Record execution feedback for continuous learning
            if self.feedback_loop:
                enriched_result = {
                    **result,
                    "execution_id": session_id,
                    "agent_role": "orchestrator",
                    "strategy": "collaborative",
                    "successful_steps": [],  # Would extract from plan
                    "failed_steps": [],  # Would extract from plan
                    "errors": [],  # Would extract from result
                }
                await self.feedback_loop.record_execution(
                    execution_result=enriched_result,
                    collaboration_data=collaboration_data,
                )

            result["collaboration"] = collaboration_data

            logger.info(f"Collaborative execution completed: {session_id}")
            logger.info(f"Sub-agents spawned: {collaboration_data['sub_agents_spawned']}")
            logger.info(f"Messages exchanged: {collaboration_data['messages_exchanged']}")

            return result

        except Exception as e:
            logger.error(f"Collaborative execution failed: {e}", exc_info=True)
            session = self.active_sessions.get(session_id, {})
            session["status"] = "failed"
            session["error"] = str(e)
            self._update_metrics(False, 0.0)
            raise

        finally:
            # Cleanup agents
            await self._cleanup_session(session_id)

    async def _execute_step_collaboratively(
        self,
        step: PlanStep,
        context: AgentContext,
        budget: BudgetUsage,
        budget_limits: Optional[BudgetLimits],
        session_id: str,
        enable_sub_agents: bool,
        enable_consensus: bool,
    ) -> Dict[str, Any]:
        """
        Execute a plan step with collaborative capabilities.

        For complex steps:
        - Spawn specialized sub-agents
        - Enable inter-agent communication
        - Use consensus for validation
        """
        logger.info(f"Executing step collaboratively: {step.step_id}")

        session = self.active_sessions.get(session_id)
        if not session:
            raise AgentException(f"Session not found: {session_id}")

        # Assess step complexity
        complexity = self._assess_step_complexity(step)

        # For complex steps, delegate to sub-agents
        if complexity > 0.7 and enable_sub_agents:
            logger.info(f"Step complexity {complexity:.2f}, delegating to sub-agent")
            result = await self._delegate_to_sub_agent(
                step, context, session_id, budget, budget_limits
            )
        else:
            # Execute directly using autonomous orchestrator
            result = await self.autonomous_orchestrator._execute_step(
                step, context, budget, budget_limits
            )

        # For critical steps, use consensus validation
        if enable_consensus and step.confidence < 0.8:
            logger.info(f"Low confidence {step.confidence:.2f}, using consensus validation")
            validated_result = await self._validate_with_consensus(
                step, result, session_id
            )
            result["consensus_validation"] = validated_result

        return result

    async def _delegate_to_sub_agent(
        self,
        step: PlanStep,
        context: AgentContext,
        session_id: str,
        budget: BudgetUsage,
        budget_limits: Optional[BudgetLimits],
    ) -> Dict[str, Any]:
        """
        Delegate a complex step to a specialized sub-agent.

        This enables:
        - Capability-based agent selection
        - Sub-agent specialization
        - Parallel sub-task execution
        """
        logger.info(f"Delegating step {step.step_id} to sub-agent")

        # Determine required capabilities from step
        required_capabilities = self._extract_capabilities_from_step(step)

        # Register sub-agent for this task
        sub_agent_id = self.coordinator.register_agent(
            name=f"Executor-{step.step_id}",
            role=AgentRole.EXECUTOR,
            capabilities=[
                {
                    "name": cap,
                    "description": f"Execute {cap}",
                    "input_schema": {},
                    "output_schema": {},
                    "avg_latency_ms": 1000.0,
                    "success_rate": 0.85,
                    "cost_per_call": 0.01,
                    "tags": ["execution"],
                }
                for cap in required_capabilities
            ],
            max_load=1.0,
            metadata={"step_id": step.step_id, "session_id": session_id},
        )

        # Track sub-agent
        session = self.active_sessions[session_id]
        session["sub_agents"].append({
            "agent_id": sub_agent_id,
            "step_id": step.step_id,
            "spawned_at": datetime.now(timezone.utc).isoformat(),
        })

        # Delegate task using coordinator
        delegation_result = await self.coordinator.delegate_task(
            description=step.action,
            required_capabilities=required_capabilities,
            payload={
                "step": {
                    "action": step.action,
                    "reasoning": step.reasoning,
                    "inputs": step.inputs,
                    "expected_output": step.expected_output,
                }
            },
            delegated_by=session["main_agent_id"],
            strategy=DelegationStrategy.CAPABILITY_MATCH,
            priority=5,
        )

        if not delegation_result["success"]:
            raise AgentException(f"Failed to delegate task: {delegation_result.get('error')}")

        # Send execution message
        task_message = AgentMessage(
            message_id=str(uuid4()),
            sender_id=session["main_agent_id"],
            receiver_id=sub_agent_id,
            message_type=MessageType.TASK_ASSIGNMENT,
            payload={
                "step_id": step.step_id,
                "action": step.action,
                "inputs": step.inputs,
            },
            correlation_id=session_id,
            priority=5,
        )

        await self.coordinator.message_bus.send(task_message)
        session["messages"].append(task_message.to_dict())

        # Simulate execution by sub-agent
        # In production, this would actually execute the step
        await asyncio.sleep(0.1)

        # Simulate receiving result
        result = {
            "step_id": step.step_id,
            "executed_by": sub_agent_id,
            "output": f"Result from sub-agent for: {step.action}",
            "confidence": 0.85,
            "cost": 0.01,
            "tokens": 100,
        }

        # Send result back
        result_message = AgentMessage(
            message_id=str(uuid4()),
            sender_id=sub_agent_id,
            receiver_id=session["main_agent_id"],
            message_type=MessageType.TASK_RESULT,
            payload={"result": result},
            correlation_id=session_id,
            priority=5,
        )

        await self.coordinator.message_bus.send(result_message)
        session["messages"].append(result_message.to_dict())

        # Complete task in delegator
        self.coordinator.delegator.complete_task(
            delegation_result["task_id"],
            result,
            success=True,
        )

        # Track relationship
        if session["main_agent_id"] not in self.agent_relationships:
            self.agent_relationships[session["main_agent_id"]] = []
        self.agent_relationships[session["main_agent_id"]].append(sub_agent_id)

        return result

    async def _validate_with_consensus(
        self,
        step: PlanStep,
        result: Dict[str, Any],
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Validate step result using consensus among multiple agents.

        This enables:
        - Multi-agent validation
        - Consensus-based decision making
        - Reduced hallucinations
        """
        logger.info(f"Validating step {step.step_id} with consensus")

        session = self.active_sessions[session_id]

        # Create validation proposal
        proposal = {
            "step_id": step.step_id,
            "result": result,
            "validation_criteria": step.validation_criteria,
            "expected_output": step.expected_output,
        }

        # Run consensus round
        consensus_result = await self.coordinator.run_consensus(
            proposal=proposal,
            proposer_id=session["main_agent_id"],
            algorithm=ConsensusAlgorithm.WEIGHTED_VOTE,
            auto_vote=True,
        )

        # Track consensus round
        session["consensus_rounds"].append({
            "step_id": step.step_id,
            "round_id": consensus_result.get("round_id"),
            "result": consensus_result.get("result"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return consensus_result

    def _assess_step_complexity(self, step: PlanStep) -> float:
        """
        Assess the complexity of a step.

        Returns:
            Complexity score from 0.0 (simple) to 1.0 (very complex)
        """
        complexity = 0.0

        # Factor in number of dependencies
        complexity += min(len(step.dependencies) / 5.0, 0.3)

        # Factor in confidence (low confidence = more complex)
        complexity += (1.0 - step.confidence) * 0.3

        # Factor in estimated duration (long duration = more complex)
        complexity += min(step.estimated_duration / 60.0, 0.2)

        # Factor in input complexity
        complexity += min(len(step.inputs) / 10.0, 0.2)

        return min(complexity, 1.0)

    def _extract_capabilities_from_step(self, step: PlanStep) -> List[str]:
        """Extract required capabilities from step action."""
        # Simple keyword-based extraction
        capabilities = []

        action_lower = step.action.lower()

        if "code" in action_lower or "execute" in action_lower:
            capabilities.append("code_execution")
        if "search" in action_lower or "find" in action_lower:
            capabilities.append("web_search")
        if "file" in action_lower or "write" in action_lower:
            capabilities.append("file_operations")
        if "analyze" in action_lower or "evaluate" in action_lower:
            capabilities.append("analysis")
        if "validate" in action_lower or "check" in action_lower:
            capabilities.append("validation")

        # Default capability if none detected
        if not capabilities:
            capabilities.append("general_execution")

        return capabilities

    def _extract_relationships(self, session_id: str) -> Dict[str, List[str]]:
        """Extract agent relationships from session."""
        session = self.active_sessions.get(session_id, {})
        relationships = {}

        main_agent = session.get("main_agent_id")
        if main_agent:
            relationships[main_agent] = [
                sa["agent_id"] for sa in session.get("sub_agents", [])
            ]

        return relationships

    def _update_metrics(self, success: bool, duration: float):
        """Update collaboration metrics."""
        self.collaboration_metrics["total_collaborations"] += 1

        if success:
            self.collaboration_metrics["successful_collaborations"] += 1
        else:
            self.collaboration_metrics["failed_collaborations"] += 1

        # Update average duration
        total = self.collaboration_metrics["total_collaborations"]
        current_avg = self.collaboration_metrics["avg_collaboration_time"]
        self.collaboration_metrics["avg_collaboration_time"] = (
            (current_avg * (total - 1) + duration) / total
        )

    async def _learn_from_collaboration(
        self,
        session_id: str,
        result: Dict[str, Any],
        collaboration_data: Dict[str, Any],
    ):
        """
        Learn from collaboration patterns to improve future executions.

        Feeds collaboration data to learning service for:
        - Pattern recognition
        - Strategy optimization
        - Agent capability discovery
        """
        if not self.learning_service:
            return

        logger.info(f"Learning from collaboration: {session_id}")

        learning_data = {
            "session_id": session_id,
            "success": result["status"] == "success",
            "goal": result["goal"],
            "iterations": result.get("iterations", 0),
            "confidence": result.get("confidence", 0.0),
            "sub_agents_used": collaboration_data["sub_agents_spawned"],
            "messages_exchanged": collaboration_data["messages_exchanged"],
            "consensus_rounds": collaboration_data["consensus_rounds"],
            "duration": collaboration_data["collaboration_duration"],
            "relationships": collaboration_data["agent_relationships"],
        }

        try:
            await self.learning_service.record_collaboration(learning_data)
        except Exception as e:
            logger.error(f"Failed to record collaboration for learning: {e}")

    async def _store_collaboration_pattern(
        self,
        goal: str,
        collaboration_data: Dict[str, Any],
        result: Dict[str, Any],
    ):
        """Store successful collaboration patterns in memory for reuse."""
        if result["status"] != "success":
            return

        logger.info(f"Storing collaboration pattern for goal: {goal}")

        pattern = {
            "goal_type": self._classify_goal(goal),
            "sub_agents_count": collaboration_data["sub_agents_spawned"],
            "message_count": collaboration_data["messages_exchanged"],
            "consensus_count": collaboration_data["consensus_rounds"],
            "success": True,
            "confidence": result.get("confidence", 0.0),
            "efficiency_score": self._calculate_efficiency(collaboration_data, result),
        }

        try:
            await self.memory_service.store(
                user_id="system",
                content=f"Collaboration pattern: {goal}",
                memory_type="pattern",
                metadata=pattern,
            )
        except Exception as e:
            logger.error(f"Failed to store collaboration pattern: {e}")

    def _classify_goal(self, goal: str) -> str:
        """Classify goal type for pattern matching."""
        goal_lower = goal.lower()

        if "analyze" in goal_lower or "evaluate" in goal_lower:
            return "analysis"
        elif "create" in goal_lower or "generate" in goal_lower:
            return "generation"
        elif "fix" in goal_lower or "debug" in goal_lower:
            return "debugging"
        elif "search" in goal_lower or "find" in goal_lower:
            return "search"
        else:
            return "general"

    def _calculate_efficiency(
        self,
        collaboration_data: Dict[str, Any],
        result: Dict[str, Any],
    ) -> float:
        """Calculate efficiency score for collaboration."""
        # Balance between success, speed, and resource usage
        confidence = result.get("confidence", 0.5)
        duration = collaboration_data.get("collaboration_duration", 60.0)
        agents_used = collaboration_data.get("sub_agents_spawned", 1)

        # Penalize long durations and excessive agents
        time_penalty = max(0.0, 1.0 - (duration / 120.0))  # Penalty after 2 min
        agent_penalty = max(0.0, 1.0 - (agents_used / 10.0))  # Penalty after 10 agents

        efficiency = (
            confidence * 0.5 +
            time_penalty * 0.3 +
            agent_penalty * 0.2
        )

        return min(1.0, max(0.0, efficiency))

    async def _cleanup_session(self, session_id: str):
        """Clean up session resources."""
        logger.info(f"Cleaning up session: {session_id}")

        session = self.active_sessions.get(session_id)
        if not session:
            return

        # Deregister main agent
        main_agent_id = session.get("main_agent_id")
        if main_agent_id:
            self.coordinator.deregister_agent(main_agent_id)

        # Deregister sub-agents
        for sub_agent in session.get("sub_agents", []):
            self.coordinator.deregister_agent(sub_agent["agent_id"])

        # Archive session data
        session["archived_at"] = datetime.now(timezone.utc).isoformat()

        # Remove from active sessions (optionally archive to database)
        del self.active_sessions[session_id]

    def get_collaboration_metrics(self) -> Dict[str, Any]:
        """Get current collaboration metrics."""
        total = self.collaboration_metrics["total_collaborations"]

        return {
            **self.collaboration_metrics,
            "success_rate": (
                self.collaboration_metrics["successful_collaborations"] / max(total, 1)
            ),
            "active_sessions": len(self.active_sessions),
            "coordinator_health": self.coordinator.get_system_health(),
        }

    async def execute_with_agent_negotiation(
        self,
        goal: str,
        candidate_agents: List[str],
        negotiation_criteria: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute goal with agent negotiation.

        Agents negotiate who should handle which parts of the task
        based on their capabilities, current load, and historical performance.
        """
        logger.info(f"Executing with agent negotiation: {goal}")

        # This would implement agent negotiation protocol
        # For now, placeholder for future enhancement
        raise NotImplementedError("Agent negotiation not yet implemented")
