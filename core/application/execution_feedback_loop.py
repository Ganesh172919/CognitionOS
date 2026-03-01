"""
Execution Feedback Loop - Closed-loop learning system

Connects execution results back to model improvement through:
- Real-time execution tracking
- Automatic prompt optimization
- Strategy adaptation from failures
- Performance-based model updates
- Continuous improvement cycles
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple
from uuid import uuid4
from dataclasses import dataclass, field, asdict
from enum import Enum

from infrastructure.intelligence.meta_learning import (
    MetaLearningSystem,
    ExecutionPattern,
    StrategyEvaluation,
    WorkflowOptimization,
)
from infrastructure.federated_learning.federated_engine import (
    FederatedLearningEngine,
    FederationConfig,
    DifferentialPrivacyConfig,
)

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback for learning."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL_SUCCESS = "partial_success"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"
    VALIDATION_FAILED = "validation_failed"
    HALLUCINATION_DETECTED = "hallucination_detected"


class OptimizationTarget(str, Enum):
    """What to optimize."""
    PROMPTS = "prompts"
    STRATEGIES = "strategies"
    MODEL_SELECTION = "model_selection"
    TOOL_SELECTION = "tool_selection"
    PARAMETERS = "parameters"
    WORKFLOW = "workflow"


@dataclass
class ExecutionFeedback:
    """Feedback from a single execution."""
    execution_id: str
    goal: str
    agent_role: str
    strategy_used: str
    feedback_type: FeedbackType
    success: bool
    confidence: float
    iterations: int
    duration_seconds: float
    cost_usd: float
    tokens_used: int

    # Detailed information
    plan_quality: float  # How good was the plan
    execution_quality: float  # How well was it executed
    validation_quality: float  # How well was it validated

    # What worked / didn't work
    successful_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Context
    input_complexity: float  # 0-1 complexity of input
    output_quality: float  # 0-1 quality of output
    user_satisfaction: Optional[float] = None  # If available

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_training_sample(self) -> Dict[str, Any]:
        """Convert to training sample for model improvement."""
        return {
            "input": self.goal,
            "strategy": self.strategy_used,
            "success": self.success,
            "confidence": self.confidence,
            "plan_quality": self.plan_quality,
            "execution_quality": self.execution_quality,
            "output_quality": self.output_quality,
            "metadata": {
                "duration": self.duration_seconds,
                "cost": self.cost_usd,
                "iterations": self.iterations,
            }
        }


@dataclass
class PromptOptimization:
    """Optimized prompt recommendation."""
    original_prompt: str
    optimized_prompt: str
    optimization_type: str
    expected_improvement: float
    confidence: float
    based_on_samples: int
    reasoning: str


@dataclass
class StrategyAdaptation:
    """Strategy adaptation recommendation."""
    strategy_name: str
    adaptation_type: str  # "replace", "modify", "remove"
    current_performance: Dict[str, float]
    recommended_changes: Dict[str, Any]
    expected_improvement: float
    confidence: float


class ExecutionFeedbackLoop:
    """
    Closed-loop learning system that continuously improves agents.

    The loop:
    1. Collect execution feedback
    2. Analyze patterns and failures
    3. Generate optimizations (prompts, strategies, parameters)
    4. Apply optimizations
    5. Measure improvements
    6. Repeat
    """

    def __init__(
        self,
        meta_learning_system: MetaLearningSystem,
        federated_learning_engine: Optional[FederatedLearningEngine] = None,
        memory_service: Optional[Any] = None,
        min_samples_for_optimization: int = 20,
        optimization_interval_seconds: int = 3600,  # 1 hour
    ):
        """
        Initialize feedback loop.

        Args:
            meta_learning_system: Meta-learning system for pattern analysis
            federated_learning_engine: Optional federated learning for privacy-preserving updates
            memory_service: Long-term memory for storing optimizations
            min_samples_for_optimization: Minimum samples before optimizing
            optimization_interval_seconds: How often to run optimization
        """
        self.meta_learning = meta_learning_system
        self.federated_learning = federated_learning_engine
        self.memory_service = memory_service
        self.min_samples = min_samples_for_optimization
        self.optimization_interval = optimization_interval_seconds

        # Execution feedback buffer
        self.feedback_buffer: List[ExecutionFeedback] = []
        self.feedback_by_strategy: Dict[str, List[ExecutionFeedback]] = defaultdict(list)
        self.feedback_by_agent: Dict[str, List[ExecutionFeedback]] = defaultdict(list)

        # Current optimizations
        self.active_optimizations: Dict[str, Any] = {}
        self.optimization_history: List[Dict[str, Any]] = []

        # Performance tracking
        self.baseline_performance: Dict[str, float] = {}
        self.current_performance: Dict[str, float] = {}

        # Learning statistics
        self.stats = {
            "total_feedback_collected": 0,
            "optimizations_generated": 0,
            "optimizations_applied": 0,
            "avg_improvement": 0.0,
            "learning_cycles": 0,
        }

        # Background optimization task
        self._optimization_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info("ExecutionFeedbackLoop initialized")

    async def start(self):
        """Start the continuous learning loop."""
        if self._running:
            logger.warning("Feedback loop already running")
            return

        self._running = True
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        logger.info("Feedback loop started")

    async def stop(self):
        """Stop the continuous learning loop."""
        self._running = False
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        logger.info("Feedback loop stopped")

    async def record_execution(
        self,
        execution_result: Dict[str, Any],
        collaboration_data: Optional[Dict[str, Any]] = None,
    ) -> ExecutionFeedback:
        """
        Record execution feedback for learning.

        Args:
            execution_result: Result from agent execution
            collaboration_data: Optional collaboration metadata

        Returns:
            Execution feedback object
        """
        # Extract feedback from execution result
        feedback = ExecutionFeedback(
            execution_id=execution_result.get("execution_id", str(uuid4())),
            goal=execution_result.get("goal", ""),
            agent_role=execution_result.get("agent_role", "unknown"),
            strategy_used=execution_result.get("strategy", "default"),
            feedback_type=self._classify_feedback(execution_result),
            success=execution_result.get("status") == "success",
            confidence=execution_result.get("confidence", 0.0),
            iterations=execution_result.get("iterations", 1),
            duration_seconds=execution_result.get("duration_seconds", 0.0),
            cost_usd=execution_result.get("budget_used", {}).get("cost_usd", 0.0),
            tokens_used=execution_result.get("budget_used", {}).get("tokens", 0),
            plan_quality=self._assess_plan_quality(execution_result),
            execution_quality=self._assess_execution_quality(execution_result),
            validation_quality=self._assess_validation_quality(execution_result),
            successful_steps=execution_result.get("successful_steps", []),
            failed_steps=execution_result.get("failed_steps", []),
            errors=execution_result.get("errors", []),
            input_complexity=self._assess_input_complexity(execution_result.get("goal", "")),
            output_quality=self._assess_output_quality(execution_result.get("result", {})),
        )

        # Add to buffers
        self.feedback_buffer.append(feedback)
        self.feedback_by_strategy[feedback.strategy_used].append(feedback)
        self.feedback_by_agent[feedback.agent_role].append(feedback)

        self.stats["total_feedback_collected"] += 1

        # Update current performance metrics
        self._update_performance_metrics(feedback)

        # If federated learning enabled, prepare training sample
        if self.federated_learning:
            training_sample = feedback.to_training_sample()
            # Would send to federated learning system
            # await self._submit_to_federated_learning(training_sample)

        logger.info(
            f"Recorded execution feedback: {feedback.execution_id} "
            f"(success={feedback.success}, confidence={feedback.confidence:.2f})"
        )

        return feedback

    async def _optimization_loop(self):
        """Background loop that periodically runs optimizations."""
        while self._running:
            try:
                await asyncio.sleep(self.optimization_interval)

                if len(self.feedback_buffer) >= self.min_samples:
                    logger.info("Running optimization cycle...")
                    await self._run_optimization_cycle()
                else:
                    logger.debug(
                        f"Not enough samples for optimization: "
                        f"{len(self.feedback_buffer)}/{self.min_samples}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}", exc_info=True)

    async def _run_optimization_cycle(self):
        """Run a complete optimization cycle."""
        cycle_start = datetime.now(timezone.utc)
        self.stats["learning_cycles"] += 1

        logger.info(f"Optimization cycle {self.stats['learning_cycles']} started")

        # 1. Analyze patterns from feedback
        patterns = await self._analyze_patterns()

        # 2. Evaluate strategies
        strategy_evals = await self._evaluate_strategies()

        # 3. Generate optimizations
        optimizations = await self._generate_optimizations(patterns, strategy_evals)

        # 4. Apply high-confidence optimizations
        applied_count = await self._apply_optimizations(optimizations)

        # 5. Store learnings in memory
        if self.memory_service:
            await self._store_learnings(patterns, optimizations)

        cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()

        logger.info(
            f"Optimization cycle complete: "
            f"{len(optimizations)} generated, {applied_count} applied, "
            f"duration={cycle_duration:.2f}s"
        )

        # Clear old feedback (keep last 1000)
        if len(self.feedback_buffer) > 1000:
            self.feedback_buffer = self.feedback_buffer[-1000:]

    async def _analyze_patterns(self) -> List[ExecutionPattern]:
        """Analyze execution patterns from feedback."""
        logger.info("Analyzing execution patterns...")

        patterns = []

        # Group by goal type
        goal_types = defaultdict(list)
        for feedback in self.feedback_buffer[-200:]:  # Last 200 samples
            goal_type = self._classify_goal_type(feedback.goal)
            goal_types[goal_type].append(feedback)

        # Analyze each goal type
        for goal_type, feedbacks in goal_types.items():
            if len(feedbacks) < 5:
                continue

            success_rate = sum(1 for f in feedbacks if f.success) / len(feedbacks)
            avg_duration = sum(f.duration_seconds for f in feedbacks) / len(feedbacks)
            avg_cost = sum(f.cost_usd for f in feedbacks) / len(feedbacks)

            # Identify optimization opportunities
            recommendations = []
            if success_rate < 0.7:
                recommendations.append("Improve strategy selection")
            if avg_duration > 30:
                recommendations.append("Optimize execution speed")
            if avg_cost > 0.5:
                recommendations.append("Reduce token usage")

            pattern = ExecutionPattern(
                pattern_id=f"pattern_{goal_type}_{uuid4().hex[:8]}",
                pattern_type=goal_type,
                task_types=[feedback.strategy_used for feedback in feedbacks],
                frequency=len(feedbacks),
                avg_execution_time_ms=avg_duration * 1000,
                avg_cost_usd=avg_cost,
                success_rate=success_rate,
                recommended_optimizations=recommendations,
            )
            patterns.append(pattern)

        logger.info(f"Found {len(patterns)} execution patterns")
        return patterns

    async def _evaluate_strategies(self) -> List[StrategyEvaluation]:
        """Evaluate performance of different strategies."""
        logger.info("Evaluating strategies...")

        evaluations = []

        for strategy_name, feedbacks in self.feedback_by_strategy.items():
            if len(feedbacks) < 10:
                continue

            success_count = sum(1 for f in feedbacks if f.success)
            usage_count = len(feedbacks)
            success_rate = success_count / usage_count

            avg_duration = sum(f.duration_seconds for f in feedbacks) / usage_count
            avg_cost = sum(f.cost_usd for f in feedbacks) / usage_count

            # Performance score (balance success, speed, cost)
            performance_score = (
                success_rate * 0.6 +
                max(0, (1 - avg_duration / 60)) * 0.2 +  # Penalty after 1 min
                max(0, (1 - avg_cost)) * 0.2  # Penalty after $1
            )

            # Recommendation
            if performance_score > 0.7:
                recommendation = "keep"
            elif performance_score > 0.5:
                recommendation = "optimize"
            else:
                recommendation = "replace"

            evaluation = StrategyEvaluation(
                strategy_name=strategy_name,
                usage_count=usage_count,
                success_count=success_count,
                success_rate=success_rate,
                avg_execution_time_ms=avg_duration * 1000,
                avg_cost_usd=avg_cost,
                performance_score=performance_score,
                recommendation=recommendation,
            )
            evaluations.append(evaluation)

        logger.info(f"Evaluated {len(evaluations)} strategies")
        return evaluations

    async def _generate_optimizations(
        self,
        patterns: List[ExecutionPattern],
        strategy_evals: List[StrategyEvaluation],
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        logger.info("Generating optimizations...")

        optimizations = []

        # Prompt optimizations from patterns
        for pattern in patterns:
            if "Improve strategy selection" in pattern.recommended_optimizations:
                optimizations.append({
                    "type": OptimizationTarget.PROMPTS,
                    "target": pattern.pattern_type,
                    "action": "improve_strategy_selection",
                    "reason": f"Low success rate: {pattern.success_rate:.2f}",
                    "confidence": 0.7,
                    "expected_improvement": 0.15,
                })

            if "Optimize execution speed" in pattern.recommended_optimizations:
                optimizations.append({
                    "type": OptimizationTarget.STRATEGIES,
                    "target": pattern.pattern_type,
                    "action": "use_parallel_execution",
                    "reason": f"High duration: {pattern.avg_execution_time_ms/1000:.1f}s",
                    "confidence": 0.8,
                    "expected_improvement": 0.3,
                })

        # Strategy adaptations from evaluations
        for eval in strategy_evals:
            if eval.recommendation == "replace":
                optimizations.append({
                    "type": OptimizationTarget.STRATEGIES,
                    "target": eval.strategy_name,
                    "action": "replace_strategy",
                    "reason": f"Low performance score: {eval.performance_score:.2f}",
                    "confidence": 0.6,
                    "expected_improvement": 0.25,
                    "data": asdict(eval),
                })
            elif eval.recommendation == "optimize":
                optimizations.append({
                    "type": OptimizationTarget.PARAMETERS,
                    "target": eval.strategy_name,
                    "action": "tune_parameters",
                    "reason": f"Moderate performance: {eval.performance_score:.2f}",
                    "confidence": 0.75,
                    "expected_improvement": 0.1,
                    "data": asdict(eval),
                })

        self.stats["optimizations_generated"] += len(optimizations)

        logger.info(f"Generated {len(optimizations)} optimizations")
        return optimizations

    async def _apply_optimizations(
        self,
        optimizations: List[Dict[str, Any]],
    ) -> int:
        """Apply high-confidence optimizations."""
        logger.info("Applying optimizations...")

        applied_count = 0

        for opt in optimizations:
            # Only apply high-confidence optimizations
            if opt["confidence"] < 0.7:
                logger.debug(f"Skipping low-confidence optimization: {opt['action']}")
                continue

            try:
                success = await self._apply_single_optimization(opt)
                if success:
                    applied_count += 1
                    self.active_optimizations[opt["target"]] = opt
                    self.optimization_history.append({
                        **opt,
                        "applied_at": datetime.now(timezone.utc).isoformat(),
                        "status": "applied",
                    })
                    logger.info(f"Applied optimization: {opt['action']} for {opt['target']}")
            except Exception as e:
                logger.error(f"Failed to apply optimization: {e}")

        self.stats["optimizations_applied"] += applied_count

        return applied_count

    async def _apply_single_optimization(self, optimization: Dict[str, Any]) -> bool:
        """Apply a single optimization."""
        opt_type = optimization["type"]
        action = optimization["action"]

        # This would integrate with actual system components
        # For now, log the optimization

        if opt_type == OptimizationTarget.PROMPTS:
            # Would update prompt templates
            logger.info(f"Would update prompts for: {optimization['target']}")
            return True

        elif opt_type == OptimizationTarget.STRATEGIES:
            # Would update strategy selection logic
            logger.info(f"Would update strategy: {optimization['target']}")
            return True

        elif opt_type == OptimizationTarget.PARAMETERS:
            # Would tune parameters
            logger.info(f"Would tune parameters: {optimization['target']}")
            return True

        return False

    async def _store_learnings(
        self,
        patterns: List[ExecutionPattern],
        optimizations: List[Dict[str, Any]],
    ):
        """Store learnings in long-term memory."""
        if not self.memory_service:
            return

        logger.info("Storing learnings in memory...")

        # Store patterns
        for pattern in patterns:
            await self.memory_service.store(
                user_id="system",
                content=f"Execution pattern: {pattern.pattern_type}",
                memory_type="pattern",
                metadata={
                    "pattern_id": pattern.pattern_id,
                    "success_rate": pattern.success_rate,
                    "avg_cost": pattern.avg_cost_usd,
                    "recommendations": pattern.recommended_optimizations,
                }
            )

        # Store optimization history
        for opt in optimizations:
            if opt.get("status") == "applied":
                await self.memory_service.store(
                    user_id="system",
                    content=f"Applied optimization: {opt['action']}",
                    memory_type="optimization",
                    metadata=opt,
                )

    def _classify_feedback(self, result: Dict[str, Any]) -> FeedbackType:
        """Classify feedback type from result."""
        if result.get("status") == "success":
            return FeedbackType.SUCCESS
        elif result.get("error_type") == "timeout":
            return FeedbackType.TIMEOUT
        elif result.get("error_type") == "budget_exceeded":
            return FeedbackType.BUDGET_EXCEEDED
        elif result.get("validation_failed"):
            return FeedbackType.VALIDATION_FAILED
        elif result.get("hallucination_detected"):
            return FeedbackType.HALLUCINATION_DETECTED
        elif result.get("partial_success"):
            return FeedbackType.PARTIAL_SUCCESS
        else:
            return FeedbackType.FAILURE

    def _assess_plan_quality(self, result: Dict[str, Any]) -> float:
        """Assess quality of the plan."""
        plan_summary = result.get("plan_summary", {})

        # Simple heuristic based on completion rate
        total_steps = plan_summary.get("total_steps", 1)
        completed_steps = plan_summary.get("steps_completed", 0)

        return completed_steps / max(total_steps, 1)

    def _assess_execution_quality(self, result: Dict[str, Any]) -> float:
        """Assess quality of execution."""
        # Based on iterations needed
        iterations = result.get("iterations", 1)

        # Fewer iterations = better execution
        return max(0.0, 1.0 - (iterations - 1) * 0.2)

    def _assess_validation_quality(self, result: Dict[str, Any]) -> float:
        """Assess quality of validation."""
        return result.get("confidence", 0.5)

    def _assess_input_complexity(self, goal: str) -> float:
        """Assess complexity of input goal."""
        # Simple heuristic based on length and keywords
        complexity = min(len(goal) / 500, 0.5)  # Length factor

        complex_keywords = ["complex", "multi", "advanced", "comprehensive", "integrate"]
        complexity += sum(0.1 for kw in complex_keywords if kw in goal.lower())

        return min(complexity, 1.0)

    def _assess_output_quality(self, result: Dict[str, Any]) -> float:
        """Assess quality of output."""
        # Would use more sophisticated quality metrics
        return result.get("quality_score", 0.7)

    def _classify_goal_type(self, goal: str) -> str:
        """Classify goal into type."""
        goal_lower = goal.lower()

        if "analyze" in goal_lower or "evaluate" in goal_lower:
            return "analysis"
        elif "create" in goal_lower or "generate" in goal_lower:
            return "generation"
        elif "fix" in goal_lower or "debug" in goal_lower:
            return "debugging"
        elif "search" in goal_lower or "find" in goal_lower:
            return "search"
        elif "optimize" in goal_lower:
            return "optimization"
        else:
            return "general"

    def _update_performance_metrics(self, feedback: ExecutionFeedback):
        """Update running performance metrics."""
        key = feedback.strategy_used

        if key not in self.current_performance:
            self.current_performance[key] = {
                "success_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_duration": 0.0,
                "count": 0,
            }

        metrics = self.current_performance[key]
        count = metrics["count"]

        # Update running averages
        metrics["success_rate"] = (
            (metrics["success_rate"] * count + (1 if feedback.success else 0)) / (count + 1)
        )
        metrics["avg_confidence"] = (
            (metrics["avg_confidence"] * count + feedback.confidence) / (count + 1)
        )
        metrics["avg_duration"] = (
            (metrics["avg_duration"] * count + feedback.duration_seconds) / (count + 1)
        )
        metrics["count"] = count + 1

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics."""
        return {
            **self.stats,
            "feedback_buffer_size": len(self.feedback_buffer),
            "strategies_tracked": len(self.feedback_by_strategy),
            "agents_tracked": len(self.feedback_by_agent),
            "active_optimizations": len(self.active_optimizations),
            "current_performance": self.current_performance,
        }
