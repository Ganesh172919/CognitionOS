"""
Meta-Learning System for CognitionOS Phase 6
Learns from execution history to improve future performance

Features:
- Execution history analysis
- Pattern recognition
- Strategy evaluation and optimization
- Performance prediction

Target: 40% workflow optimization through learning
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from collections import defaultdict
import statistics

from infrastructure.observability import get_logger


logger = get_logger(__name__)


@dataclass
class ExecutionPattern:
    """Identified execution pattern"""
    pattern_id: str
    pattern_type: str
    task_types: List[str]
    frequency: int
    avg_execution_time_ms: float
    avg_cost_usd: float
    success_rate: float
    recommended_optimizations: List[str]


@dataclass
class StrategyEvaluation:
    """Evaluation of a decomposition or execution strategy"""
    strategy_name: str
    usage_count: int
    success_count: int
    success_rate: float
    avg_execution_time_ms: float
    avg_cost_usd: float
    performance_score: float  # 0.0 to 1.0
    recommendation: str  # keep, optimize, replace


@dataclass
class WorkflowOptimization:
    """Workflow optimization recommendation"""
    workflow_id: str
    optimization_type: str
    current_performance: Dict[str, float]
    predicted_performance: Dict[str, float]
    improvement_percent: float
    confidence: float
    actions: List[str]


class MetaLearningSystem:
    """
    Meta-Learning System
    
    Analyzes execution history to identify patterns, evaluate strategies,
    and recommend optimizations for improved performance.
    """
    
    def __init__(self, db_connection=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Meta-Learning System
        
        Args:
            db_connection: Database connection for querying execution history
            config: Configuration options
        """
        self.db = db_connection
        self.config = config or {}
        
        # Learning parameters
        self.min_sample_size = self.config.get("min_sample_size", 50)
        self.lookback_days = self.config.get("lookback_days", 30)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        
        # Performance improvement threshold
        self.improvement_threshold = self.config.get("improvement_threshold", 0.10)  # 10%
        
        logger.info("MetaLearningSystem initialized")
    
    async def analyze_execution_history(
        self,
        time_window_days: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze execution history for patterns and insights
        
        Args:
            time_window_days: Time window for analysis
            
        Returns:
            Analysis summary with patterns and insights
        """
        logger.info(f"Analyzing execution history for last {time_window_days} days")
        
        # In production, would query execution_history table
        # For now, use mock data
        
        summary = {
            "time_window_days": time_window_days,
            "total_executions": 1000,
            "successful_executions": 920,
            "failed_executions": 80,
            "overall_success_rate": 0.92,
            "avg_execution_time_ms": 1250,
            "avg_cost_usd": 0.015,
            "task_types": {
                "simple_qa": {"count": 400, "success_rate": 0.98, "avg_cost": 0.005},
                "code_generation": {"count": 300, "success_rate": 0.90, "avg_cost": 0.020},
                "complex_reasoning": {"count": 200, "success_rate": 0.85, "avg_cost": 0.030},
                "text_summarization": {"count": 100, "success_rate": 0.95, "avg_cost": 0.008},
            },
            "models_used": {
                "gpt-3.5-turbo": {"count": 600, "success_rate": 0.94, "avg_cost": 0.008},
                "gpt-4": {"count": 300, "success_rate": 0.95, "avg_cost": 0.025},
                "gpt-4-turbo": {"count": 100, "success_rate": 0.96, "avg_cost": 0.018},
            },
            "cache_effectiveness": {
                "l1_redis": {"hit_rate": 0.85, "avg_latency_ms": 1.2},
                "l2_database": {"hit_rate": 0.80, "avg_latency_ms": 8.5},
                "l3_semantic": {"hit_rate": 0.90, "avg_latency_ms": 95.0},
            }
        }
        
        return summary
    
    async def identify_patterns(
        self,
        time_window_days: int = 7
    ) -> List[ExecutionPattern]:
        """
        Identify execution patterns from history
        
        Args:
            time_window_days: Time window for analysis
            
        Returns:
            List of identified patterns
        """
        logger.info("Identifying execution patterns")
        
        # Analyze execution history
        history = await self.analyze_execution_history(time_window_days)
        
        patterns = []
        
        # Pattern 1: Frequent simple tasks that could benefit from caching
        if history["task_types"]["simple_qa"]["count"] > 100:
            patterns.append(ExecutionPattern(
                pattern_id="pattern_1",
                pattern_type="high_frequency_simple_tasks",
                task_types=["simple_qa"],
                frequency=history["task_types"]["simple_qa"]["count"],
                avg_execution_time_ms=500,
                avg_cost_usd=history["task_types"]["simple_qa"]["avg_cost"],
                success_rate=history["task_types"]["simple_qa"]["success_rate"],
                recommended_optimizations=[
                    "Increase L1 cache TTL for simple_qa tasks",
                    "Use GPT-3.5-turbo exclusively for simple_qa",
                    "Implement request deduplication"
                ]
            ))
        
        # Pattern 2: Complex tasks with variable performance
        if history["task_types"]["complex_reasoning"]["success_rate"] < 0.90:
            patterns.append(ExecutionPattern(
                pattern_id="pattern_2",
                pattern_type="variable_performance_complex_tasks",
                task_types=["complex_reasoning"],
                frequency=history["task_types"]["complex_reasoning"]["count"],
                avg_execution_time_ms=2500,
                avg_cost_usd=history["task_types"]["complex_reasoning"]["avg_cost"],
                success_rate=history["task_types"]["complex_reasoning"]["success_rate"],
                recommended_optimizations=[
                    "Always use GPT-4 or better for complex_reasoning",
                    "Add retry logic with different prompting strategies",
                    "Implement multi-step decomposition for complex tasks"
                ]
            ))
        
        # Pattern 3: Cache inefficiency
        for cache_layer, metrics in history["cache_effectiveness"].items():
            if metrics["hit_rate"] < 0.85:
                patterns.append(ExecutionPattern(
                    pattern_id=f"pattern_cache_{cache_layer}",
                    pattern_type="cache_inefficiency",
                    task_types=["all"],
                    frequency=100,
                    avg_execution_time_ms=metrics["avg_latency_ms"],
                    avg_cost_usd=0.01,
                    success_rate=1.0,
                    recommended_optimizations=[
                        f"Increase {cache_layer} TTL",
                        f"Improve similarity threshold for {cache_layer}",
                        f"Implement preemptive cache warming"
                    ]
                ))
        
        logger.info(f"Identified {len(patterns)} execution patterns")
        return patterns
    
    async def evaluate_strategies(
        self,
        time_window_days: int = 7
    ) -> List[StrategyEvaluation]:
        """
        Evaluate performance of different strategies
        
        Args:
            time_window_days: Time window for evaluation
            
        Returns:
            List of strategy evaluations
        """
        logger.info("Evaluating execution strategies")
        
        # In production, would query execution_history and model_routing_decisions tables
        
        evaluations = []
        
        # Evaluate model selection strategies
        model_strategies = {
            "always_gpt35": {
                "usage_count": 600,
                "success_count": 564,
                "avg_execution_time_ms": 800,
                "avg_cost_usd": 0.008,
            },
            "always_gpt4": {
                "usage_count": 300,
                "success_count": 285,
                "avg_execution_time_ms": 1500,
                "avg_cost_usd": 0.025,
            },
            "complexity_based_routing": {
                "usage_count": 100,
                "success_count": 96,
                "avg_execution_time_ms": 1100,
                "avg_cost_usd": 0.012,
            },
        }
        
        for strategy_name, metrics in model_strategies.items():
            success_rate = metrics["success_count"] / metrics["usage_count"]
            
            # Calculate performance score (balance of success, speed, and cost)
            # Higher success rate and lower cost = better score
            normalized_cost = 1.0 - (metrics["avg_cost_usd"] / 0.030)  # Normalize to 0-1
            normalized_speed = 1.0 - (metrics["avg_execution_time_ms"] / 2000)
            
            performance_score = (
                success_rate * 0.5 +
                normalized_cost * 0.3 +
                normalized_speed * 0.2
            )
            
            # Recommendation based on performance
            if performance_score > 0.8:
                recommendation = "keep"
            elif performance_score > 0.6:
                recommendation = "optimize"
            else:
                recommendation = "replace"
            
            evaluations.append(StrategyEvaluation(
                strategy_name=strategy_name,
                usage_count=metrics["usage_count"],
                success_count=metrics["success_count"],
                success_rate=success_rate,
                avg_execution_time_ms=metrics["avg_execution_time_ms"],
                avg_cost_usd=metrics["avg_cost_usd"],
                performance_score=performance_score,
                recommendation=recommendation
            ))
        
        logger.info(f"Evaluated {len(evaluations)} strategies")
        return evaluations
    
    async def generate_optimization_recommendations(
        self,
        workflow_id: Optional[str] = None,
        time_window_days: int = 7
    ) -> List[WorkflowOptimization]:
        """
        Generate workflow optimization recommendations
        
        Args:
            workflow_id: Optional specific workflow to optimize
            time_window_days: Time window for analysis
            
        Returns:
            List of optimization recommendations
        """
        logger.info("Generating optimization recommendations")
        
        # Analyze patterns and strategies
        patterns = await self.identify_patterns(time_window_days)
        strategies = await self.evaluate_strategies(time_window_days)
        
        optimizations = []
        
        # Optimization 1: Switch to complexity-based routing
        best_strategy = max(strategies, key=lambda s: s.performance_score)
        if best_strategy.strategy_name == "complexity_based_routing":
            optimizations.append(WorkflowOptimization(
                workflow_id=workflow_id or "all_workflows",
                optimization_type="model_selection_strategy",
                current_performance={
                    "avg_cost_usd": 0.015,
                    "success_rate": 0.92,
                    "avg_execution_time_ms": 1250
                },
                predicted_performance={
                    "avg_cost_usd": 0.012,  # 20% cost reduction
                    "success_rate": 0.96,    # 4% improvement
                    "avg_execution_time_ms": 1100  # 12% faster
                },
                improvement_percent=20.0,
                confidence=0.85,
                actions=[
                    "Enable IntelligentModelRouter for all workflows",
                    "Configure complexity thresholds: basic<0.3, advanced>0.7",
                    "Monitor model selection accuracy for 7 days"
                ]
            ))
        
        # Optimization 2: Cache improvements
        cache_inefficient = [p for p in patterns if p.pattern_type == "cache_inefficiency"]
        if cache_inefficient:
            optimizations.append(WorkflowOptimization(
                workflow_id=workflow_id or "all_workflows",
                optimization_type="cache_optimization",
                current_performance={
                    "cache_hit_rate": 0.80,
                    "avg_cost_usd": 0.015,
                },
                predicted_performance={
                    "cache_hit_rate": 0.90,  # 10% improvement
                    "avg_cost_usd": 0.010,   # 33% cost reduction
                },
                improvement_percent=33.0,
                confidence=0.80,
                actions=[
                    "Enable AdaptiveCacheOptimizer",
                    "Increase L2 cache TTL to 7200s",
                    "Implement cache warming for common queries"
                ]
            ))
        
        # Optimization 3: Task-specific model assignment
        simple_tasks = [p for p in patterns if "simple" in p.pattern_type.lower()]
        if simple_tasks:
            optimizations.append(WorkflowOptimization(
                workflow_id=workflow_id or "all_workflows",
                optimization_type="task_specific_routing",
                current_performance={
                    "avg_cost_usd": 0.005,
                    "success_rate": 0.98,
                },
                predicted_performance={
                    "avg_cost_usd": 0.004,   # 20% cost reduction
                    "success_rate": 0.98,    # Same quality
                },
                improvement_percent=20.0,
                confidence=0.90,
                actions=[
                    "Route simple_qa tasks exclusively to GPT-3.5-turbo",
                    "Increase cache TTL for simple tasks to 600s",
                    "Enable request deduplication"
                ]
            ))
        
        logger.info(f"Generated {len(optimizations)} optimization recommendations")
        return optimizations
    
    async def predict_performance(
        self,
        task_type: str,
        model: str,
        use_cache: bool = True
    ) -> Dict[str, float]:
        """
        Predict performance for a given task configuration
        
        Args:
            task_type: Type of task
            model: Model to use
            use_cache: Whether to use caching
            
        Returns:
            Predicted performance metrics
        """
        logger.info(f"Predicting performance for {task_type} with {model}")
        
        # Base predictions from historical data
        base_predictions = {
            "execution_time_ms": 1000,
            "cost_usd": 0.015,
            "success_probability": 0.90,
        }
        
        # Adjust based on task type
        task_multipliers = {
            "simple_qa": {"time": 0.5, "cost": 0.3},
            "code_generation": {"time": 1.5, "cost": 1.8},
            "complex_reasoning": {"time": 2.0, "cost": 2.5},
            "text_summarization": {"time": 0.8, "cost": 0.6},
        }
        
        multipliers = task_multipliers.get(task_type, {"time": 1.0, "cost": 1.0})
        
        # Adjust based on model
        model_factors = {
            "gpt-3.5-turbo": {"time": 0.8, "cost": 0.5, "quality": 0.92},
            "gpt-4": {"time": 1.5, "cost": 3.0, "quality": 0.95},
            "gpt-4-turbo": {"time": 1.2, "cost": 1.8, "quality": 0.97},
        }
        
        model_factor = model_factors.get(model, {"time": 1.0, "cost": 1.0, "quality": 0.90})
        
        # Calculate predictions
        predicted_time = base_predictions["execution_time_ms"] * multipliers["time"] * model_factor["time"]
        predicted_cost = base_predictions["cost_usd"] * multipliers["cost"] * model_factor["cost"]
        
        # Cache reduces time and cost
        if use_cache:
            cache_hit_probability = 0.85  # Typical cache hit rate
            predicted_time = predicted_time * (1 - cache_hit_probability * 0.95)  # 95% time saving on hit
            predicted_cost = predicted_cost * (1 - cache_hit_probability * 0.98)  # 98% cost saving on hit
        
        return {
            "predicted_execution_time_ms": predicted_time,
            "predicted_cost_usd": predicted_cost,
            "predicted_success_probability": model_factor["quality"],
            "cache_hit_probability": 0.85 if use_cache else 0.0,
        }
    
    async def learn_from_execution(
        self,
        execution_id: str,
        task_type: str,
        model_used: str,
        actual_execution_time_ms: int,
        actual_cost_usd: float,
        success: bool,
        cache_hit: bool
    ) -> None:
        """
        Learn from a completed execution
        
        Args:
            execution_id: Execution ID
            task_type: Type of task
            model_used: Model that was used
            actual_execution_time_ms: Actual execution time
            actual_cost_usd: Actual cost
            success: Whether execution was successful
            cache_hit: Whether cache was hit
        """
        logger.info(f"Learning from execution {execution_id}")
        
        # In production, would:
        # 1. Store execution in execution_history table
        # 2. Update ML training data
        # 3. Retrain models if enough new data
        # 4. Update adaptive configuration
        
        # For now, just log
        logger.debug(
            f"Execution learned: {task_type}/{model_used} - "
            f"{actual_execution_time_ms}ms, ${actual_cost_usd}, "
            f"success={success}, cache_hit={cache_hit}"
        )
    
    async def run_learning_cycle(self) -> Dict[str, Any]:
        """
        Run a complete learning cycle
        
        Returns:
            Learning cycle results
        """
        logger.info("Starting meta-learning cycle")
        
        try:
            # 1. Analyze execution history
            history = await self.analyze_execution_history(time_window_days=7)
            
            # 2. Identify patterns
            patterns = await self.identify_patterns(time_window_days=7)
            
            # 3. Evaluate strategies
            strategies = await self.evaluate_strategies(time_window_days=7)
            
            # 4. Generate optimizations
            optimizations = await self.generate_optimization_recommendations(time_window_days=7)
            
            # 5. Calculate improvement potential
            total_improvement = sum(opt.improvement_percent for opt in optimizations)
            avg_improvement = total_improvement / len(optimizations) if optimizations else 0
            
            results = {
                "patterns_identified": len(patterns),
                "strategies_evaluated": len(strategies),
                "optimizations_recommended": len(optimizations),
                "avg_improvement_percent": avg_improvement,
                "high_confidence_optimizations": len([o for o in optimizations if o.confidence >= 0.8]),
                "execution_summary": history,
                "top_patterns": [asdict(p) for p in patterns[:3]],
                "best_strategy": asdict(max(strategies, key=lambda s: s.performance_score)) if strategies else None,
                "top_optimizations": [asdict(o) for o in sorted(optimizations, key=lambda o: o.improvement_percent, reverse=True)[:3]]
            }
            
            logger.info(f"Learning cycle complete: {len(patterns)} patterns, {len(optimizations)} optimizations")
            return results
            
        except Exception as e:
            logger.error(f"Error in learning cycle: {e}")
            raise
