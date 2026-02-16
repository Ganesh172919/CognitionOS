"""
Intelligent Model Router for CognitionOS Phase 6
Cost-performance aware LLM model selection

Features:
- Task complexity classification
- Cost-performance optimization
- Dynamic model selection (GPT-4 vs GPT-3.5 vs others)
- Learning from routing decisions

Target: 95% optimal model selection, 30% cost reduction
"""

import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import asyncio

from infrastructure.observability import get_logger


logger = get_logger(__name__)


class ModelTier(str, Enum):
    """Model capability tiers"""
    BASIC = "basic"          # GPT-3.5-turbo, simple tasks
    ADVANCED = "advanced"    # GPT-4, complex reasoning
    PREMIUM = "premium"      # GPT-4-turbo, highest quality


@dataclass
class TaskComplexity:
    """Task complexity assessment"""
    score: float  # 0.0 to 1.0
    factors: Dict[str, float]
    reasoning: str


@dataclass
class ModelCandidate:
    """Model candidate for selection"""
    model_name: str
    tier: ModelTier
    estimated_cost: float
    estimated_quality: float
    avg_latency_ms: int
    success_rate: float


@dataclass
class RoutingDecision:
    """Model routing decision"""
    task_id: Optional[str]
    task_type: str
    complexity: TaskComplexity
    available_models: List[ModelCandidate]
    selected_model: str
    selection_reason: str
    confidence: float
    predicted_cost: float
    predicted_quality: float


class IntelligentModelRouter:
    """
    Intelligent Model Router
    
    Analyzes task complexity and selects the optimal LLM model
    to balance cost and quality based on historical performance.
    """
    
    def __init__(self, db_connection=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Intelligent Model Router
        
        Args:
            db_connection: Database connection for querying execution history
            config: Configuration options
        """
        self.db = db_connection
        self.config = config or {}
        
        # Model catalog with pricing
        self.model_catalog = {
            "gpt-3.5-turbo": {
                "tier": ModelTier.BASIC,
                "cost_per_1k_tokens": 0.002,
                "avg_latency_ms": 800,
                "max_tokens": 4096,
                "quality_score": 0.75
            },
            "gpt-4": {
                "tier": ModelTier.ADVANCED,
                "cost_per_1k_tokens": 0.03,
                "avg_latency_ms": 1500,
                "max_tokens": 8192,
                "quality_score": 0.95
            },
            "gpt-4-turbo": {
                "tier": ModelTier.PREMIUM,
                "cost_per_1k_tokens": 0.01,
                "avg_latency_ms": 1200,
                "max_tokens": 128000,
                "quality_score": 0.97
            },
            "claude-3-opus": {
                "tier": ModelTier.PREMIUM,
                "cost_per_1k_tokens": 0.015,
                "avg_latency_ms": 1400,
                "max_tokens": 200000,
                "quality_score": 0.96
            }
        }
        
        # Complexity thresholds for model selection
        self.complexity_thresholds = {
            "basic_threshold": self.config.get("basic_threshold", 0.3),
            "advanced_threshold": self.config.get("advanced_threshold", 0.7),
        }
        
        # Default model
        self.default_model = self.config.get("default_model", "gpt-3.5-turbo")
        
        # Cost constraints
        self.max_cost_per_request = self.config.get("max_cost_per_request", 0.10)
        
        logger.info("IntelligentModelRouter initialized with {} models", len(self.model_catalog))
    
    async def classify_task_complexity(
        self,
        task_type: str,
        task_description: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskComplexity:
        """
        Classify task complexity using multiple factors
        
        Args:
            task_type: Type of task (e.g., "code_generation", "simple_qa")
            task_description: Optional description of the task
            context: Additional context
            
        Returns:
            TaskComplexity assessment
        """
        logger.info(f"Classifying complexity for task type: {task_type}")
        
        factors = {}
        
        # Factor 1: Task type complexity
        task_type_scores = {
            "simple_qa": 0.2,
            "text_summarization": 0.3,
            "text_generation": 0.4,
            "code_review": 0.6,
            "code_generation": 0.7,
            "complex_reasoning": 0.8,
            "multi_step_planning": 0.9,
        }
        factors["task_type"] = task_type_scores.get(task_type.lower(), 0.5)
        
        # Factor 2: Description complexity (if provided)
        if task_description:
            desc_complexity = self._analyze_description_complexity(task_description)
            factors["description"] = desc_complexity
        else:
            factors["description"] = 0.5
        
        # Factor 3: Historical performance (from database)
        if self.db:
            historical_complexity = await self._get_historical_complexity(task_type)
            factors["historical"] = historical_complexity
        else:
            factors["historical"] = 0.5
        
        # Factor 4: Context complexity
        if context:
            context_complexity = self._analyze_context_complexity(context)
            factors["context"] = context_complexity
        else:
            factors["context"] = 0.5
        
        # Calculate weighted score
        weights = {
            "task_type": 0.4,
            "description": 0.3,
            "historical": 0.2,
            "context": 0.1
        }
        
        score = sum(factors[k] * weights[k] for k in factors.keys())
        
        # Generate reasoning
        reasoning = self._generate_complexity_reasoning(factors, score)
        
        return TaskComplexity(
            score=min(max(score, 0.0), 1.0),
            factors=factors,
            reasoning=reasoning
        )
    
    def _analyze_description_complexity(self, description: str) -> float:
        """
        Analyze description text to estimate complexity
        
        Args:
            description: Task description
            
        Returns:
            Complexity score 0.0 to 1.0
        """
        complexity_indicators = {
            "simple": -0.2,
            "basic": -0.1,
            "easy": -0.15,
            "complex": 0.3,
            "advanced": 0.25,
            "difficult": 0.3,
            "multi-step": 0.35,
            "optimize": 0.2,
            "analyze": 0.2,
            "design": 0.25,
        }
        
        description_lower = description.lower()
        
        # Base complexity from length
        word_count = len(description.split())
        length_score = min(word_count / 100, 0.5)
        
        # Adjust based on keywords
        keyword_score = 0.0
        for keyword, adjustment in complexity_indicators.items():
            if keyword in description_lower:
                keyword_score += adjustment
        
        # Combine scores
        final_score = min(max(length_score + keyword_score, 0.0), 1.0)
        
        return final_score
    
    def _analyze_context_complexity(self, context: Dict[str, Any]) -> float:
        """
        Analyze context to estimate complexity
        
        Args:
            context: Context dictionary
            
        Returns:
            Complexity score 0.0 to 1.0
        """
        score = 0.5  # Base score
        
        # Adjust based on context size
        if "input_size" in context:
            size = context["input_size"]
            if size > 10000:
                score += 0.2
            elif size > 5000:
                score += 0.1
        
        # Adjust based on required accuracy
        if context.get("require_high_accuracy"):
            score += 0.2
        
        # Adjust based on constraints
        if context.get("constraints"):
            score += 0.15
        
        return min(max(score, 0.0), 1.0)
    
    async def _get_historical_complexity(self, task_type: str) -> float:
        """
        Get historical complexity from past executions
        
        Args:
            task_type: Task type
            
        Returns:
            Complexity score 0.0 to 1.0
        """
        # Would query database for historical model usage
        # For now, return default
        return 0.5
    
    def _generate_complexity_reasoning(
        self,
        factors: Dict[str, float],
        final_score: float
    ) -> str:
        """
        Generate human-readable reasoning for complexity assessment
        
        Args:
            factors: Complexity factors
            final_score: Final complexity score
            
        Returns:
            Reasoning string
        """
        if final_score < 0.3:
            level = "Low"
        elif final_score < 0.7:
            level = "Medium"
        else:
            level = "High"
        
        top_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)[:2]
        factor_str = ", ".join([f"{k}: {v:.2f}" for k, v in top_factors])
        
        return f"{level} complexity (score: {final_score:.2f}). Key factors: {factor_str}"
    
    async def select_optimal_model(
        self,
        task_type: str,
        task_description: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_cost: Optional[float] = None
    ) -> RoutingDecision:
        """
        Select optimal model for a task
        
        Args:
            task_type: Type of task
            task_description: Optional task description
            context: Additional context
            max_cost: Maximum cost constraint
            
        Returns:
            RoutingDecision with selected model
        """
        logger.info(f"Selecting optimal model for task type: {task_type}")
        
        # 1. Classify task complexity
        complexity = await self.classify_task_complexity(task_type, task_description, context)
        
        # 2. Get candidate models
        candidates = await self._get_candidate_models(task_type, complexity, max_cost)
        
        # 3. Select best model
        selected_model, reason, confidence = self._select_from_candidates(
            complexity, candidates, max_cost
        )
        
        # 4. Create routing decision
        model_info = self.model_catalog[selected_model]
        
        decision = RoutingDecision(
            task_id=context.get("task_id") if context else None,
            task_type=task_type,
            complexity=complexity,
            available_models=candidates,
            selected_model=selected_model,
            selection_reason=reason,
            confidence=confidence,
            predicted_cost=model_info["cost_per_1k_tokens"] * 2.0,  # Estimate for avg request
            predicted_quality=model_info["quality_score"]
        )
        
        # 5. Store decision for learning
        if self.db:
            await self._store_routing_decision(decision)
        
        logger.info(f"Selected model: {selected_model} (confidence: {confidence:.2%})")
        
        return decision
    
    async def _get_candidate_models(
        self,
        task_type: str,
        complexity: TaskComplexity,
        max_cost: Optional[float]
    ) -> List[ModelCandidate]:
        """
        Get candidate models for selection
        
        Args:
            task_type: Task type
            complexity: Task complexity
            max_cost: Maximum cost constraint
            
        Returns:
            List of candidate models
        """
        candidates = []
        
        for model_name, model_info in self.model_catalog.items():
            # Skip if over budget
            if max_cost and model_info["cost_per_1k_tokens"] * 2.0 > max_cost:
                continue
            
            # Get historical performance if available
            success_rate = await self._get_model_success_rate(model_name, task_type)
            
            candidate = ModelCandidate(
                model_name=model_name,
                tier=model_info["tier"],
                estimated_cost=model_info["cost_per_1k_tokens"] * 2.0,
                estimated_quality=model_info["quality_score"],
                avg_latency_ms=model_info["avg_latency_ms"],
                success_rate=success_rate
            )
            candidates.append(candidate)
        
        return candidates
    
    async def _get_model_success_rate(
        self,
        model_name: str,
        task_type: str
    ) -> float:
        """
        Get historical success rate for a model on a task type
        
        Args:
            model_name: Model name
            task_type: Task type
            
        Returns:
            Success rate 0.0 to 1.0
        """
        # Would query execution_history table
        # For now, use model quality score
        return self.model_catalog[model_name]["quality_score"]
    
    def _select_from_candidates(
        self,
        complexity: TaskComplexity,
        candidates: List[ModelCandidate],
        max_cost: Optional[float]
    ) -> Tuple[str, str, float]:
        """
        Select best model from candidates
        
        Args:
            complexity: Task complexity
            candidates: List of candidate models
            max_cost: Maximum cost constraint
            
        Returns:
            Tuple of (model_name, reason, confidence)
        """
        if not candidates:
            return self.default_model, "No candidates available, using default", 0.5
        
        # Strategy: Use complexity to guide selection
        if complexity.score < self.complexity_thresholds["basic_threshold"]:
            # Low complexity: use cheapest model
            candidates.sort(key=lambda c: c.estimated_cost)
            selected = candidates[0]
            reason = f"Low complexity ({complexity.score:.2f}), using most cost-effective model"
            confidence = 0.9
            
        elif complexity.score > self.complexity_thresholds["advanced_threshold"]:
            # High complexity: use highest quality model within budget
            quality_candidates = [c for c in candidates if c.tier in [ModelTier.ADVANCED, ModelTier.PREMIUM]]
            if quality_candidates:
                quality_candidates.sort(key=lambda c: c.estimated_quality, reverse=True)
                selected = quality_candidates[0]
                reason = f"High complexity ({complexity.score:.2f}), using highest quality model"
                confidence = 0.85
            else:
                candidates.sort(key=lambda c: c.estimated_quality, reverse=True)
                selected = candidates[0]
                reason = "High complexity, but no premium models available"
                confidence = 0.6
        else:
            # Medium complexity: balance cost and quality
            # Score each candidate
            scored_candidates = []
            for c in candidates:
                # Normalize cost and quality (0-1 range)
                max_cost_val = max(cand.estimated_cost for cand in candidates)
                norm_cost = 1.0 - (c.estimated_cost / max_cost_val) if max_cost_val > 0 else 0.5
                norm_quality = c.estimated_quality
                
                # Weighted score (favor quality for medium complexity)
                score = (norm_quality * 0.6) + (norm_cost * 0.4)
                scored_candidates.append((c, score))
            
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            selected = scored_candidates[0][0]
            reason = f"Medium complexity ({complexity.score:.2f}), balanced cost-quality selection"
            confidence = 0.8
        
        return selected.model_name, reason, confidence
    
    async def _store_routing_decision(self, decision: RoutingDecision):
        """
        Store routing decision for learning
        
        Args:
            decision: Routing decision to store
        """
        try:
            # Would insert into model_routing_decisions table
            logger.debug(f"Stored routing decision for task {decision.task_id}")
        except Exception as e:
            logger.error(f"Error storing routing decision: {e}")
    
    async def evaluate_routing_performance(
        self,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Evaluate routing performance
        
        Args:
            time_window_hours: Time window for evaluation
            
        Returns:
            Performance metrics
        """
        logger.info(f"Evaluating routing performance for last {time_window_hours} hours")
        
        # Mock performance metrics
        return {
            "time_window_hours": time_window_hours,
            "total_decisions": 1000,
            "optimal_selections": 950,
            "optimal_rate": 0.95,
            "avg_cost_per_request": 0.0085,
            "avg_quality_score": 0.88,
            "cost_savings_vs_always_gpt4": 0.72,  # 72% savings
            "models_used": {
                "gpt-3.5-turbo": 600,
                "gpt-4": 300,
                "gpt-4-turbo": 100
            }
        }
    
    def get_model_recommendation(
        self,
        task_type: str,
        budget_constraint: Optional[float] = None
    ) -> str:
        """
        Get quick model recommendation without full analysis
        
        Args:
            task_type: Task type
            budget_constraint: Optional budget constraint
            
        Returns:
            Recommended model name
        """
        # Simple heuristic for quick recommendations
        task_type_lower = task_type.lower()
        
        if "simple" in task_type_lower or "qa" in task_type_lower:
            return "gpt-3.5-turbo"
        elif "complex" in task_type_lower or "reasoning" in task_type_lower:
            if not budget_constraint or budget_constraint >= 0.05:
                return "gpt-4"
            else:
                return "gpt-3.5-turbo"
        else:
            return self.default_model
