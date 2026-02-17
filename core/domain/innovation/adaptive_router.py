"""
Adaptive Execution Router - Innovation Feature

Routes AI tasks to cheapest viable model/tool chain based on latency, 
confidence, and cost budgets. Uses task complexity analysis, model scoring,
and fallback chains to optimize for cost-performance trade-offs.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4


class TaskComplexity(str, Enum):
    """Task complexity levels"""
    TRIVIAL = "trivial"          # Simple pattern matching, extraction
    SIMPLE = "simple"            # Basic reasoning, straightforward tasks
    MODERATE = "moderate"        # Multi-step reasoning, context needed
    COMPLEX = "complex"          # Deep reasoning, multiple dependencies
    EXPERT = "expert"            # Specialized domain knowledge required


class ModelTier(str, Enum):
    """Model cost/capability tiers"""
    NANO = "nano"                # Ultra-cheap: regex, rules, simple ML
    MICRO = "micro"              # Cheap LLMs: GPT-3.5-turbo, Claude Haiku
    STANDARD = "standard"        # Standard LLMs: GPT-4o-mini, Claude Sonnet
    PREMIUM = "premium"          # Premium LLMs: GPT-4, Claude Opus
    SPECIALIZED = "specialized"  # Domain-specific models


class RoutingStrategy(str, Enum):
    """Routing optimization strategies"""
    COST_FIRST = "cost_first"              # Minimize cost
    LATENCY_FIRST = "latency_first"        # Minimize latency
    QUALITY_FIRST = "quality_first"        # Maximize quality
    BALANCED = "balanced"                  # Balance all factors


class RoutingDecision(str, Enum):
    """Routing decision outcomes"""
    ROUTED = "routed"                      # Successfully routed
    FALLBACK = "fallback"                  # Fallback triggered
    REJECTED = "rejected"                  # No viable route
    BUDGET_EXCEEDED = "budget_exceeded"    # Budget insufficient


# ==================== Value Objects ====================

@dataclass(frozen=True)
class ModelCapability:
    """Model capability specification"""
    model_id: str
    tier: ModelTier
    max_tokens: int
    supports_streaming: bool
    supports_function_calling: bool
    supports_vision: bool
    average_latency_ms: float
    cost_per_1k_tokens: float
    quality_score: float  # 0.0 - 1.0

    def __post_init__(self):
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("Quality score must be between 0.0 and 1.0")
        if self.cost_per_1k_tokens < 0:
            raise ValueError("Cost cannot be negative")
        if self.average_latency_ms < 0:
            raise ValueError("Latency cannot be negative")


@dataclass(frozen=True)
class RoutingConstraints:
    """Constraints for routing decisions"""
    max_cost_usd: Optional[float] = None
    max_latency_ms: Optional[float] = None
    min_quality_score: float = 0.5
    require_streaming: bool = False
    require_function_calling: bool = False
    require_vision: bool = False
    preferred_providers: List[str] = field(default_factory=list)
    blocked_models: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not 0.0 <= self.min_quality_score <= 1.0:
            raise ValueError("Min quality score must be between 0.0 and 1.0")
        if self.max_cost_usd is not None and self.max_cost_usd < 0:
            raise ValueError("Max cost cannot be negative")
        if self.max_latency_ms is not None and self.max_latency_ms < 0:
            raise ValueError("Max latency cannot be negative")


@dataclass
class ComplexitySignals:
    """Signals used for task complexity analysis"""
    input_length: int
    required_reasoning_steps: int
    context_dependencies: int
    domain_specificity: float  # 0.0 - 1.0
    instruction_clarity: float  # 0.0 - 1.0
    output_structure_complexity: float  # 0.0 - 1.0
    historical_similar_complexity: Optional[TaskComplexity] = None

    def __post_init__(self):
        if self.input_length < 0:
            raise ValueError("Input length cannot be negative")
        if self.required_reasoning_steps < 0:
            raise ValueError("Reasoning steps cannot be negative")
        if not 0.0 <= self.domain_specificity <= 1.0:
            raise ValueError("Domain specificity must be between 0.0 and 1.0")
        if not 0.0 <= self.instruction_clarity <= 1.0:
            raise ValueError("Instruction clarity must be between 0.0 and 1.0")
        if not 0.0 <= self.output_structure_complexity <= 1.0:
            raise ValueError("Output structure complexity must be between 0.0 and 1.0")


# ==================== Entities ====================

@dataclass
class TaskAnalysis:
    """
    Task complexity analysis result.
    
    Analyzes task characteristics to determine optimal routing.
    """
    id: UUID
    task_id: UUID
    complexity: TaskComplexity
    signals: ComplexitySignals
    confidence: float  # 0.0 - 1.0
    recommended_tier: ModelTier
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    @staticmethod
    def create(
        task_id: UUID,
        signals: ComplexitySignals
    ) -> "TaskAnalysis":
        """Create task analysis from signals"""
        complexity = TaskAnalysis._determine_complexity(signals)
        tier = TaskAnalysis._recommend_tier(complexity)
        confidence = TaskAnalysis._calculate_confidence(signals)
        reasoning = TaskAnalysis._generate_reasoning(complexity, signals)
        
        return TaskAnalysis(
            id=uuid4(),
            task_id=task_id,
            complexity=complexity,
            signals=signals,
            confidence=confidence,
            recommended_tier=tier,
            reasoning=reasoning
        )

    @staticmethod
    def _determine_complexity(signals: ComplexitySignals) -> TaskComplexity:
        """Determine task complexity from signals"""
        score = 0.0
        
        # Input length score
        if signals.input_length < 100:
            score += 1
        elif signals.input_length < 500:
            score += 2
        elif signals.input_length < 2000:
            score += 3
        else:
            score += 4
        
        # Reasoning steps
        score += min(signals.required_reasoning_steps, 5)
        
        # Context dependencies
        score += min(signals.context_dependencies * 0.5, 3)
        
        # Domain specificity and clarity
        score += signals.domain_specificity * 3
        score += (1 - signals.instruction_clarity) * 2
        score += signals.output_structure_complexity * 2
        
        # Historical data
        if signals.historical_similar_complexity:
            history_map = {
                TaskComplexity.TRIVIAL: 0,
                TaskComplexity.SIMPLE: 5,
                TaskComplexity.MODERATE: 10,
                TaskComplexity.COMPLEX: 15,
                TaskComplexity.EXPERT: 20
            }
            score = (score + history_map[signals.historical_similar_complexity]) / 2
        
        # Map score to complexity
        if score < 3:
            return TaskComplexity.TRIVIAL
        elif score < 6:
            return TaskComplexity.SIMPLE
        elif score < 12:
            return TaskComplexity.MODERATE
        elif score < 18:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.EXPERT

    @staticmethod
    def _recommend_tier(complexity: TaskComplexity) -> ModelTier:
        """Recommend model tier based on complexity"""
        tier_map = {
            TaskComplexity.TRIVIAL: ModelTier.NANO,
            TaskComplexity.SIMPLE: ModelTier.MICRO,
            TaskComplexity.MODERATE: ModelTier.STANDARD,
            TaskComplexity.COMPLEX: ModelTier.PREMIUM,
            TaskComplexity.EXPERT: ModelTier.SPECIALIZED
        }
        return tier_map[complexity]

    @staticmethod
    def _calculate_confidence(signals: ComplexitySignals) -> float:
        """Calculate confidence in analysis"""
        confidence = 0.7  # Base confidence
        
        # Higher clarity -> higher confidence
        confidence += signals.instruction_clarity * 0.2
        
        # Historical data increases confidence
        if signals.historical_similar_complexity:
            confidence += 0.1
        
        return min(confidence, 1.0)

    @staticmethod
    def _generate_reasoning(complexity: TaskComplexity, signals: ComplexitySignals) -> str:
        """Generate human-readable reasoning"""
        reasons = []
        
        if signals.input_length > 2000:
            reasons.append("long input text")
        if signals.required_reasoning_steps > 3:
            reasons.append(f"{signals.required_reasoning_steps} reasoning steps")
        if signals.domain_specificity > 0.7:
            reasons.append("high domain specificity")
        if signals.instruction_clarity < 0.5:
            reasons.append("ambiguous instructions")
        
        if not reasons:
            reasons.append("straightforward task")
        
        return f"Classified as {complexity.value}: {', '.join(reasons)}"


@dataclass
class ModelScore:
    """
    Model scoring for routing decision.
    
    Scores a model based on multiple factors for a specific task.
    """
    model_id: str
    capability: ModelCapability
    viability_score: float  # 0.0 - 1.0
    cost_score: float  # 0.0 - 1.0 (lower cost = higher score)
    latency_score: float  # 0.0 - 1.0 (lower latency = higher score)
    quality_score: float  # 0.0 - 1.0
    composite_score: float  # 0.0 - 1.0
    meets_constraints: bool
    reasoning: List[str] = field(default_factory=list)

    def __post_init__(self):
        scores = [
            self.viability_score, self.cost_score, self.latency_score,
            self.quality_score, self.composite_score
        ]
        for score in scores:
            if not 0.0 <= score <= 1.0:
                raise ValueError("All scores must be between 0.0 and 1.0")

    @staticmethod
    def calculate(
        capability: ModelCapability,
        task_analysis: TaskAnalysis,
        constraints: RoutingConstraints,
        strategy: RoutingStrategy
    ) -> "ModelScore":
        """Calculate model score for routing"""
        reasoning = []
        
        # Check if model is blocked
        if capability.model_id in constraints.blocked_models:
            return ModelScore._create_rejected(capability, "Model is blocked")
        
        # Check capability requirements
        if constraints.require_streaming and not capability.supports_streaming:
            return ModelScore._create_rejected(capability, "Streaming not supported")
        if constraints.require_function_calling and not capability.supports_function_calling:
            return ModelScore._create_rejected(capability, "Function calling not supported")
        if constraints.require_vision and not capability.supports_vision:
            return ModelScore._create_rejected(capability, "Vision not supported")
        
        # Check quality threshold
        if capability.quality_score < constraints.min_quality_score:
            return ModelScore._create_rejected(
                capability, 
                f"Quality {capability.quality_score} below minimum {constraints.min_quality_score}"
            )
        
        # Calculate individual scores
        viability_score = ModelScore._calculate_viability(capability, task_analysis)
        cost_score = ModelScore._calculate_cost_score(capability, constraints)
        latency_score = ModelScore._calculate_latency_score(capability, constraints)
        quality_score = capability.quality_score
        
        # Calculate composite based on strategy
        composite = ModelScore._calculate_composite(
            cost_score, latency_score, quality_score, strategy
        )
        
        meets_constraints = (
            (constraints.max_cost_usd is None or 
             capability.cost_per_1k_tokens <= constraints.max_cost_usd) and
            (constraints.max_latency_ms is None or 
             capability.average_latency_ms <= constraints.max_latency_ms)
        )
        
        if meets_constraints:
            reasoning.append("Meets all constraints")
        else:
            if constraints.max_cost_usd and capability.cost_per_1k_tokens > constraints.max_cost_usd:
                reasoning.append(f"Exceeds cost budget: ${capability.cost_per_1k_tokens:.4f} > ${constraints.max_cost_usd:.4f}")
            if constraints.max_latency_ms and capability.average_latency_ms > constraints.max_latency_ms:
                reasoning.append(f"Exceeds latency budget: {capability.average_latency_ms}ms > {constraints.max_latency_ms}ms")
        
        return ModelScore(
            model_id=capability.model_id,
            capability=capability,
            viability_score=viability_score,
            cost_score=cost_score,
            latency_score=latency_score,
            quality_score=quality_score,
            composite_score=composite,
            meets_constraints=meets_constraints,
            reasoning=reasoning
        )

    @staticmethod
    def _create_rejected(capability: ModelCapability, reason: str) -> "ModelScore":
        """Create a rejected model score"""
        return ModelScore(
            model_id=capability.model_id,
            capability=capability,
            viability_score=0.0,
            cost_score=0.0,
            latency_score=0.0,
            quality_score=0.0,
            composite_score=0.0,
            meets_constraints=False,
            reasoning=[reason]
        )

    @staticmethod
    def _calculate_viability(
        capability: ModelCapability,
        task_analysis: TaskAnalysis
    ) -> float:
        """Calculate if model tier is appropriate for task complexity"""
        tier_order = [
            ModelTier.NANO, ModelTier.MICRO, ModelTier.STANDARD,
            ModelTier.PREMIUM, ModelTier.SPECIALIZED
        ]
        
        model_tier_idx = tier_order.index(capability.tier)
        recommended_tier_idx = tier_order.index(task_analysis.recommended_tier)
        
        # Perfect match
        if model_tier_idx == recommended_tier_idx:
            return 1.0
        
        # Higher tier than needed is okay but wasteful
        if model_tier_idx > recommended_tier_idx:
            diff = model_tier_idx - recommended_tier_idx
            return max(0.7 - (diff * 0.15), 0.5)
        
        # Lower tier than needed may not work
        diff = recommended_tier_idx - model_tier_idx
        return max(0.3 - (diff * 0.15), 0.0)

    @staticmethod
    def _calculate_cost_score(
        capability: ModelCapability,
        constraints: RoutingConstraints
    ) -> float:
        """Calculate cost score (cheaper = higher)"""
        if constraints.max_cost_usd is None:
            # Normalize against typical range ($0.0001 - $0.06 per 1k tokens)
            normalized = 1.0 - min(capability.cost_per_1k_tokens / 0.06, 1.0)
            return max(normalized, 0.0)
        
        if capability.cost_per_1k_tokens > constraints.max_cost_usd:
            return 0.0
        
        # Linear score within budget
        return 1.0 - (capability.cost_per_1k_tokens / constraints.max_cost_usd)

    @staticmethod
    def _calculate_latency_score(
        capability: ModelCapability,
        constraints: RoutingConstraints
    ) -> float:
        """Calculate latency score (faster = higher)"""
        if constraints.max_latency_ms is None:
            # Normalize against typical range (100ms - 10000ms)
            normalized = 1.0 - min(capability.average_latency_ms / 10000, 1.0)
            return max(normalized, 0.0)
        
        if capability.average_latency_ms > constraints.max_latency_ms:
            return 0.0
        
        # Linear score within budget
        return 1.0 - (capability.average_latency_ms / constraints.max_latency_ms)

    @staticmethod
    def _calculate_composite(
        cost: float,
        latency: float,
        quality: float,
        strategy: RoutingStrategy
    ) -> float:
        """Calculate composite score based on strategy"""
        if strategy == RoutingStrategy.COST_FIRST:
            return cost * 0.6 + quality * 0.3 + latency * 0.1
        elif strategy == RoutingStrategy.LATENCY_FIRST:
            return latency * 0.6 + quality * 0.3 + cost * 0.1
        elif strategy == RoutingStrategy.QUALITY_FIRST:
            return quality * 0.6 + cost * 0.2 + latency * 0.2
        else:  # BALANCED
            return (cost + latency + quality) / 3


@dataclass
class FallbackChain:
    """
    Fallback chain for progressive degradation.
    
    Defines a sequence of models to try if primary routing fails.
    """
    id: UUID
    name: str
    models: List[str]  # Ordered list of model IDs
    trigger_conditions: List[str]  # Conditions that trigger fallback
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if not self.models:
            raise ValueError("Fallback chain must have at least one model")
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")

    @staticmethod
    def create(
        name: str,
        models: List[str],
        trigger_conditions: Optional[List[str]] = None
    ) -> "FallbackChain":
        """Create a new fallback chain"""
        return FallbackChain(
            id=uuid4(),
            name=name,
            models=models,
            trigger_conditions=trigger_conditions or ["error", "timeout", "rate_limit"]
        )

    def get_next_model(self, current_model: Optional[str], attempt: int) -> Optional[str]:
        """Get next model in chain"""
        if attempt >= self.max_retries:
            return None
        
        if current_model is None:
            return self.models[0] if self.models else None
        
        try:
            current_idx = self.models.index(current_model)
            next_idx = current_idx + 1
            return self.models[next_idx] if next_idx < len(self.models) else None
        except (ValueError, IndexError):
            return None


@dataclass
class RoutingResult:
    """
    Result of adaptive routing decision.
    
    Contains the selected model and all decision metadata.
    """
    id: UUID
    task_id: UUID
    tenant_id: UUID
    decision: RoutingDecision
    selected_model: Optional[ModelCapability]
    fallback_chain: Optional[FallbackChain]
    task_analysis: TaskAnalysis
    model_scores: List[ModelScore]
    strategy: RoutingStrategy
    constraints: RoutingConstraints
    estimated_cost: float
    estimated_latency_ms: float
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    routed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if self.estimated_cost < 0:
            raise ValueError("Estimated cost cannot be negative")
        if self.estimated_latency_ms < 0:
            raise ValueError("Estimated latency cannot be negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    @property
    def is_successful(self) -> bool:
        """Check if routing was successful"""
        return self.decision == RoutingDecision.ROUTED

    @property
    def requires_fallback(self) -> bool:
        """Check if fallback is available"""
        return self.fallback_chain is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence"""
        return {
            "id": str(self.id),
            "task_id": str(self.task_id),
            "tenant_id": str(self.tenant_id),
            "decision": self.decision.value,
            "selected_model": self.selected_model.model_id if self.selected_model else None,
            "fallback_chain_id": str(self.fallback_chain.id) if self.fallback_chain else None,
            "task_complexity": self.task_analysis.complexity.value,
            "strategy": self.strategy.value,
            "estimated_cost": self.estimated_cost,
            "estimated_latency_ms": self.estimated_latency_ms,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
            "routed_at": self.routed_at.isoformat()
        }


# ==================== Service ====================

class AdaptiveRouterService:
    """
    Adaptive routing service for optimal model selection.
    
    Routes tasks to the most cost-effective model that meets requirements.
    """

    def __init__(
        self,
        available_models: List[ModelCapability],
        default_strategy: RoutingStrategy = RoutingStrategy.BALANCED
    ):
        """
        Initialize adaptive router.
        
        Args:
            available_models: List of available model capabilities
            default_strategy: Default routing strategy
        """
        self.available_models = available_models
        self.default_strategy = default_strategy
        self._model_index = {m.model_id: m for m in available_models}

    async def route_task(
        self,
        task_id: UUID,
        tenant_id: UUID,
        signals: ComplexitySignals,
        constraints: Optional[RoutingConstraints] = None,
        strategy: Optional[RoutingStrategy] = None,
        fallback_chain: Optional[FallbackChain] = None
    ) -> RoutingResult:
        """
        Route task to optimal model.
        
        Args:
            task_id: Task identifier
            tenant_id: Tenant identifier
            signals: Complexity signals for analysis
            constraints: Routing constraints
            strategy: Routing strategy (uses default if None)
            fallback_chain: Fallback chain for retries
            
        Returns:
            Routing result with selected model
        """
        # Analyze task complexity
        task_analysis = TaskAnalysis.create(task_id, signals)
        
        # Apply constraints
        if constraints is None:
            constraints = RoutingConstraints()
        
        # Use specified or default strategy
        routing_strategy = strategy or self.default_strategy
        
        # Score all available models
        model_scores = [
            ModelScore.calculate(model, task_analysis, constraints, routing_strategy)
            for model in self.available_models
        ]
        
        # Sort by composite score (highest first)
        model_scores.sort(key=lambda s: s.composite_score, reverse=True)
        
        # Find best viable model
        viable_scores = [s for s in model_scores if s.meets_constraints and s.viability_score > 0]
        
        if not viable_scores:
            # No viable model found
            return RoutingResult(
                id=uuid4(),
                task_id=task_id,
                tenant_id=tenant_id,
                decision=RoutingDecision.REJECTED,
                selected_model=None,
                fallback_chain=fallback_chain,
                task_analysis=task_analysis,
                model_scores=model_scores,
                strategy=routing_strategy,
                constraints=constraints,
                estimated_cost=0.0,
                estimated_latency_ms=0.0,
                confidence=0.0,
                reasoning="No models meet constraints and requirements"
            )
        
        # Select best model
        best_score = viable_scores[0]
        selected_model = best_score.capability
        
        # Estimate cost and latency (simplified - based on 1k tokens)
        estimated_cost = selected_model.cost_per_1k_tokens
        estimated_latency = selected_model.average_latency_ms
        
        reasoning = (
            f"Selected {selected_model.model_id} ({selected_model.tier.value}) "
            f"for {task_analysis.complexity.value} task. "
            f"Score: {best_score.composite_score:.2f} "
            f"(cost={best_score.cost_score:.2f}, "
            f"latency={best_score.latency_score:.2f}, "
            f"quality={best_score.quality_score:.2f})"
        )
        
        return RoutingResult(
            id=uuid4(),
            task_id=task_id,
            tenant_id=tenant_id,
            decision=RoutingDecision.ROUTED,
            selected_model=selected_model,
            fallback_chain=fallback_chain,
            task_analysis=task_analysis,
            model_scores=model_scores,
            strategy=routing_strategy,
            constraints=constraints,
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency,
            confidence=task_analysis.confidence * best_score.composite_score,
            reasoning=reasoning
        )

    async def get_fallback_model(
        self,
        routing_result: RoutingResult,
        current_attempt: int
    ) -> Optional[ModelCapability]:
        """
        Get fallback model from chain.
        
        Args:
            routing_result: Original routing result
            current_attempt: Current attempt number
            
        Returns:
            Fallback model capability or None
        """
        if not routing_result.fallback_chain:
            return None
        
        current_model = routing_result.selected_model.model_id if routing_result.selected_model else None
        next_model_id = routing_result.fallback_chain.get_next_model(current_model, current_attempt)
        
        return self._model_index.get(next_model_id) if next_model_id else None

    def get_model_by_id(self, model_id: str) -> Optional[ModelCapability]:
        """Get model capability by ID"""
        return self._model_index.get(model_id)

    def add_model(self, capability: ModelCapability) -> None:
        """Add a new model to the routing pool"""
        self.available_models.append(capability)
        self._model_index[capability.model_id] = capability

    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the routing pool"""
        if model_id in self._model_index:
            del self._model_index[model_id]
            self.available_models = [m for m in self.available_models if m.model_id != model_id]
            return True
        return False
