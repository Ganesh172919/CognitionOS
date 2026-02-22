"""
Advanced Multi-Model AI Orchestration Layer

Intelligently orchestrates multiple AI models, providers, and strategies
for optimal performance, cost, and quality.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json


class ModelProvider(Enum):
    """Supported AI model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class ModelCapability(Enum):
    """Model capability types"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    REASONING = "reasoning"
    VISION = "vision"
    AUDIO = "audio"


class SelectionStrategy(Enum):
    """Model selection strategies"""
    LOWEST_COST = "lowest_cost"
    FASTEST = "fastest"
    HIGHEST_QUALITY = "highest_quality"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"


@dataclass
class ModelConfig:
    """AI model configuration"""
    model_id: str
    provider: ModelProvider
    capabilities: List[ModelCapability]
    cost_per_1k_tokens: float
    avg_latency_ms: float
    quality_score: float  # 0-100
    max_tokens: int
    supports_streaming: bool = False
    supports_function_calling: bool = False
    context_window: int = 4096
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelRequest:
    """Request for AI model"""
    prompt: str
    capability: ModelCapability
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    stream: bool = False
    functions: Optional[List[Dict]] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Response from AI model"""
    content: str
    model_id: str
    provider: ModelProvider
    tokens_used: int
    latency_ms: float
    cost_usd: float
    quality_estimate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble inference"""
    models: List[str]
    voting_strategy: str = "majority"  # majority, weighted, unanimous
    confidence_threshold: float = 0.8
    fallback_on_disagreement: bool = True


class MultiModelOrchestrator:
    """
    Advanced Multi-Model AI Orchestration System

    Features:
    - Intelligent model selection based on cost, latency, and quality
    - Automatic fallback and retry across providers
    - Ensemble inference for critical tasks
    - A/B testing of different models
    - Cost optimization and budget management
    - Performance monitoring and adaptive learning
    - Prompt caching and optimization
    - Function calling orchestration
    """

    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.selection_strategy = SelectionStrategy.BALANCED
        self._request_history: List[Dict[str, Any]] = []
        self._model_performance: Dict[str, Dict[str, float]] = {}
        self._prompt_cache: Dict[str, ModelResponse] = {}
        self._budget_tracker: Dict[str, float] = {}

    def register_model(self, config: ModelConfig):
        """Register an AI model"""
        self.models[config.model_id] = config
        self._model_performance[config.model_id] = {
            "total_requests": 0,
            "success_rate": 1.0,
            "avg_latency": config.avg_latency_ms,
            "avg_cost": config.cost_per_1k_tokens,
            "quality_score": config.quality_score
        }

    async def generate(
        self,
        request: ModelRequest,
        selection_strategy: Optional[SelectionStrategy] = None
    ) -> ModelResponse:
        """
        Generate response using optimal model

        Args:
            request: Model request configuration
            selection_strategy: Override default selection strategy

        Returns:
            Model response with metadata
        """
        # Check cache first
        cache_key = self._get_cache_key(request)
        if cache_key in self._prompt_cache and not request.stream:
            cached = self._prompt_cache[cache_key]
            return ModelResponse(
                content=cached.content,
                model_id=cached.model_id + " (cached)",
                provider=cached.provider,
                tokens_used=0,
                latency_ms=0,
                cost_usd=0,
                quality_estimate=cached.quality_estimate,
                metadata={"cached": True}
            )

        # Select optimal model
        strategy = selection_strategy or self.selection_strategy
        model = self._select_model(request, strategy)

        if not model:
            raise ValueError(f"No suitable model found for capability: {request.capability}")

        # Execute request with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self._execute_request(model, request)

                # Cache successful response
                if not request.stream:
                    self._prompt_cache[cache_key] = response

                # Update performance metrics
                self._update_metrics(model.model_id, response, success=True)

                return response

            except Exception as e:
                self._update_metrics(model.model_id, None, success=False)

                if attempt < max_retries - 1:
                    # Try fallback model
                    model = self._get_fallback_model(model, request)
                    if not model:
                        raise
                else:
                    raise

    async def ensemble_generate(
        self,
        request: ModelRequest,
        config: EnsembleConfig
    ) -> ModelResponse:
        """
        Generate response using ensemble of models

        Args:
            request: Model request
            config: Ensemble configuration

        Returns:
            Aggregated response from ensemble
        """
        # Execute requests in parallel
        tasks = []
        for model_id in config.models:
            if model_id in self.models:
                model = self.models[model_id]
                tasks.append(self._execute_request(model, request))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        valid_responses = [r for r in responses if isinstance(r, ModelResponse)]

        if not valid_responses:
            raise Exception("All ensemble models failed")

        # Apply voting strategy
        if config.voting_strategy == "majority":
            return self._majority_vote(valid_responses)
        elif config.voting_strategy == "weighted":
            return self._weighted_vote(valid_responses)
        elif config.voting_strategy == "unanimous":
            return self._unanimous_vote(valid_responses)

        return valid_responses[0]

    def _select_model(
        self,
        request: ModelRequest,
        strategy: SelectionStrategy
    ) -> Optional[ModelConfig]:
        """Select optimal model based on strategy"""
        # Filter models by capability
        capable_models = [
            m for m in self.models.values()
            if request.capability in m.capabilities
        ]

        if not capable_models:
            return None

        # Filter by function calling support if needed
        if request.functions:
            capable_models = [m for m in capable_models if m.supports_function_calling]

        if not capable_models:
            return None

        # Select based on strategy
        if strategy == SelectionStrategy.LOWEST_COST:
            return min(capable_models, key=lambda m: m.cost_per_1k_tokens)

        elif strategy == SelectionStrategy.FASTEST:
            return min(
                capable_models,
                key=lambda m: self._model_performance[m.model_id]["avg_latency"]
            )

        elif strategy == SelectionStrategy.HIGHEST_QUALITY:
            return max(
                capable_models,
                key=lambda m: self._model_performance[m.model_id]["quality_score"]
            )

        elif strategy == SelectionStrategy.BALANCED:
            return self._balanced_selection(capable_models)

        elif strategy == SelectionStrategy.ADAPTIVE:
            return self._adaptive_selection(capable_models, request)

        return capable_models[0]

    def _balanced_selection(self, models: List[ModelConfig]) -> ModelConfig:
        """Select model with balanced cost/quality/latency"""
        scores = []

        for model in models:
            perf = self._model_performance[model.model_id]

            # Normalize metrics (0-1)
            cost_score = 1.0 / (model.cost_per_1k_tokens + 0.001)
            latency_score = 1.0 / (perf["avg_latency"] + 1)
            quality_score = perf["quality_score"] / 100.0
            success_score = perf["success_rate"]

            # Weighted combination
            composite_score = (
                0.25 * cost_score +
                0.25 * latency_score +
                0.35 * quality_score +
                0.15 * success_score
            )

            scores.append((model, composite_score))

        return max(scores, key=lambda x: x[1])[0]

    def _adaptive_selection(
        self,
        models: List[ModelConfig],
        request: ModelRequest
    ) -> ModelConfig:
        """
        Adaptive model selection based on:
        - Request complexity
        - Historical performance
        - Time of day
        - User preferences
        """
        # Estimate request complexity
        complexity = self._estimate_complexity(request)

        scores = []
        for model in models:
            perf = self._model_performance[model.model_id]

            # Match model capability to request complexity
            if complexity > 0.7 and model.quality_score < 80:
                # Complex request needs high-quality model
                capability_match = 0.3
            elif complexity < 0.3 and model.cost_per_1k_tokens > 0.01:
                # Simple request shouldn't use expensive model
                capability_match = 0.5
            else:
                capability_match = 1.0

            # Calculate adaptive score
            quality_weight = complexity * 0.5
            cost_weight = (1 - complexity) * 0.3
            latency_weight = 0.2

            score = (
                quality_weight * (model.quality_score / 100.0) +
                cost_weight * (1.0 / (model.cost_per_1k_tokens + 0.001)) +
                latency_weight * (1.0 / (perf["avg_latency"] + 1))
            ) * capability_match * perf["success_rate"]

            scores.append((model, score))

        return max(scores, key=lambda x: x[1])[0]

    def _estimate_complexity(self, request: ModelRequest) -> float:
        """Estimate request complexity (0-1)"""
        complexity = 0.5  # baseline

        # Adjust based on prompt length
        if len(request.prompt) > 2000:
            complexity += 0.2
        elif len(request.prompt) > 1000:
            complexity += 0.1

        # Adjust based on capability
        complex_capabilities = [
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.VISION
        ]
        if request.capability in complex_capabilities:
            complexity += 0.2

        # Adjust based on function calling
        if request.functions:
            complexity += 0.15

        return min(complexity, 1.0)

    def _get_fallback_model(
        self,
        failed_model: ModelConfig,
        request: ModelRequest
    ) -> Optional[ModelConfig]:
        """Get fallback model when primary fails"""
        # Get all capable models except the failed one
        fallback_models = [
            m for m in self.models.values()
            if request.capability in m.capabilities and m.model_id != failed_model.model_id
        ]

        if not fallback_models:
            return None

        # Select based on success rate
        return max(
            fallback_models,
            key=lambda m: self._model_performance[m.model_id]["success_rate"]
        )

    async def _execute_request(
        self,
        model: ModelConfig,
        request: ModelRequest
    ) -> ModelResponse:
        """Execute request to AI model"""
        start_time = time.time()

        # Simulate API call (would integrate with actual providers)
        await asyncio.sleep(model.avg_latency_ms / 1000)

        # Calculate token usage (rough estimate)
        tokens_used = len(request.prompt.split()) * 1.3 + (request.max_tokens or 100)

        # Calculate cost
        cost = (tokens_used / 1000) * model.cost_per_1k_tokens

        latency_ms = (time.time() - start_time) * 1000

        response = ModelResponse(
            content=f"Generated response from {model.model_id}",
            model_id=model.model_id,
            provider=model.provider,
            tokens_used=int(tokens_used),
            latency_ms=latency_ms,
            cost_usd=cost,
            quality_estimate=model.quality_score,
            metadata={
                "capability": request.capability.value,
                "temperature": request.temperature
            }
        )

        # Track budget
        user_id = request.user_id or "default"
        self._budget_tracker[user_id] = self._budget_tracker.get(user_id, 0) + cost

        return response

    def _update_metrics(
        self,
        model_id: str,
        response: Optional[ModelResponse],
        success: bool
    ):
        """Update performance metrics"""
        perf = self._model_performance[model_id]
        perf["total_requests"] += 1

        # Update success rate (exponential moving average)
        alpha = 0.1
        perf["success_rate"] = alpha * (1 if success else 0) + (1 - alpha) * perf["success_rate"]

        if response:
            # Update latency
            perf["avg_latency"] = (
                alpha * response.latency_ms + (1 - alpha) * perf["avg_latency"]
            )

            # Update cost
            perf["avg_cost"] = (
                alpha * response.cost_usd + (1 - alpha) * perf["avg_cost"]
            )

    def _majority_vote(self, responses: List[ModelResponse]) -> ModelResponse:
        """Select response with majority agreement"""
        # Simple implementation - would use semantic similarity in production
        content_votes = {}
        for response in responses:
            content_votes[response.content] = content_votes.get(response.content, 0) + 1

        majority_content = max(content_votes.items(), key=lambda x: x[1])[0]

        # Return first response with majority content
        for response in responses:
            if response.content == majority_content:
                return response

        return responses[0]

    def _weighted_vote(self, responses: List[ModelResponse]) -> ModelResponse:
        """Select response with weighted voting by quality"""
        # Weight responses by quality score
        weighted_responses = sorted(
            responses,
            key=lambda r: r.quality_estimate,
            reverse=True
        )

        return weighted_responses[0]

    def _unanimous_vote(self, responses: List[ModelResponse]) -> ModelResponse:
        """Require unanimous agreement or fallback"""
        # Check if all responses agree (simplified)
        if len(set(r.content for r in responses)) == 1:
            return responses[0]

        # If no unanimous agreement, use highest quality
        return self._weighted_vote(responses)

    def _get_cache_key(self, request: ModelRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            "prompt": request.prompt,
            "capability": request.capability.value,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }
        return json.dumps(key_data, sort_keys=True)

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics"""
        return {
            "registered_models": len(self.models),
            "model_performance": self._model_performance,
            "cache_size": len(self._prompt_cache),
            "budget_tracking": self._budget_tracker,
            "total_requests": sum(
                perf["total_requests"]
                for perf in self._model_performance.values()
            )
        }

    def optimize_costs(self, target_reduction: float = 0.2) -> Dict[str, Any]:
        """
        Analyze usage and provide cost optimization recommendations

        Args:
            target_reduction: Target cost reduction (0-1)

        Returns:
            Optimization recommendations
        """
        recommendations = []

        # Analyze model usage patterns
        for model_id, perf in self._model_performance.items():
            model = self.models[model_id]

            # Check if high-cost model is used for simple tasks
            if model.cost_per_1k_tokens > 0.01 and perf["total_requests"] > 100:
                recommendations.append({
                    "type": "model_downgrade",
                    "model": model_id,
                    "suggestion": "Consider using lower-cost model for routine tasks",
                    "potential_savings": perf["total_requests"] * perf["avg_cost"] * 0.5
                })

        # Check cache hit rate
        cache_hit_rate = 0.3  # Would calculate actual rate
        if cache_hit_rate < 0.5:
            recommendations.append({
                "type": "cache_optimization",
                "suggestion": "Increase prompt caching to reduce redundant API calls",
                "potential_savings": sum(self._budget_tracker.values()) * 0.2
            })

        return {
            "recommendations": recommendations,
            "current_spend": sum(self._budget_tracker.values()),
            "target_spend": sum(self._budget_tracker.values()) * (1 - target_reduction)
        }
