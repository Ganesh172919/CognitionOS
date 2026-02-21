"""
Intelligent LLM Router

Routes LLM requests to the optimal provider/model based on:
- Task type and complexity
- Cost constraints and budget
- Latency requirements
- Provider health and availability
- Historical performance metrics
- Token efficiency optimization
"""

import asyncio
import hashlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from infrastructure.llm.provider import LLMProvider, LLMRequest, LLMResponse


class RoutingStrategy(str, Enum):
    """Routing decision strategy"""
    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    BALANCED = "balanced"
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"


class TaskComplexity(str, Enum):
    """Task complexity classification"""
    TRIVIAL = "trivial"       # Simple Q&A, classification
    SIMPLE = "simple"         # Short-form generation, summarization
    MODERATE = "moderate"     # Code generation, analysis
    COMPLEX = "complex"       # Multi-step reasoning, long-form
    EXPERT = "expert"         # Advanced research, system design


@dataclass
class ModelProfile:
    """Profile of an LLM model"""
    provider: LLMProvider
    model_id: str
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    max_context_tokens: int
    avg_latency_ms: float
    quality_score: float           # 0.0–1.0 normalized quality
    supports_function_calling: bool = False
    supports_streaming: bool = True
    max_concurrent_requests: int = 20
    min_complexity: TaskComplexity = TaskComplexity.TRIVIAL
    max_complexity: TaskComplexity = TaskComplexity.EXPERT
    tags: List[str] = field(default_factory=list)


@dataclass
class ProviderHealth:
    """Live health status of a provider"""
    provider: LLMProvider
    model_id: str
    is_healthy: bool = True
    consecutive_failures: int = 0
    last_failure_at: Optional[float] = None
    last_success_at: Optional[float] = None
    circuit_open: bool = False
    circuit_open_until: Optional[float] = None
    error_rate_window: Deque[bool] = field(default_factory=lambda: deque(maxlen=50))
    latency_window: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    active_requests: int = 0

    @property
    def error_rate(self) -> float:
        if not self.error_rate_window:
            return 0.0
        return sum(1 for e in self.error_rate_window if not e) / len(self.error_rate_window)

    @property
    def p95_latency_ms(self) -> float:
        if not self.latency_window:
            return 0.0
        sorted_latencies = sorted(self.latency_window)
        idx = max(0, int(len(sorted_latencies) * 0.95) - 1)
        return sorted_latencies[idx]

    def record_success(self, latency_ms: float) -> None:
        self.is_healthy = True
        self.consecutive_failures = 0
        self.last_success_at = time.time()
        self.error_rate_window.append(True)
        self.latency_window.append(latency_ms)

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.last_failure_at = time.time()
        self.error_rate_window.append(False)
        if self.consecutive_failures >= 5 or self.error_rate >= 0.5:
            self.is_healthy = False
            self.circuit_open = True
            self.circuit_open_until = time.time() + 60  # 60s circuit open

    def check_circuit(self) -> bool:
        """Returns True if circuit is closed (healthy for requests)"""
        if self.circuit_open:
            if self.circuit_open_until and time.time() > self.circuit_open_until:
                self.circuit_open = False
                self.consecutive_failures = 0
                return True
            return False
        return True


@dataclass
class RoutingDecision:
    """Routing decision with reasoning"""
    model_profile: ModelProfile
    strategy_used: RoutingStrategy
    estimated_cost_usd: float
    estimated_latency_ms: float
    confidence: float
    fallback_models: List[str]
    reasoning: str


@dataclass
class RouterMetrics:
    """Aggregate router metrics"""
    total_requests: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    provider_breakdown: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    model_breakdown: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    strategy_breakdown: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    avg_latency_ms: float = 0.0
    cache_hits: int = 0
    fallback_activations: int = 0
    circuit_breaks: int = 0
    cost_savings_usd: float = 0.0


# Pre-defined model catalog
DEFAULT_MODEL_CATALOG: Dict[str, ModelProfile] = {
    "gpt-4o-mini": ModelProfile(
        provider=LLMProvider.OPENAI,
        model_id="gpt-4o-mini",
        cost_per_1k_input_tokens=0.00015,
        cost_per_1k_output_tokens=0.0006,
        max_context_tokens=128000,
        avg_latency_ms=800,
        quality_score=0.82,
        supports_function_calling=True,
        supports_streaming=True,
        max_concurrent_requests=100,
        min_complexity=TaskComplexity.TRIVIAL,
        max_complexity=TaskComplexity.MODERATE,
        tags=["fast", "cheap", "general"],
    ),
    "gpt-4o": ModelProfile(
        provider=LLMProvider.OPENAI,
        model_id="gpt-4o",
        cost_per_1k_input_tokens=0.005,
        cost_per_1k_output_tokens=0.015,
        max_context_tokens=128000,
        avg_latency_ms=1800,
        quality_score=0.96,
        supports_function_calling=True,
        supports_streaming=True,
        max_concurrent_requests=20,
        min_complexity=TaskComplexity.MODERATE,
        max_complexity=TaskComplexity.EXPERT,
        tags=["high-quality", "reasoning", "code"],
    ),
    "claude-3-haiku-20240307": ModelProfile(
        provider=LLMProvider.ANTHROPIC,
        model_id="claude-3-haiku-20240307",
        cost_per_1k_input_tokens=0.00025,
        cost_per_1k_output_tokens=0.00125,
        max_context_tokens=200000,
        avg_latency_ms=600,
        quality_score=0.84,
        supports_function_calling=True,
        supports_streaming=True,
        max_concurrent_requests=100,
        min_complexity=TaskComplexity.TRIVIAL,
        max_complexity=TaskComplexity.MODERATE,
        tags=["fast", "cheap", "long-context"],
    ),
    "claude-3-5-sonnet-20241022": ModelProfile(
        provider=LLMProvider.ANTHROPIC,
        model_id="claude-3-5-sonnet-20241022",
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.015,
        max_context_tokens=200000,
        avg_latency_ms=2200,
        quality_score=0.97,
        supports_function_calling=True,
        supports_streaming=True,
        max_concurrent_requests=20,
        min_complexity=TaskComplexity.MODERATE,
        max_complexity=TaskComplexity.EXPERT,
        tags=["high-quality", "reasoning", "code", "analysis"],
    ),
}


class TaskComplexityClassifier:
    """Classifies task complexity from request context"""

    COMPLEX_KEYWORDS = {
        "design", "architect", "system", "analyze", "compare", "evaluate",
        "explain", "research", "implement", "refactor", "optimize",
    }
    EXPERT_KEYWORDS = {
        "enterprise", "production", "scalab", "distributed", "microservice",
        "algorithm", "proof", "theorem", "scientific",
    }
    TRIVIAL_KEYWORDS = {
        "what is", "define", "list", "name", "who", "when", "where",
    }

    def classify(self, request: LLMRequest) -> TaskComplexity:
        """Classify task complexity from request messages"""
        full_text = " ".join(
            msg.get("content", "") for msg in request.messages
        ).lower()

        token_estimate = sum(
            len(msg.get("content", "").split()) for msg in request.messages
        ) * 1.3  # rough tokens

        if any(kw in full_text for kw in self.EXPERT_KEYWORDS) or token_estimate > 3000:
            return TaskComplexity.EXPERT
        if any(kw in full_text for kw in self.COMPLEX_KEYWORDS) or token_estimate > 1000:
            return TaskComplexity.COMPLEX
        if any(kw in full_text for kw in self.TRIVIAL_KEYWORDS) or token_estimate < 50:
            return TaskComplexity.TRIVIAL
        if token_estimate < 200:
            return TaskComplexity.SIMPLE
        return TaskComplexity.MODERATE


class CostEstimator:
    """Estimates cost before making LLM request"""

    def estimate(
        self,
        profile: ModelProfile,
        request: LLMRequest,
    ) -> float:
        """Estimate total cost in USD"""
        input_tokens = sum(
            len(msg.get("content", "").split()) * 1.3
            for msg in request.messages
        )
        output_tokens = request.max_tokens * 0.5  # assume ~50% utilization
        cost = (
            (input_tokens / 1000) * profile.cost_per_1k_input_tokens
            + (output_tokens / 1000) * profile.cost_per_1k_output_tokens
        )
        return round(cost, 6)


class IntelligentLLMRouter:
    """
    Routes LLM requests to the optimal model/provider.

    Features:
    - Multi-model support (OpenAI, Anthropic, local)
    - Circuit breaker per model
    - Cost-aware routing
    - Latency-aware routing
    - Task complexity classification
    - Automatic fallback chain
    - Real-time health tracking
    - Budget enforcement
    """

    def __init__(
        self,
        model_catalog: Optional[Dict[str, ModelProfile]] = None,
        default_strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        max_cost_per_request_usd: float = 1.0,
        provider_factory: Optional[Callable[[LLMProvider, str], Any]] = None,
    ):
        self.catalog = model_catalog or DEFAULT_MODEL_CATALOG
        self.default_strategy = default_strategy
        self.max_cost_per_request = max_cost_per_request_usd
        self.provider_factory = provider_factory
        self._health: Dict[str, ProviderHealth] = {
            model_id: ProviderHealth(
                provider=profile.provider,
                model_id=model_id,
            )
            for model_id, profile in self.catalog.items()
        }
        self._metrics = RouterMetrics()
        self._classifier = TaskComplexityClassifier()
        self._cost_estimator = CostEstimator()
        self._round_robin_index: int = 0
        self._providers: Dict[str, Any] = {}

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    async def route(
        self,
        request: LLMRequest,
        strategy: Optional[RoutingStrategy] = None,
        required_tags: Optional[List[str]] = None,
        budget_usd: Optional[float] = None,
        tenant_id: Optional[str] = None,
    ) -> LLMResponse:
        """
        Route request to the best available model and return response.
        Tries fallback chain on failure.
        """
        effective_strategy = strategy or self.default_strategy
        decision = self._decide(request, effective_strategy, required_tags, budget_usd)

        self._metrics.total_requests += 1
        self._metrics.strategy_breakdown[effective_strategy] += 1

        errors: List[str] = []
        candidates = [decision.model_profile.model_id] + decision.fallback_models

        for model_id in candidates:
            profile = self.catalog.get(model_id)
            if profile is None:
                continue
            health = self._health[model_id]
            if not health.check_circuit():
                self._metrics.circuit_breaks += 1
                errors.append(f"{model_id}: circuit open")
                continue
            if health.active_requests >= profile.max_concurrent_requests:
                errors.append(f"{model_id}: at capacity")
                continue

            try:
                response = await self._call_model(profile, request)
                health.record_success(response.latency_ms)
                self._update_metrics(response, effective_strategy)
                if model_id != decision.model_profile.model_id:
                    self._metrics.fallback_activations += 1
                return response
            except Exception as exc:  # noqa: BLE001
                health.record_failure()
                errors.append(f"{model_id}: {exc}")

        raise RuntimeError(
            f"All LLM models failed for request. Errors: {'; '.join(errors)}"
        )

    def decide(
        self,
        request: LLMRequest,
        strategy: Optional[RoutingStrategy] = None,
        required_tags: Optional[List[str]] = None,
        budget_usd: Optional[float] = None,
    ) -> RoutingDecision:
        """Return routing decision without executing the request"""
        return self._decide(
            request,
            strategy or self.default_strategy,
            required_tags,
            budget_usd,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Return current router metrics"""
        return {
            "total_requests": self._metrics.total_requests,
            "total_cost_usd": round(self._metrics.total_cost_usd, 4),
            "total_tokens": self._metrics.total_tokens,
            "avg_latency_ms": round(self._metrics.avg_latency_ms, 2),
            "cache_hits": self._metrics.cache_hits,
            "fallback_activations": self._metrics.fallback_activations,
            "circuit_breaks": self._metrics.circuit_breaks,
            "cost_savings_usd": round(self._metrics.cost_savings_usd, 4),
            "provider_breakdown": dict(self._metrics.provider_breakdown),
            "model_breakdown": dict(self._metrics.model_breakdown),
            "strategy_breakdown": dict(self._metrics.strategy_breakdown),
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Return health status of all models"""
        return {
            model_id: {
                "is_healthy": h.is_healthy,
                "circuit_open": h.circuit_open,
                "error_rate": round(h.error_rate, 3),
                "p95_latency_ms": round(h.p95_latency_ms, 1),
                "consecutive_failures": h.consecutive_failures,
                "active_requests": h.active_requests,
            }
            for model_id, h in self._health.items()
        }

    def register_provider(self, provider: LLMProvider, model_id: str, client: Any) -> None:
        """Register a concrete provider client"""
        key = f"{provider.value}:{model_id}"
        self._providers[key] = client

    # ──────────────────────────────────────────────
    # Internal routing logic
    # ──────────────────────────────────────────────

    def _decide(
        self,
        request: LLMRequest,
        strategy: RoutingStrategy,
        required_tags: Optional[List[str]],
        budget_usd: Optional[float],
    ) -> RoutingDecision:
        complexity = self._classifier.classify(request)
        candidates = self._get_candidates(complexity, required_tags, budget_usd, request)

        if not candidates:
            # fall back to all healthy models ignoring complexity filter
            candidates = [
                (mid, p) for mid, p in self.catalog.items()
                if self._health[mid].check_circuit()
            ]

        if not candidates:
            # Last resort – pick any model
            candidates = list(self.catalog.items())

        scored = self._score_candidates(candidates, strategy, request)
        scored.sort(key=lambda x: x[1], reverse=True)

        best_id, best_score = scored[0]
        best_profile = self.catalog[best_id]
        fallbacks = [mid for mid, _ in scored[1:4]]

        est_cost = self._cost_estimator.estimate(best_profile, request)
        est_latency = self._health[best_id].p95_latency_ms or best_profile.avg_latency_ms

        return RoutingDecision(
            model_profile=best_profile,
            strategy_used=strategy,
            estimated_cost_usd=est_cost,
            estimated_latency_ms=est_latency,
            confidence=min(1.0, best_score / 10.0),
            fallback_models=fallbacks,
            reasoning=(
                f"complexity={complexity.value}, strategy={strategy.value}, "
                f"score={best_score:.2f}, cost=${est_cost:.5f}"
            ),
        )

    def _get_candidates(
        self,
        complexity: TaskComplexity,
        required_tags: Optional[List[str]],
        budget_usd: Optional[float],
        request: LLMRequest,
    ) -> List[Tuple[str, ModelProfile]]:
        complexity_order = [
            TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE,
            TaskComplexity.MODERATE, TaskComplexity.COMPLEX, TaskComplexity.EXPERT,
        ]
        complexity_rank = complexity_order.index(complexity)

        result = []
        for mid, profile in self.catalog.items():
            min_rank = complexity_order.index(profile.min_complexity)
            max_rank = complexity_order.index(profile.max_complexity)
            if not (min_rank <= complexity_rank <= max_rank):
                continue
            if required_tags:
                if not all(t in profile.tags for t in required_tags):
                    continue
            if budget_usd is not None:
                est = self._cost_estimator.estimate(profile, request)
                if est > budget_usd:
                    continue
            if profile.max_context_tokens < sum(
                len(m.get("content", "").split()) * 1.3 for m in request.messages
            ):
                continue
            result.append((mid, profile))
        return result

    def _score_candidates(
        self,
        candidates: List[Tuple[str, ModelProfile]],
        strategy: RoutingStrategy,
        request: LLMRequest,
    ) -> List[Tuple[str, float]]:
        scored: List[Tuple[str, float]] = []
        for mid, profile in candidates:
            health = self._health[mid]
            if not health.check_circuit():
                continue

            cost = self._cost_estimator.estimate(profile, request)
            latency = health.p95_latency_ms or profile.avg_latency_ms
            quality = profile.quality_score
            error_penalty = health.error_rate * 5

            if strategy == RoutingStrategy.COST_OPTIMIZED:
                score = 10 - (cost * 1000) - error_penalty
            elif strategy == RoutingStrategy.LATENCY_OPTIMIZED:
                score = 10 - (latency / 500) - error_penalty
            elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
                score = quality * 10 - error_penalty
            elif strategy == RoutingStrategy.ROUND_ROBIN:
                model_ids = [c[0] for c in candidates]
                if mid == model_ids[self._round_robin_index % len(model_ids)]:
                    score = 10.0
                else:
                    score = 1.0
            elif strategy == RoutingStrategy.LEAST_LOADED:
                load = health.active_requests / max(profile.max_concurrent_requests, 1)
                score = 10 - (load * 10) - error_penalty
            else:  # BALANCED
                cost_norm = max(0, 10 - (cost * 500))
                latency_norm = max(0, 10 - (latency / 400))
                quality_norm = quality * 10
                score = (cost_norm * 0.35 + latency_norm * 0.30 + quality_norm * 0.35) - error_penalty

            scored.append((mid, score))

        if strategy == RoutingStrategy.ROUND_ROBIN:
            self._round_robin_index += 1

        return scored

    # ──────────────────────────────────────────────
    # Provider execution
    # ──────────────────────────────────────────────

    async def _call_model(
        self, profile: ModelProfile, request: LLMRequest
    ) -> LLMResponse:
        """Call the model via registered provider or factory"""
        health = self._health[profile.model_id]
        health.active_requests += 1
        start = time.time()
        try:
            provider_key = f"{profile.provider.value}:{profile.model_id}"
            provider = self._providers.get(provider_key)

            if provider is None and self.provider_factory:
                provider = self.provider_factory(profile.provider, profile.model_id)
                self._providers[provider_key] = provider

            if provider is None:
                raise RuntimeError(
                    f"No provider registered for {profile.provider.value}/{profile.model_id}. "
                    "Register a provider with router.register_provider()."
                )

            req_copy = LLMRequest(
                messages=request.messages,
                model=profile.model_id,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stream=request.stream,
                metadata=request.metadata,
            )
            response: LLMResponse = await provider.generate(req_copy)
            response.provider = profile.provider
            return response
        finally:
            health.active_requests = max(0, health.active_requests - 1)

    def _update_metrics(self, response: LLMResponse, strategy: RoutingStrategy) -> None:
        m = self._metrics
        m.total_cost_usd += response.cost_usd or 0.0
        m.total_tokens += sum(response.usage.values())
        m.provider_breakdown[response.provider.value] += 1
        m.model_breakdown[response.model] += 1
        n = m.total_requests
        m.avg_latency_ms = (m.avg_latency_ms * (n - 1) + response.latency_ms) / n
