"""
Unit Tests for Intelligent Model Router

Tests for cost-performance aware model selection and routing.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from infrastructure.intelligence.intelligent_router import (
    IntelligentModelRouter,
    ModelTier,
    TaskComplexity,
    ModelCandidate,
    RoutingDecision,
)


class TestIntelligentModelRouter:
    """Tests for IntelligentModelRouter"""
    
    @pytest.fixture
    def router(self):
        """Create router instance"""
        config = {
            "default_model": "gpt-3.5-turbo",
            "basic_threshold": 0.3,
            "advanced_threshold": 0.7,
            "max_cost_per_request": 0.10,
        }
        return IntelligentModelRouter(db_connection=None, config=config)
    
    @pytest.mark.asyncio
    async def test_classify_simple_task(self, router):
        """Test classification of simple task"""
        complexity = await router.classify_task_complexity(
            task_type="simple_qa",
            task_description="What is the capital of France?"
        )
        
        assert isinstance(complexity, TaskComplexity)
        assert 0.0 <= complexity.score <= 1.0
        assert complexity.score < 0.5  # Should be classified as simple
        assert "task_type" in complexity.factors
        assert len(complexity.reasoning) > 0
    
    @pytest.mark.asyncio
    async def test_classify_complex_task(self, router):
        """Test classification of complex task"""
        complexity = await router.classify_task_complexity(
            task_type="complex_reasoning",
            task_description="Design a multi-step algorithm to optimize complex distributed system performance"
        )
        
        assert isinstance(complexity, TaskComplexity)
        assert complexity.score > 0.5  # Should be classified as complex
        assert "task_type" in complexity.factors
        assert "description" in complexity.factors
    
    @pytest.mark.asyncio
    async def test_classify_code_generation_task(self, router):
        """Test classification of code generation task"""
        complexity = await router.classify_task_complexity(
            task_type="code_generation",
            task_description="Generate Python code for sorting algorithm"
        )
        
        assert isinstance(complexity, TaskComplexity)
        assert 0.4 <= complexity.score <= 1.0  # Code gen should be medium to high
    
    def test_analyze_description_complexity_simple(self, router):
        """Test description analysis for simple text"""
        score = router._analyze_description_complexity("Simple basic task")
        
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Simple keywords should reduce score
    
    def test_analyze_description_complexity_complex(self, router):
        """Test description analysis for complex text"""
        description = "Complex multi-step advanced optimization requiring detailed analysis"
        score = router._analyze_description_complexity(description)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.4  # Complex keywords should increase score
    
    def test_analyze_context_complexity(self, router):
        """Test context complexity analysis"""
        context = {
            "input_size": 15000,
            "require_high_accuracy": True,
            "constraints": ["real-time", "low-latency"]
        }
        
        score = router._analyze_context_complexity(context)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Complex context should increase score
    
    @pytest.mark.asyncio
    async def test_select_optimal_model_simple_task(self, router):
        """Test model selection for simple task"""
        decision = await router.select_optimal_model(
            task_type="simple_qa",
            task_description="What is 2+2?",
            context={"task_id": "test-1"}
        )
        
        assert isinstance(decision, RoutingDecision)
        assert decision.selected_model == "gpt-3.5-turbo"  # Should use cheapest for simple
        assert decision.complexity.score < 0.5
        assert decision.confidence >= 0.0
        assert len(decision.available_models) > 0
    
    @pytest.mark.asyncio
    async def test_select_optimal_model_complex_task(self, router):
        """Test model selection for complex task"""
        decision = await router.select_optimal_model(
            task_type="complex_reasoning",
            task_description="Design advanced multi-step distributed system architecture",
            context={"task_id": "test-2"}
        )
        
        assert isinstance(decision, RoutingDecision)
        # Should use high-quality model for complex tasks
        assert decision.selected_model in ["gpt-4", "gpt-4-turbo", "claude-3-opus"]
        assert decision.complexity.score > 0.7
        assert decision.confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_select_optimal_model_with_budget(self, router):
        """Test model selection with budget constraint"""
        decision = await router.select_optimal_model(
            task_type="code_generation",
            task_description="Generate code",
            context={"task_id": "test-3"},
            max_cost=0.01  # Very low budget
        )
        
        assert isinstance(decision, RoutingDecision)
        # Should respect budget constraint
        assert decision.predicted_cost <= 0.01
    
    @pytest.mark.asyncio
    async def test_get_candidate_models(self, router):
        """Test getting candidate models"""
        complexity = TaskComplexity(
            score=0.5,
            factors={"task_type": 0.5},
            reasoning="Medium complexity"
        )
        
        candidates = await router._get_candidate_models(
            task_type="code_generation",
            complexity=complexity,
            max_cost=None
        )
        
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        
        for candidate in candidates:
            assert isinstance(candidate, ModelCandidate)
            assert candidate.model_name in router.model_catalog
            assert 0.0 <= candidate.success_rate <= 1.0
    
    @pytest.mark.asyncio
    async def test_get_candidate_models_with_budget(self, router):
        """Test getting candidate models with budget constraint"""
        complexity = TaskComplexity(
            score=0.5,
            factors={"task_type": 0.5},
            reasoning="Medium complexity"
        )
        
        candidates = await router._get_candidate_models(
            task_type="code_generation",
            complexity=complexity,
            max_cost=0.01  # Very low budget
        )
        
        # Should filter out expensive models
        for candidate in candidates:
            assert candidate.estimated_cost <= 0.01
    
    def test_select_from_candidates_low_complexity(self, router):
        """Test candidate selection for low complexity"""
        complexity = TaskComplexity(
            score=0.2,
            factors={"task_type": 0.2},
            reasoning="Low complexity"
        )
        
        candidates = [
            ModelCandidate("gpt-3.5-turbo", ModelTier.BASIC, 0.004, 0.75, 800, 0.80),
            ModelCandidate("gpt-4", ModelTier.ADVANCED, 0.06, 0.95, 1500, 0.92),
        ]
        
        model, reason, confidence = router._select_from_candidates(complexity, candidates, None)
        
        assert model == "gpt-3.5-turbo"  # Should pick cheapest
        assert "Low complexity" in reason
        assert confidence >= 0.0
    
    def test_select_from_candidates_high_complexity(self, router):
        """Test candidate selection for high complexity"""
        complexity = TaskComplexity(
            score=0.85,
            factors={"task_type": 0.85},
            reasoning="High complexity"
        )
        
        candidates = [
            ModelCandidate("gpt-3.5-turbo", ModelTier.BASIC, 0.004, 0.75, 800, 0.80),
            ModelCandidate("gpt-4", ModelTier.ADVANCED, 0.06, 0.95, 1500, 0.92),
        ]
        
        model, reason, confidence = router._select_from_candidates(complexity, candidates, None)
        
        assert model == "gpt-4"  # Should pick highest quality
        assert "High complexity" in reason
    
    def test_select_from_candidates_medium_complexity(self, router):
        """Test candidate selection for medium complexity"""
        complexity = TaskComplexity(
            score=0.5,
            factors={"task_type": 0.5},
            reasoning="Medium complexity"
        )
        
        candidates = [
            ModelCandidate("gpt-3.5-turbo", ModelTier.BASIC, 0.004, 0.75, 800, 0.80),
            ModelCandidate("gpt-4-turbo", ModelTier.PREMIUM, 0.02, 0.97, 1200, 0.94),
        ]
        
        model, reason, confidence = router._select_from_candidates(complexity, candidates, None)
        
        # Should balance cost and quality
        assert model in ["gpt-3.5-turbo", "gpt-4-turbo"]
        assert "Medium complexity" in reason
    
    @pytest.mark.asyncio
    async def test_evaluate_routing_performance(self, router):
        """Test routing performance evaluation"""
        metrics = await router.evaluate_routing_performance(time_window_hours=24)
        
        assert isinstance(metrics, dict)
        assert "total_decisions" in metrics
        assert "optimal_selections" in metrics
        assert "optimal_rate" in metrics
        assert "avg_cost_per_request" in metrics
        assert "models_used" in metrics
        
        assert 0.0 <= metrics["optimal_rate"] <= 1.0
    
    def test_get_model_recommendation_simple(self, router):
        """Test quick recommendation for simple task"""
        model = router.get_model_recommendation("simple_qa")
        
        assert model == "gpt-3.5-turbo"
    
    def test_get_model_recommendation_complex(self, router):
        """Test quick recommendation for complex task"""
        model = router.get_model_recommendation("complex_reasoning")
        
        assert model in ["gpt-4", "gpt-3.5-turbo"]  # Depends on budget
    
    def test_get_model_recommendation_with_budget(self, router):
        """Test quick recommendation with budget constraint"""
        model = router.get_model_recommendation("complex_reasoning", budget_constraint=0.01)
        
        # Should use cheaper model due to budget
        assert model == "gpt-3.5-turbo"
    
    def test_model_catalog_initialized(self, router):
        """Test model catalog is properly initialized"""
        assert len(router.model_catalog) > 0
        
        # Verify GPT models are present
        assert "gpt-3.5-turbo" in router.model_catalog
        assert "gpt-4" in router.model_catalog
        
        # Verify catalog structure
        for model_name, model_info in router.model_catalog.items():
            assert "tier" in model_info
            assert "cost_per_1k_tokens" in model_info
            assert "avg_latency_ms" in model_info
            assert "quality_score" in model_info
            assert 0.0 <= model_info["quality_score"] <= 1.0


class TestTaskComplexity:
    """Tests for TaskComplexity dataclass"""
    
    def test_create_complexity(self):
        """Test creating task complexity"""
        complexity = TaskComplexity(
            score=0.75,
            factors={"task_type": 0.8, "description": 0.7},
            reasoning="High complexity due to advanced requirements"
        )
        
        assert complexity.score == 0.75
        assert "task_type" in complexity.factors
        assert len(complexity.reasoning) > 0


class TestModelCandidate:
    """Tests for ModelCandidate dataclass"""
    
    def test_create_candidate(self):
        """Test creating model candidate"""
        candidate = ModelCandidate(
            model_name="gpt-4",
            tier=ModelTier.ADVANCED,
            estimated_cost=0.06,
            estimated_quality=0.95,
            avg_latency_ms=1500,
            success_rate=0.92
        )
        
        assert candidate.model_name == "gpt-4"
        assert candidate.tier == ModelTier.ADVANCED
        assert candidate.estimated_cost == 0.06
        assert 0.0 <= candidate.success_rate <= 1.0


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass"""
    
    def test_create_decision(self):
        """Test creating routing decision"""
        complexity = TaskComplexity(
            score=0.6,
            factors={"task_type": 0.6},
            reasoning="Medium complexity"
        )
        
        decision = RoutingDecision(
            task_id="test-123",
            task_type="code_generation",
            complexity=complexity,
            available_models=[],
            selected_model="gpt-4",
            selection_reason="Best for code generation",
            confidence=0.85,
            predicted_cost=0.05,
            predicted_quality=0.92
        )
        
        assert decision.task_id == "test-123"
        assert decision.selected_model == "gpt-4"
        assert decision.confidence == 0.85
        assert decision.predicted_cost == 0.05
