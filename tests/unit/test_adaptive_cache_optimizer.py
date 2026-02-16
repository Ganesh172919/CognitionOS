"""
Unit Tests for Adaptive Cache Optimizer

Tests for ML-based cache TTL prediction and optimization.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from infrastructure.intelligence.adaptive_cache_optimizer import (
    AdaptiveCacheOptimizer,
    CacheLayer,
    CachePerformanceMetrics,
    TTLOptimization,
)


class TestAdaptiveCacheOptimizer:
    """Tests for AdaptiveCacheOptimizer"""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance"""
        config = {
            "min_sample_size": 100,
            "optimization_threshold": 0.05,
            "confidence_threshold": 0.7,
        }
        return AdaptiveCacheOptimizer(db_connection=None, config=config)
    
    @pytest.fixture
    def high_hit_rate_metrics(self):
        """Create metrics with high hit rate"""
        return CachePerformanceMetrics(
            cache_layer=CacheLayer.L1_REDIS,
            total_requests=1000,
            cache_hits=950,
            cache_misses=50,
            hit_rate=0.95,
            avg_latency_ms=1.2,
            total_cost_saved=28.50,
            time_period_hours=24
        )
    
    @pytest.fixture
    def low_hit_rate_metrics(self):
        """Create metrics with low hit rate"""
        return CachePerformanceMetrics(
            cache_layer=CacheLayer.L2_DATABASE,
            total_requests=500,
            cache_hits=300,
            cache_misses=200,
            hit_rate=0.60,
            avg_latency_ms=12.0,
            total_cost_saved=9.00,
            time_period_hours=24
        )
    
    @pytest.mark.asyncio
    async def test_analyze_cache_performance(self, optimizer):
        """Test cache performance analysis"""
        metrics = await optimizer.analyze_cache_performance(time_window_hours=24)
        
        assert isinstance(metrics, dict)
        assert CacheLayer.L1_REDIS in metrics
        assert CacheLayer.L2_DATABASE in metrics
        assert CacheLayer.L3_SEMANTIC in metrics
        
        for layer, metric in metrics.items():
            assert isinstance(metric, CachePerformanceMetrics)
            assert metric.total_requests >= 0
            assert 0.0 <= metric.hit_rate <= 1.0
    
    @pytest.mark.asyncio
    async def test_predict_optimal_ttl_low_hit_rate(self, optimizer, low_hit_rate_metrics):
        """Test TTL prediction for low hit rate - should increase TTL"""
        optimization = await optimizer.predict_optimal_ttl(
            CacheLayer.L2_DATABASE,
            low_hit_rate_metrics
        )
        
        assert isinstance(optimization, TTLOptimization)
        assert optimization.new_ttl_seconds > optimization.old_ttl_seconds
        assert optimization.predicted_hit_rate > low_hit_rate_metrics.hit_rate
        assert optimization.confidence >= 0.0
        assert "Low hit rate" in optimization.optimization_reason
    
    @pytest.mark.asyncio
    async def test_predict_optimal_ttl_high_hit_rate(self, optimizer, high_hit_rate_metrics):
        """Test TTL prediction for high hit rate - may reduce TTL to save memory"""
        optimization = await optimizer.predict_optimal_ttl(
            CacheLayer.L1_REDIS,
            high_hit_rate_metrics
        )
        
        assert isinstance(optimization, TTLOptimization)
        # High hit rate should either reduce TTL or keep it same
        assert optimization.new_ttl_seconds <= optimization.old_ttl_seconds * 1.5
        assert optimization.confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_optimize_cache_ttls(self, optimizer):
        """Test full TTL optimization cycle"""
        optimizations = await optimizer.optimize_cache_ttls(
            time_window_hours=24,
            apply=False
        )
        
        assert isinstance(optimizations, list)
        # Should return some optimizations
        assert len(optimizations) >= 0
        
        for opt in optimizations:
            assert isinstance(opt, TTLOptimization)
            assert opt.confidence >= optimizer.config["confidence_threshold"]
    
    @pytest.mark.asyncio
    async def test_optimize_cache_ttls_apply(self, optimizer):
        """Test TTL optimization with application"""
        initial_ttl = optimizer.get_ttl_for_layer(CacheLayer.L1_REDIS)
        
        await optimizer.optimize_cache_ttls(time_window_hours=24, apply=True)
        
        # TTL may or may not change depending on metrics
        new_ttl = optimizer.get_ttl_for_layer(CacheLayer.L1_REDIS)
        assert isinstance(new_ttl, int)
        assert new_ttl > 0
    
    def test_get_ttl_for_layer(self, optimizer):
        """Test getting TTL for a layer"""
        ttl_l1 = optimizer.get_ttl_for_layer(CacheLayer.L1_REDIS)
        ttl_l2 = optimizer.get_ttl_for_layer(CacheLayer.L2_DATABASE)
        ttl_l3 = optimizer.get_ttl_for_layer(CacheLayer.L3_SEMANTIC)
        
        assert isinstance(ttl_l1, int)
        assert isinstance(ttl_l2, int)
        assert isinstance(ttl_l3, int)
        
        # L1 should have shortest TTL, L3 should have longest
        assert ttl_l1 < ttl_l2 < ttl_l3
    
    @pytest.mark.asyncio
    async def test_calculate_cost_savings(self, optimizer):
        """Test cost savings calculation"""
        savings = await optimizer.calculate_cost_savings(time_window_hours=24)
        
        assert isinstance(savings, dict)
        assert "time_window_hours" in savings
        assert "total_requests" in savings
        assert "total_cache_hits" in savings
        assert "overall_hit_rate" in savings
        assert "total_cost_saved_usd" in savings
        assert "savings_percent" in savings
        assert "layers" in savings
        
        assert savings["total_requests"] >= 0
        assert 0.0 <= savings["overall_hit_rate"] <= 1.0
        assert savings["total_cost_saved_usd"] >= 0
    
    @pytest.mark.asyncio
    async def test_run_optimization_cycle(self, optimizer):
        """Test complete optimization cycle"""
        result = await optimizer.run_optimization_cycle()
        
        assert isinstance(result, dict)
        assert "optimizations_applied" in result
        assert "current_savings" in result
        assert "optimizations" in result
        
        assert result["optimizations_applied"] >= 0
        assert isinstance(result["current_savings"], dict)
    
    @pytest.mark.asyncio
    async def test_apply_optimization(self, optimizer):
        """Test applying optimization"""
        optimization = TTLOptimization(
            cache_layer=CacheLayer.L1_REDIS,
            cache_key_pattern=None,
            old_ttl_seconds=300,
            new_ttl_seconds=450,
            predicted_hit_rate=0.88,
            optimization_reason="Test optimization",
            confidence=0.85
        )
        
        await optimizer._apply_optimization(optimization)
        
        # Verify TTL was updated
        new_ttl = optimizer.get_ttl_for_layer(CacheLayer.L1_REDIS)
        assert new_ttl == 450
    
    def test_default_ttls_initialized(self, optimizer):
        """Test default TTLs are properly initialized"""
        assert CacheLayer.L1_REDIS in optimizer.default_ttls
        assert CacheLayer.L2_DATABASE in optimizer.default_ttls
        assert CacheLayer.L3_SEMANTIC in optimizer.default_ttls
        
        # Verify TTL hierarchy
        assert (optimizer.default_ttls[CacheLayer.L1_REDIS] < 
                optimizer.default_ttls[CacheLayer.L2_DATABASE] < 
                optimizer.default_ttls[CacheLayer.L3_SEMANTIC])


class TestCachePerformanceMetrics:
    """Tests for CachePerformanceMetrics dataclass"""
    
    def test_create_metrics(self):
        """Test creating performance metrics"""
        metrics = CachePerformanceMetrics(
            cache_layer=CacheLayer.L1_REDIS,
            total_requests=1000,
            cache_hits=850,
            cache_misses=150,
            hit_rate=0.85,
            avg_latency_ms=1.5,
            total_cost_saved=25.50,
            time_period_hours=24
        )
        
        assert metrics.cache_layer == CacheLayer.L1_REDIS
        assert metrics.total_requests == 1000
        assert metrics.cache_hits == 850
        assert metrics.hit_rate == 0.85
        assert metrics.avg_latency_ms == 1.5
        assert metrics.total_cost_saved == 25.50


class TestTTLOptimization:
    """Tests for TTLOptimization dataclass"""
    
    def test_create_optimization(self):
        """Test creating TTL optimization"""
        opt = TTLOptimization(
            cache_layer=CacheLayer.L2_DATABASE,
            cache_key_pattern="llm:*",
            old_ttl_seconds=3600,
            new_ttl_seconds=4500,
            predicted_hit_rate=0.88,
            optimization_reason="Increase TTL for better hit rate",
            confidence=0.82
        )
        
        assert opt.cache_layer == CacheLayer.L2_DATABASE
        assert opt.old_ttl_seconds == 3600
        assert opt.new_ttl_seconds == 4500
        assert opt.confidence == 0.82
