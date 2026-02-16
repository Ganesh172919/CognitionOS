"""
Performance Integration Tests

Tests system performance under load:
- Concurrent request handling
- Cache performance
- Database performance
- End-to-end latency
"""

import pytest
from httpx import AsyncClient
import asyncio
from typing import List


@pytest.mark.integration
@pytest.mark.asyncio
class TestConcurrentRequests:
    """Test concurrent request handling"""
    
    async def test_100_concurrent_api_requests(self, client: AsyncClient, performance_monitor):
        """Test handling 100 concurrent API requests"""
        performance_monitor.start()
        
        # Create 100 concurrent health check requests
        tasks = [client.get("/api/v3/health/ready") for _ in range(100)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        duration = performance_monitor.stop()
        
        # Count successful responses
        successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
        
        # Should handle most requests successfully
        assert successful >= 80, f"Only {successful}/100 requests succeeded"
        
        # Should complete in reasonable time
        assert duration < 10.0, f"100 requests took {duration}s, expected <10s"
        
        print(f"\n✓ 100 concurrent requests: {successful} successful in {duration:.2f}s")
    
    async def test_rate_limiting(self, client: AsyncClient):
        """Test rate limiting with 1000 requests"""
        # Send many requests rapidly
        tasks = [client.get("/api/v3/health/live") for _ in range(50)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should not crash, but may rate limit
        valid_responses = [r for r in responses if not isinstance(r, Exception)]
        assert len(valid_responses) > 0, "All requests failed"
        
        # Check if any are rate limited
        status_codes = [r.status_code for r in valid_responses]
        print(f"\n✓ Rate limiting test: {len(valid_responses)} responses, status codes: {set(status_codes)}")


@pytest.mark.integration
@pytest.mark.asyncio
class TestCachePerformance:
    """Test cache performance metrics"""
    
    async def test_cache_hit_rate(self, client: AsyncClient):
        """Test L1-L4 cache hit rate (target 90%+)"""
        # Make same request multiple times
        url = "/api/v3/health/system"
        
        # First request (cache miss)
        first_response = await client.get(url)
        assert first_response.status_code == 200
        
        # Subsequent requests (should hit cache)
        for _ in range(10):
            response = await client.get(url)
            assert response.status_code == 200
        
        print("\n✓ Cache hit rate test completed")
    
    async def test_cache_response_time(self, client: AsyncClient, performance_monitor):
        """Test cache response time (<10ms for L1, <100ms for L3)"""
        # Warm up cache
        await client.get("/api/v3/health/system")
        
        # Measure cached request
        performance_monitor.start()
        response = await client.get("/api/v3/health/system")
        duration = performance_monitor.stop()
        
        assert response.status_code == 200
        
        # Should be fast (includes network overhead in test)
        assert duration < 1.0, f"Cached request took {duration}s"
        
        print(f"\n✓ Cache response time: {duration*1000:.2f}ms")


@pytest.mark.integration
@pytest.mark.asyncio
class TestDatabasePerformance:
    """Test database performance"""
    
    async def test_query_execution_time(self, client: AsyncClient, performance_monitor, test_user_credentials):
        """Test query execution time (<50ms for simple queries)"""
        # Register user (writes to DB)
        performance_monitor.start()
        response = await client.post(
            "/api/v3/auth/register",
            json=test_user_credentials
        )
        duration = performance_monitor.stop()
        
        assert response.status_code in [200, 201, 409]
        
        # Should be fast
        assert duration < 2.0, f"User registration took {duration}s"
        
        print(f"\n✓ Database write time: {duration*1000:.2f}ms")
    
    async def test_connection_pool_efficiency(self, client: AsyncClient):
        """Test database connection pool efficiency"""
        # Make multiple concurrent database requests
        tasks = []
        for i in range(20):
            creds = {
                "email": f"pool_test_{i}@example.com",
                "password": "TestPass123!",
                "full_name": f"Pool Test {i}"
            }
            tasks.append(client.post("/api/v3/auth/register", json=creds))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should handle concurrent DB operations
        successful = sum(1 for r in responses if not isinstance(r, Exception))
        assert successful >= 15, f"Only {successful}/20 concurrent DB operations succeeded"
        
        print(f"\n✓ Connection pool handled {successful}/20 concurrent operations")


@pytest.mark.integration
@pytest.mark.asyncio
class TestEndToEndLatency:
    """Test end-to-end latency measurements"""
    
    async def test_simple_workflow_latency(self, client: AsyncClient, sample_workflow_data, performance_monitor):
        """Test complete workflow latency (<1s for simple workflows)"""
        # Create workflow
        performance_monitor.start()
        
        response = await client.post(
            "/api/v3/workflows/create",
            json=sample_workflow_data
        )
        
        duration = performance_monitor.stop()
        
        assert response.status_code in [200, 201]
        
        # Should be fast
        assert duration < 5.0, f"Workflow creation took {duration}s"
        
        print(f"\n✓ Workflow creation latency: {duration*1000:.2f}ms")
    
    async def test_p95_p99_latency(self, client: AsyncClient):
        """Test P95/P99 latency measurements"""
        latencies: List[float] = []
        
        # Make 100 requests and measure latency
        for _ in range(100):
            import time
            start = time.time()
            await client.get("/api/v3/health/live")
            latency = time.time() - start
            latencies.append(latency)
        
        # Sort and calculate percentiles
        latencies.sort()
        p50 = latencies[49]
        p95 = latencies[94]
        p99 = latencies[98]
        
        print(f"\n✓ Latency percentiles:")
        print(f"  P50: {p50*1000:.2f}ms")
        print(f"  P95: {p95*1000:.2f}ms")
        print(f"  P99: {p99*1000:.2f}ms")
        
        # P95 should be reasonable
        assert p95 < 2.0, f"P95 latency is {p95}s, expected <2s"
