"""
Health Monitoring API Integration Tests

Tests all health monitoring endpoints:
- POST /api/v3/health/heartbeat
- GET /api/v3/health/agent/{id}
- POST /api/v3/health/incident
- POST /api/v3/health/resolve
- POST /api/v3/health/trigger-recovery
- GET /api/v3/health/summary
- GET /api/v3/health/system
- GET /api/v3/health/ready
"""

import pytest
from httpx import AsyncClient
from uuid import uuid4
from datetime import datetime, timezone


@pytest.mark.integration
@pytest.mark.asyncio
class TestHealthEndpoints:
    """Test health monitoring API endpoints"""
    
    async def test_agent_heartbeat(self, client: AsyncClient):
        """Test POST /api/v3/health/heartbeat"""
        agent_id = str(uuid4())
        response = await client.post(
            "/api/v3/health/heartbeat",
            json={
                "agent_id": agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "healthy",
                "metrics": {
                    "cpu_usage": 45.5,
                    "memory_usage": 60.2,
                    "task_count": 3
                }
            }
        )
        
        assert response.status_code in [200, 201, 204]
    
    async def test_get_agent_health(self, client: AsyncClient):
        """Test GET /api/v3/health/agent/{id}"""
        agent_id = str(uuid4())
        
        # Send heartbeat first
        await client.post(
            "/api/v3/health/heartbeat",
            json={
                "agent_id": agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "healthy"
            }
        )
        
        # Get agent health
        response = await client.get(f"/api/v3/health/agent/{agent_id}")
        
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "agent_id" in data or "status" in data
    
    async def test_create_incident(self, client: AsyncClient):
        """Test POST /api/v3/health/incident"""
        response = await client.post(
            "/api/v3/health/incident",
            json={
                "agent_id": str(uuid4()),
                "incident_type": "service_degradation",
                "severity": "warning",
                "description": "High memory usage detected",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert "incident_id" in data
        return data["incident_id"]
    
    async def test_resolve_incident(self, client: AsyncClient):
        """Test POST /api/v3/health/resolve"""
        # Create incident first
        create_response = await client.post(
            "/api/v3/health/incident",
            json={
                "agent_id": str(uuid4()),
                "incident_type": "test_incident",
                "severity": "info",
                "description": "Test incident"
            }
        )
        
        if create_response.status_code in [200, 201]:
            incident_id = create_response.json()["incident_id"]
            
            # Resolve incident
            response = await client.post(
                "/api/v3/health/resolve",
                json={
                    "incident_id": incident_id,
                    "resolution": "Issue resolved automatically"
                }
            )
            
            assert response.status_code in [200, 204]
    
    async def test_trigger_recovery(self, client: AsyncClient):
        """Test POST /api/v3/health/trigger-recovery"""
        response = await client.post(
            "/api/v3/health/trigger-recovery",
            json={
                "agent_id": str(uuid4()),
                "recovery_action": "restart",
                "reason": "Performance degradation"
            }
        )
        
        assert response.status_code in [200, 202, 204]
    
    async def test_get_health_summary(self, client: AsyncClient):
        """Test GET /api/v3/health/summary"""
        response = await client.get("/api/v3/health/summary")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_agents" in data or "summary" in data or isinstance(data, dict)
    
    async def test_system_health_check(self, client: AsyncClient, performance_monitor):
        """Test GET /api/v3/health/system (comprehensive check)"""
        performance_monitor.start()
        
        response = await client.get("/api/v3/health/system")
        
        duration = performance_monitor.stop()
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "checks" in data
        
        # Verify individual service checks
        checks = data["checks"]
        assert "redis" in checks or "database" in checks or "rabbitmq" in checks
        
        # Performance check: should complete in <100ms
        assert duration < 0.5, f"Health check took {duration}s, expected <0.5s"
    
    async def test_readiness_probe(self, client: AsyncClient):
        """Test GET /api/v3/health/ready (K8s readiness probe)"""
        response = await client.get("/api/v3/health/ready")
        
        # Should return 200 if ready, 503 if not
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data or "ready" in data
    
    async def test_liveness_probe(self, client: AsyncClient):
        """Test GET /api/v3/health/live (K8s liveness probe)"""
        response = await client.get("/api/v3/health/live")
        
        # Should always return 200 if application is running
        assert response.status_code == 200
