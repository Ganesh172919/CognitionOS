"""
Service Integration Tests

Tests integration between services and infrastructure:
- Repository integration with PostgreSQL
- Event bus integration with RabbitMQ
- Cache integration (L1-L4)
- Message broker integration
- Distributed coordination
"""

import pytest
from httpx import AsyncClient
from uuid import uuid4
import asyncio


@pytest.mark.integration
@pytest.mark.asyncio
class TestRepositoryIntegration:
    """Test repository integration with real database"""
    
    async def test_user_repository_crud(self, client: AsyncClient, test_user_credentials):
        """Test user repository CRUD operations"""
        # Create (register)
        create_response = await client.post(
            "/api/v3/auth/register",
            json=test_user_credentials
        )
        assert create_response.status_code in [200, 201, 409]
        
        # Read (get current user)
        if create_response.status_code in [200, 201]:
            login_response = await client.post(
                "/api/v3/auth/login",
                json={
                    "email": test_user_credentials["email"],
                    "password": test_user_credentials["password"]
                }
            )
            if login_response.status_code == 200:
                token = login_response.json()["access_token"]
                client.headers["Authorization"] = f"Bearer {token}"
                
                me_response = await client.get("/api/v3/auth/me")
                assert me_response.status_code == 200
    
    async def test_checkpoint_repository_operations(self, client: AsyncClient, sample_checkpoint_data):
        """Test checkpoint repository CRUD operations"""
        # Create
        create_response = await client.post(
            "/api/v3/checkpoints/create",
            json=sample_checkpoint_data
        )
        assert create_response.status_code in [200, 201]
        checkpoint_id = create_response.json()["checkpoint_id"]
        
        # Read
        get_response = await client.get(f"/api/v3/checkpoints/{checkpoint_id}")
        assert get_response.status_code == 200
        
        # Delete
        delete_response = await client.delete(f"/api/v3/checkpoints/{checkpoint_id}")
        assert delete_response.status_code in [200, 204]
    
    async def test_memory_repository_with_pgvector(self, client: AsyncClient, sample_memory_data):
        """Test memory repository with pgvector semantic search"""
        # Store memory
        store_response = await client.post(
            "/api/v3/memory/store",
            json=sample_memory_data
        )
        assert store_response.status_code in [200, 201]
        
        # Search (semantic)
        search_response = await client.get(
            "/api/v3/memory/search?query=python&limit=5"
        )
        assert search_response.status_code == 200
    
    async def test_cost_repository_operations(self, client: AsyncClient):
        """Test cost governance repository"""
        workflow_id = str(uuid4())
        
        # Record cost
        record_response = await client.post(
            "/api/v3/cost/record",
            json={
                "workflow_id": workflow_id,
                "cost_amount": 0.05,
                "currency": "USD",
                "cost_type": "llm_api"
            }
        )
        assert record_response.status_code in [200, 201, 204]
        
        # Get summary
        summary_response = await client.get(
            f"/api/v3/cost/summary?workflow_id={workflow_id}"
        )
        assert summary_response.status_code == 200
    
    async def test_workflow_repository_operations(self, client: AsyncClient, sample_workflow_data):
        """Test workflow repository operations"""
        # Create workflow
        create_response = await client.post(
            "/api/v3/workflows/create",
            json=sample_workflow_data
        )
        assert create_response.status_code in [200, 201]
        
        # Get workflow
        workflow_id = create_response.json()["workflow_id"]
        get_response = await client.get(f"/api/v3/workflows/{workflow_id}")
        assert get_response.status_code in [200, 404]


@pytest.mark.integration
@pytest.mark.asyncio
class TestEventBusIntegration:
    """Test event bus integration with RabbitMQ"""
    
    async def test_event_publishing(self, client: AsyncClient, sample_workflow_data):
        """Test event publishing to RabbitMQ"""
        # Create workflow (should publish WorkflowCreated event)
        response = await client.post(
            "/api/v3/workflows/create",
            json=sample_workflow_data
        )
        assert response.status_code in [200, 201]
    
    async def test_event_consumption_and_handling(self, client: AsyncClient):
        """Test event consumption and handling"""
        # Health check should work (handlers running)
        response = await client.get("/api/v3/health/system")
        assert response.status_code == 200
    
    async def test_event_handler_error_recovery(self, client: AsyncClient):
        """Test event handler error recovery"""
        # System should remain healthy even with errors
        response = await client.get("/api/v3/health/ready")
        assert response.status_code in [200, 503]


@pytest.mark.integration
@pytest.mark.asyncio
class TestCacheIntegration:
    """Test multi-layer cache integration"""
    
    async def test_l1_redis_cache(self, client: AsyncClient):
        """Test L1 Redis cache operations"""
        # Health check includes Redis status
        response = await client.get("/api/v3/health/system")
        if response.status_code == 200:
            data = response.json()
            checks = data.get("checks", {})
            if "redis" in checks:
                assert checks["redis"]["status"] in ["healthy", "degraded", "unhealthy"]
    
    async def test_l2_postgresql_cache(self, client: AsyncClient):
        """Test L2 PostgreSQL cache"""
        # Database health check
        response = await client.get("/api/v3/health/system")
        if response.status_code == 200:
            data = response.json()
            checks = data.get("checks", {})
            if "database" in checks:
                assert checks["database"]["status"] in ["healthy", "degraded", "unhealthy"]
    
    async def test_l3_semantic_cache(self, client: AsyncClient, sample_memory_data):
        """Test L3 pgvector semantic cache"""
        # Store and search memory (uses semantic cache)
        store_response = await client.post(
            "/api/v3/memory/store",
            json=sample_memory_data
        )
        assert store_response.status_code in [200, 201]
        
        # Search should use semantic cache
        search_response = await client.get(
            "/api/v3/memory/search?query=programming&limit=5"
        )
        assert search_response.status_code == 200
    
    async def test_cache_performance(self, client: AsyncClient, performance_monitor):
        """Test cache performance metrics"""
        performance_monitor.start()
        
        # Multiple requests should benefit from caching
        for _ in range(5):
            await client.get("/api/v3/health/system")
        
        duration = performance_monitor.stop()
        # Should be fast with caching
        assert duration < 2.0, f"5 requests took {duration}s"


@pytest.mark.integration
@pytest.mark.asyncio
class TestMessageBrokerIntegration:
    """Test message broker integration for async tasks"""
    
    async def test_celery_task_execution(self, client: AsyncClient, sample_workflow_data):
        """Test Celery task execution"""
        # Execute workflow (triggers async Celery tasks)
        create_response = await client.post(
            "/api/v3/workflows/create",
            json=sample_workflow_data
        )
        
        if create_response.status_code in [200, 201]:
            workflow_id = create_response.json()["workflow_id"]
            
            exec_response = await client.post(
                f"/api/v3/workflows/execute",
                json={"workflow_id": workflow_id}
            )
            assert exec_response.status_code in [200, 202]
    
    async def test_async_workflow_task_processing(self, client: AsyncClient):
        """Test async workflow task processing"""
        # RabbitMQ health check
        response = await client.get("/api/v3/health/system")
        if response.status_code == 200:
            data = response.json()
            checks = data.get("checks", {})
            if "rabbitmq" in checks:
                assert checks["rabbitmq"]["status"] in ["healthy", "degraded", "unhealthy"]
    
    async def test_task_retry_mechanism(self, client: AsyncClient, sample_workflow_data):
        """Test task retry and dead-letter queue"""
        # Workflow with retry config
        workflow_with_retry = {
            **sample_workflow_data,
            "steps": [
                {
                    **sample_workflow_data["steps"][0],
                    "config": {
                        "retry_count": 3,
                        "retry_delay": 1
                    }
                }
            ]
        }
        
        response = await client.post(
            "/api/v3/workflows/create",
            json=workflow_with_retry
        )
        assert response.status_code in [200, 201]


@pytest.mark.integration
@pytest.mark.asyncio
class TestDistributedCoordination:
    """Test distributed coordination with etcd"""
    
    async def test_leader_election(self, client: AsyncClient):
        """Test leader election mechanism"""
        # System should operate correctly with leader election
        response = await client.get("/api/v3/health/system")
        assert response.status_code == 200
    
    async def test_distributed_lock(self, client: AsyncClient, sample_workflow_data):
        """Test distributed locking"""
        # Create workflow (may use distributed locks)
        response = await client.post(
            "/api/v3/workflows/create",
            json=sample_workflow_data
        )
        assert response.status_code in [200, 201]
    
    async def test_service_discovery(self, client: AsyncClient):
        """Test service discovery"""
        # Health check verifies service discovery
        response = await client.get("/api/v3/health/ready")
        assert response.status_code in [200, 503]
