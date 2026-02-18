"""
Integration Test Fixtures and Configuration

Provides shared fixtures for integration tests including:
- Test database setup/teardown
- Test client configuration
- Mock external services
- Test data factories
"""

import pytest
import asyncio
from typing import AsyncGenerator, Dict, Any
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from datetime import datetime, timedelta, timezone

# Add project root to path
import os

from services.api.src.main import app


# ==================== Event Loop ====================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ==================== HTTP Client ====================

@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing API"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


# ==================== Authentication ====================

@pytest.fixture
def test_user_credentials() -> Dict[str, str]:
    """Sample user credentials for testing"""
    return {
        "email": f"test_{uuid4().hex[:8]}@example.com",
        "password": "TestPass123!",
        "full_name": "Test User"
    }


@pytest.fixture
async def authenticated_client(client: AsyncClient, test_user_credentials: Dict[str, str]) -> AsyncClient:
    """Create authenticated client with valid JWT token"""
    # Register user
    await client.post(
        "/api/v3/auth/register",
        json=test_user_credentials
    )
    
    # Login to get token
    response = await client.post(
        "/api/v3/auth/login",
        json={
            "email": test_user_credentials["email"],
            "password": test_user_credentials["password"]
        }
    )
    
    if response.status_code == 200:
        token_data = response.json()
        client.headers["Authorization"] = f"Bearer {token_data['access_token']}"
    
    return client


# ==================== Test Data Factories ====================

@pytest.fixture
def sample_workflow_data() -> Dict[str, Any]:
    """Sample workflow data for testing"""
    workflow_id = f"workflow-{uuid4().hex[:8]}"
    return {
        "workflow_id": workflow_id,
        "version": "1.0.0",
        "name": f"Test Workflow {workflow_id}",
        "description": "Integration test workflow",
        "steps": [
            {
                "step_id": "step-1",
                "name": "First Step",
                "type": "task",
                "dependencies": [],
                "config": {
                    "timeout": 300,
                    "retry_count": 3
                }
            },
            {
                "step_id": "step-2",
                "name": "Second Step",
                "type": "task",
                "dependencies": ["step-1"],
                "config": {
                    "timeout": 300,
                    "retry_count": 3
                }
            }
        ],
        "metadata": {
            "created_by": "test",
            "environment": "test"
        }
    }


@pytest.fixture
def sample_checkpoint_data() -> Dict[str, Any]:
    """Sample checkpoint data for testing"""
    return {
        "workflow_id": str(uuid4()),
        "step_id": "step-1",
        "state": {
            "current_step": 1,
            "total_steps": 5,
            "context": {"key": "value"},
            "intermediate_results": ["result1", "result2"]
        },
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": "integration_test"
        }
    }


@pytest.fixture
def sample_memory_data() -> Dict[str, Any]:
    """Sample memory data for testing"""
    return {
        "content": "Python is a high-level programming language",
        "memory_type": "semantic",
        "importance": 0.85,
        "tags": ["programming", "python", "language"],
        "metadata": {
            "source": "test",
            "category": "knowledge"
        }
    }


@pytest.fixture
def sample_cost_budget() -> Dict[str, Any]:
    """Sample cost budget for testing"""
    return {
        "workflow_id": str(uuid4()),
        "budget_amount": 100.00,
        "currency": "USD",
        "period_start": datetime.now(timezone.utc).isoformat(),
        "period_end": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
        "alert_thresholds": {
            "warning": 0.75,
            "critical": 0.90
        }
    }


@pytest.fixture
def sample_task_decomposition() -> Dict[str, Any]:
    """Sample task decomposition data for testing"""
    return {
        "task_id": str(uuid4()),
        "description": "Build a web application with authentication",
        "strategy": "breadth_first",
        "max_depth": 5,
        "max_nodes": 100
    }


# ==================== Mock Services ====================

@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing"""
    mock = AsyncMock()
    mock.complete.return_value = {
        "content": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)",
        "model_used": "gpt-3.5-turbo",
        "tokens_used": 45,
        "cost_usd": 0.000045
    }
    mock.embed.return_value = {
        "embeddings": [[0.1] * 1536],
        "model_used": "text-embedding-ada-002"
    }
    return mock


@pytest.fixture
def mock_redis_service():
    """Mock Redis service for testing"""
    mock = MagicMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=True)
    mock.ping = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_rabbitmq_service():
    """Mock RabbitMQ service for testing"""
    mock = AsyncMock()
    mock.publish = AsyncMock(return_value=True)
    mock.consume = AsyncMock()
    mock.is_connected = AsyncMock(return_value=True)
    return mock


# ==================== Cleanup ====================

@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Cleanup resources after each test"""
    yield
    # Add any cleanup logic here
    # For example: clear test database, reset mocks, etc.


# ==================== Performance Monitoring ====================

@pytest.fixture
def performance_monitor():
    """Monitor test performance"""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.duration = None
        
        def start(self):
            self.start_time = datetime.now(timezone.utc)
        
        def stop(self):
            self.end_time = datetime.now(timezone.utc)
            if self.start_time:
                self.duration = (self.end_time - self.start_time).total_seconds()
            return self.duration
        
        def assert_duration_under(self, max_seconds: float):
            """Assert test completed under max_seconds"""
            if self.duration is None:
                self.stop()
            assert self.duration <= max_seconds, f"Test took {self.duration}s, expected <{max_seconds}s"
    
    return PerformanceMonitor()
