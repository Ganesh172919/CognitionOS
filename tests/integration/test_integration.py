"""
Integration tests for CognitionOS services.

Tests the complete workflow from task creation to execution.
"""

import pytest
import asyncio
import httpx
from uuid import uuid4
from datetime import datetime


# Test configuration
API_GATEWAY_URL = "http://localhost:8000"
AUTH_SERVICE_URL = "http://localhost:8001"
AI_RUNTIME_URL = "http://localhost:8005"
MEMORY_SERVICE_URL = "http://localhost:8004"

# Test credentials
TEST_USER = {
    "username": "test_user",
    "email": "test@cognitionos.com",
    "password": "Test123!@#"
}


class TestAuthFlow:
    """Test authentication and authorization flow."""

    @pytest.mark.asyncio
    async def test_user_registration(self):
        """Test user can register successfully."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{AUTH_SERVICE_URL}/register",
                json=TEST_USER
            )
            assert response.status_code in [200, 201, 409]  # 409 if user exists
            if response.status_code in [200, 201]:
                data = response.json()
                assert "user_id" in data or "id" in data

    @pytest.mark.asyncio
    async def test_user_login(self):
        """Test user can login and receive JWT token."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{AUTH_SERVICE_URL}/login",
                json={
                    "username": TEST_USER["username"],
                    "password": TEST_USER["password"]
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data
            assert "token_type" in data
            return data["access_token"]


class TestAIRuntime:
    """Test AI Runtime service."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test AI Runtime health endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{AI_RUNTIME_URL}/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "providers" in data

    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test listing available models."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{AI_RUNTIME_URL}/models")
            assert response.status_code == 200
            data = response.json()
            assert "models" in data
            assert "role_assignments" in data

    @pytest.mark.asyncio
    async def test_completion_request(self):
        """Test LLM completion request."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{AI_RUNTIME_URL}/complete",
                json={
                    "role": "executor",
                    "prompt": "Write a simple hello world function in Python",
                    "context": [],
                    "max_tokens": 500,
                    "temperature": 0.7,
                    "user_id": str(uuid4())
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert "content" in data
            assert "model_used" in data
            assert "tokens_used" in data
            assert "cost_usd" in data
            assert len(data["content"]) > 0

    @pytest.mark.asyncio
    async def test_embedding_request(self):
        """Test embedding generation."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{AI_RUNTIME_URL}/embed",
                json={
                    "texts": [
                        "Hello world",
                        "How are you?",
                        "CognitionOS is an AI operating system"
                    ],
                    "model": "text-embedding-ada-002"
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert "embeddings" in data
            assert len(data["embeddings"]) == 3
            assert len(data["embeddings"][0]) == 1536  # ada-002 dimension


class TestMemoryService:
    """Test Memory Service."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test Memory Service health endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{MEMORY_SERVICE_URL}/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ["healthy", "unhealthy"]

    @pytest.mark.asyncio
    async def test_store_memory(self):
        """Test storing a memory."""
        user_id = uuid4()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{MEMORY_SERVICE_URL}/memories",
                json={
                    "user_id": str(user_id),
                    "content": "Python is a programming language",
                    "memory_type": "semantic",
                    "scope": "user",
                    "metadata": {"language": "python"},
                    "source": "test",
                    "confidence": 1.0,
                    "is_sensitive": False
                }
            )
            assert response.status_code == 201
            data = response.json()
            assert "id" in data
            assert data["content"] == "Python is a programming language"
            return data["id"]

    @pytest.mark.asyncio
    async def test_retrieve_memories(self):
        """Test retrieving memories."""
        user_id = uuid4()

        async with httpx.AsyncClient(timeout=30.0) as client:
            # First, store some memories
            memories_to_store = [
                "Python is a programming language",
                "JavaScript is used for web development",
                "Docker is a containerization platform"
            ]

            for content in memories_to_store:
                await client.post(
                    f"{MEMORY_SERVICE_URL}/memories",
                    json={
                        "user_id": str(user_id),
                        "content": content,
                        "memory_type": "semantic",
                        "scope": "user",
                        "metadata": {},
                        "source": "test",
                        "confidence": 1.0,
                        "is_sensitive": False
                    }
                )

            # Now retrieve memories
            response = await client.post(
                f"{MEMORY_SERVICE_URL}/retrieve",
                json={
                    "user_id": str(user_id),
                    "query": "What programming languages do you know?",
                    "k": 5,
                    "memory_types": ["semantic"],
                    "min_confidence": 0.0
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) > 0


class TestFullWorkflow:
    """Test complete end-to-end workflow."""

    @pytest.mark.asyncio
    async def test_complete_task_workflow(self):
        """
        Test complete workflow:
        1. Register/Login user
        2. Create task
        3. Generate completion with AI Runtime
        4. Store result in memory
        5. Retrieve from memory
        """
        user_id = uuid4()

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 1: Health checks
            print("\n=== Checking service health ===")
            ai_health = await client.get(f"{AI_RUNTIME_URL}/health")
            assert ai_health.status_code == 200
            print(f"✓ AI Runtime: {ai_health.json()['status']}")

            memory_health = await client.get(f"{MEMORY_SERVICE_URL}/health")
            assert memory_health.status_code == 200
            print(f"✓ Memory Service: {memory_health.json()['status']}")

            # Step 2: Generate a completion
            print("\n=== Generating AI completion ===")
            completion_response = await client.post(
                f"{AI_RUNTIME_URL}/complete",
                json={
                    "role": "executor",
                    "prompt": "Write a Python function to calculate factorial",
                    "context": [],
                    "max_tokens": 500,
                    "temperature": 0.7,
                    "user_id": str(user_id)
                }
            )
            assert completion_response.status_code == 200
            completion_data = completion_response.json()
            generated_code = completion_data["content"]
            print(f"✓ Generated code ({completion_data['tokens_used']} tokens)")
            print(f"  Model: {completion_data['model_used']}")
            print(f"  Cost: ${completion_data['cost_usd']:.6f}")

            # Step 3: Store in memory
            print("\n=== Storing in memory ===")
            memory_response = await client.post(
                f"{MEMORY_SERVICE_URL}/memories",
                json={
                    "user_id": str(user_id),
                    "content": f"Factorial function implementation: {generated_code}",
                    "memory_type": "semantic",
                    "scope": "user",
                    "metadata": {
                        "task": "factorial",
                        "language": "python",
                        "tokens": completion_data["tokens_used"]
                    },
                    "source": "ai_runtime",
                    "confidence": 0.9,
                    "is_sensitive": False
                }
            )
            assert memory_response.status_code == 201
            memory_data = memory_response.json()
            print(f"✓ Stored memory ID: {memory_data['id']}")

            # Step 4: Retrieve from memory
            print("\n=== Retrieving from memory ===")
            retrieve_response = await client.post(
                f"{MEMORY_SERVICE_URL}/retrieve",
                json={
                    "user_id": str(user_id),
                    "query": "How do I calculate factorial?",
                    "k": 3,
                    "memory_types": ["semantic"],
                    "min_confidence": 0.5
                }
            )
            assert retrieve_response.status_code == 200
            retrieved = retrieve_response.json()
            assert len(retrieved) > 0
            print(f"✓ Retrieved {len(retrieved)} relevant memories")
            if retrieved:
                print(f"  Top match (score: {retrieved[0].get('relevance_score', 'N/A')}):")
                print(f"  {retrieved[0]['content'][:100]}...")

            print("\n=== Workflow completed successfully! ===")


# Performance tests
class TestPerformance:
    """Test system performance under load."""

    @pytest.mark.asyncio
    async def test_concurrent_completions(self):
        """Test handling multiple concurrent completion requests."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            tasks = []
            for i in range(5):
                task = client.post(
                    f"{AI_RUNTIME_URL}/complete",
                    json={
                        "role": "executor",
                        "prompt": f"Count to {i+1}",
                        "context": [],
                        "max_tokens": 100,
                        "temperature": 0.7,
                        "user_id": str(uuid4())
                    }
                )
                tasks.append(task)

            responses = await asyncio.gather(*tasks)
            assert all(r.status_code == 200 for r in responses)
            print(f"\n✓ Successfully handled {len(responses)} concurrent requests")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
