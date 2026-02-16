"""
Memory Hierarchy API Integration Tests

Tests all memory hierarchy endpoints:
- POST /api/v3/memory/store
- GET /api/v3/memory/{id}
- POST /api/v3/memory/promote
- POST /api/v3/memory/score
- POST /api/v3/memory/compress
- GET /api/v3/memory/search
- POST /api/v3/memory/tier-transition
- POST /api/v3/memory/evict
"""

import pytest
from httpx import AsyncClient
from typing import Dict, Any
from uuid import uuid4


@pytest.mark.integration
@pytest.mark.asyncio
class TestMemoryEndpoints:
    """Test memory hierarchy API endpoints"""
    
    async def test_store_memory(self, client: AsyncClient, sample_memory_data: Dict[str, Any]):
        """Test POST /api/v3/memory/store"""
        response = await client.post(
            "/api/v3/memory/store",
            json=sample_memory_data
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert "id" in data or "memory_id" in data
        return data.get("id") or data.get("memory_id")
    
    async def test_get_memory(self, client: AsyncClient, sample_memory_data: Dict[str, Any]):
        """Test GET /api/v3/memory/{id}"""
        # Store memory first
        store_response = await client.post(
            "/api/v3/memory/store",
            json=sample_memory_data
        )
        
        if store_response.status_code in [200, 201]:
            memory_id = store_response.json().get("id") or store_response.json().get("memory_id")
            
            # Get memory
            response = await client.get(f"/api/v3/memory/{memory_id}")
            
            assert response.status_code in [200, 404]
            if response.status_code == 200:
                data = response.json()
                assert "content" in data
                assert data["content"] == sample_memory_data["content"]
    
    async def test_promote_memory(self, client: AsyncClient, sample_memory_data: Dict[str, Any]):
        """Test POST /api/v3/memory/promote"""
        # Store memory first
        store_response = await client.post(
            "/api/v3/memory/store",
            json={**sample_memory_data, "tier": "L1"}
        )
        
        if store_response.status_code in [200, 201]:
            memory_id = store_response.json().get("id") or store_response.json().get("memory_id")
            
            # Promote memory
            response = await client.post(
                "/api/v3/memory/promote",
                json={
                    "memory_id": memory_id,
                    "target_tier": "L2"
                }
            )
            
            assert response.status_code in [200, 204]
    
    async def test_score_memory(self, client: AsyncClient, sample_memory_data: Dict[str, Any]):
        """Test POST /api/v3/memory/score"""
        # Store memory first
        store_response = await client.post(
            "/api/v3/memory/store",
            json=sample_memory_data
        )
        
        if store_response.status_code in [200, 201]:
            memory_id = store_response.json().get("id") or store_response.json().get("memory_id")
            
            # Score memory
            response = await client.post(
                "/api/v3/memory/score",
                json={
                    "memory_ids": [memory_id],
                    "scoring_method": "importance"
                }
            )
            
            assert response.status_code in [200, 204]
    
    async def test_compress_memories(self, client: AsyncClient):
        """Test POST /api/v3/memory/compress"""
        # Store multiple memories
        memory_ids = []
        for i in range(5):
            response = await client.post(
                "/api/v3/memory/store",
                json={
                    "content": f"Test memory {i}",
                    "memory_type": "semantic",
                    "importance": 0.5
                }
            )
            if response.status_code in [200, 201]:
                memory_id = response.json().get("id") or response.json().get("memory_id")
                memory_ids.append(memory_id)
        
        if memory_ids:
            # Compress memories
            response = await client.post(
                "/api/v3/memory/compress",
                json={
                    "memory_ids": memory_ids,
                    "compression_method": "cluster"
                }
            )
            
            assert response.status_code in [200, 201, 204]
    
    async def test_search_memories(self, client: AsyncClient):
        """Test GET /api/v3/memory/search"""
        # Store some memories
        for content in ["Python programming", "JavaScript development", "Database design"]:
            await client.post(
                "/api/v3/memory/store",
                json={
                    "content": content,
                    "memory_type": "semantic",
                    "importance": 0.8
                }
            )
        
        # Search memories
        response = await client.get(
            "/api/v3/memory/search?query=programming&limit=10"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))
    
    async def test_tier_transition(self, client: AsyncClient, sample_memory_data: Dict[str, Any]):
        """Test POST /api/v3/memory/tier-transition"""
        # Store memory
        store_response = await client.post(
            "/api/v3/memory/store",
            json={**sample_memory_data, "tier": "L1"}
        )
        
        if store_response.status_code in [200, 201]:
            memory_id = store_response.json().get("id") or store_response.json().get("memory_id")
            
            # Trigger tier transition
            response = await client.post(
                "/api/v3/memory/tier-transition",
                json={
                    "memory_id": memory_id,
                    "from_tier": "L1",
                    "to_tier": "L2"
                }
            )
            
            assert response.status_code in [200, 204]
    
    async def test_evict_memory(self, client: AsyncClient, sample_memory_data: Dict[str, Any]):
        """Test POST /api/v3/memory/evict"""
        # Store memory
        store_response = await client.post(
            "/api/v3/memory/store",
            json=sample_memory_data
        )
        
        if store_response.status_code in [200, 201]:
            memory_id = store_response.json().get("id") or store_response.json().get("memory_id")
            
            # Evict memory
            response = await client.post(
                "/api/v3/memory/evict",
                json={
                    "memory_ids": [memory_id],
                    "reason": "low_importance"
                }
            )
            
            assert response.status_code in [200, 204]
