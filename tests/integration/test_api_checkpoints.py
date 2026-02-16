"""
Checkpoint API Integration Tests

Tests all checkpoint endpoints:
- POST /api/v3/checkpoints/create
- POST /api/v3/checkpoints/restore
- GET /api/v3/checkpoints/{id}
- DELETE /api/v3/checkpoints/{id}
- GET /api/v3/checkpoints/list
"""

import pytest
from httpx import AsyncClient
from typing import Dict, Any


@pytest.mark.integration
@pytest.mark.asyncio
class TestCheckpointEndpoints:
    """Test checkpoint API endpoints"""
    
    async def test_create_checkpoint(self, client: AsyncClient, sample_checkpoint_data: Dict[str, Any]):
        """Test POST /api/v3/checkpoints/create"""
        response = await client.post(
            "/api/v3/checkpoints/create",
            json=sample_checkpoint_data
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert "checkpoint_id" in data
        return data["checkpoint_id"]
    
    async def test_get_checkpoint(self, client: AsyncClient, sample_checkpoint_data: Dict[str, Any]):
        """Test GET /api/v3/checkpoints/{id}"""
        # Create checkpoint first
        create_response = await client.post(
            "/api/v3/checkpoints/create",
            json=sample_checkpoint_data
        )
        assert create_response.status_code in [200, 201]
        checkpoint_id = create_response.json()["checkpoint_id"]
        
        # Get checkpoint
        response = await client.get(f"/api/v3/checkpoints/{checkpoint_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["checkpoint_id"] == checkpoint_id
        assert "state" in data
    
    async def test_restore_checkpoint(self, client: AsyncClient, sample_checkpoint_data: Dict[str, Any]):
        """Test POST /api/v3/checkpoints/restore"""
        # Create checkpoint first
        create_response = await client.post(
            "/api/v3/checkpoints/create",
            json=sample_checkpoint_data
        )
        assert create_response.status_code in [200, 201]
        checkpoint_id = create_response.json()["checkpoint_id"]
        
        # Restore checkpoint
        response = await client.post(
            "/api/v3/checkpoints/restore",
            json={"checkpoint_id": checkpoint_id}
        )
        
        assert response.status_code in [200, 204]
    
    async def test_delete_checkpoint(self, client: AsyncClient, sample_checkpoint_data: Dict[str, Any]):
        """Test DELETE /api/v3/checkpoints/{id}"""
        # Create checkpoint first
        create_response = await client.post(
            "/api/v3/checkpoints/create",
            json=sample_checkpoint_data
        )
        assert create_response.status_code in [200, 201]
        checkpoint_id = create_response.json()["checkpoint_id"]
        
        # Delete checkpoint
        response = await client.delete(f"/api/v3/checkpoints/{checkpoint_id}")
        
        assert response.status_code in [200, 204]
        
        # Verify deletion
        get_response = await client.get(f"/api/v3/checkpoints/{checkpoint_id}")
        assert get_response.status_code == 404
    
    async def test_list_checkpoints(self, client: AsyncClient, sample_checkpoint_data: Dict[str, Any]):
        """Test GET /api/v3/checkpoints/list"""
        # Create a few checkpoints
        for _ in range(3):
            await client.post(
                "/api/v3/checkpoints/create",
                json=sample_checkpoint_data
            )
        
        # List checkpoints
        response = await client.get("/api/v3/checkpoints/list")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))
        if isinstance(data, dict):
            assert "checkpoints" in data or "items" in data
        elif isinstance(data, list):
            assert len(data) >= 0
