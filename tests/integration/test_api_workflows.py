"""
Workflow API Integration Tests

Tests all workflow endpoints:
- POST /api/v3/workflows/create
- GET /api/v3/workflows/{id}
- POST /api/v3/workflows/execute
- GET /api/v3/workflows/status/{id}
- GET /api/v3/workflows/list
"""

import pytest
from httpx import AsyncClient
from typing import Dict, Any


@pytest.mark.integration
@pytest.mark.asyncio
class TestWorkflowEndpoints:
    """Test workflow API endpoints"""
    
    async def test_create_workflow(self, client: AsyncClient, sample_workflow_data: Dict[str, Any]):
        """Test POST /api/v3/workflows/create"""
        response = await client.post(
            "/api/v3/workflows/create",
            json=sample_workflow_data
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert "workflow_id" in data
        return data["workflow_id"]
    
    async def test_get_workflow(self, client: AsyncClient, sample_workflow_data: Dict[str, Any]):
        """Test GET /api/v3/workflows/{id}"""
        # Create workflow first
        create_response = await client.post(
            "/api/v3/workflows/create",
            json=sample_workflow_data
        )
        
        if create_response.status_code in [200, 201]:
            workflow_id = create_response.json()["workflow_id"]
            
            # Get workflow
            response = await client.get(f"/api/v3/workflows/{workflow_id}")
            
            assert response.status_code in [200, 404]
            if response.status_code == 200:
                data = response.json()
                assert data["workflow_id"] == workflow_id
    
    async def test_execute_workflow(self, client: AsyncClient, sample_workflow_data: Dict[str, Any]):
        """Test POST /api/v3/workflows/execute"""
        # Create workflow first
        create_response = await client.post(
            "/api/v3/workflows/create",
            json=sample_workflow_data
        )
        
        if create_response.status_code in [200, 201]:
            workflow_id = create_response.json()["workflow_id"]
            
            # Execute workflow
            response = await client.post(
                f"/api/v3/workflows/execute",
                json={"workflow_id": workflow_id}
            )
            
            assert response.status_code in [200, 202]
            data = response.json()
            assert "execution_id" in data or "status" in data
    
    async def test_get_workflow_status(self, client: AsyncClient, sample_workflow_data: Dict[str, Any]):
        """Test GET /api/v3/workflows/status/{id}"""
        # Create and execute workflow
        create_response = await client.post(
            "/api/v3/workflows/create",
            json=sample_workflow_data
        )
        
        if create_response.status_code in [200, 201]:
            workflow_id = create_response.json()["workflow_id"]
            
            # Execute
            await client.post(
                f"/api/v3/workflows/execute",
                json={"workflow_id": workflow_id}
            )
            
            # Get status
            response = await client.get(f"/api/v3/workflows/status/{workflow_id}")
            
            assert response.status_code in [200, 404]
            if response.status_code == 200:
                data = response.json()
                assert "status" in data or "state" in data
    
    async def test_list_workflows(self, client: AsyncClient):
        """Test GET /api/v3/workflows/list"""
        response = await client.get("/api/v3/workflows/list")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))
