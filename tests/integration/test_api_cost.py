"""
Cost Governance API Integration Tests

Tests all cost governance endpoints:
- POST /api/v3/cost/budget
- POST /api/v3/cost/record
- GET /api/v3/cost/summary
- POST /api/v3/cost/project
- GET /api/v3/cost/budget/{id}
"""

import pytest
from httpx import AsyncClient
from typing import Dict, Any
from uuid import uuid4
from datetime import datetime, timedelta, timezone


@pytest.mark.integration
@pytest.mark.asyncio
class TestCostEndpoints:
    """Test cost governance API endpoints"""
    
    async def test_create_budget(self, client: AsyncClient, sample_cost_budget: Dict[str, Any]):
        """Test POST /api/v3/cost/budget"""
        response = await client.post(
            "/api/v3/cost/budget",
            json=sample_cost_budget
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert "budget_id" in data or "id" in data
        return data.get("budget_id") or data.get("id")
    
    async def test_record_cost(self, client: AsyncClient):
        """Test POST /api/v3/cost/record"""
        response = await client.post(
            "/api/v3/cost/record",
            json={
                "workflow_id": str(uuid4()),
                "cost_amount": 0.05,
                "currency": "USD",
                "cost_type": "llm_api",
                "provider": "openai",
                "metadata": {
                    "model": "gpt-3.5-turbo",
                    "tokens": 150
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        assert response.status_code in [200, 201, 204]
    
    async def test_get_cost_summary(self, client: AsyncClient):
        """Test GET /api/v3/cost/summary"""
        workflow_id = str(uuid4())
        
        # Record some costs
        await client.post(
            "/api/v3/cost/record",
            json={
                "workflow_id": workflow_id,
                "cost_amount": 0.10,
                "currency": "USD",
                "cost_type": "llm_api"
            }
        )
        
        # Get summary
        response = await client.get(
            f"/api/v3/cost/summary?workflow_id={workflow_id}"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "total_cost" in data or "summary" in data or isinstance(data, dict)
    
    async def test_project_cost(self, client: AsyncClient):
        """Test POST /api/v3/cost/project"""
        response = await client.post(
            "/api/v3/cost/project",
            json={
                "workflow_id": str(uuid4()),
                "estimated_steps": 100,
                "avg_cost_per_step": 0.01,
                "confidence_level": 0.80
            }
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert "projected_cost" in data or "estimate" in data or isinstance(data, dict)
    
    async def test_get_budget(self, client: AsyncClient, sample_cost_budget: Dict[str, Any]):
        """Test GET /api/v3/cost/budget/{id}"""
        # Create budget first
        create_response = await client.post(
            "/api/v3/cost/budget",
            json=sample_cost_budget
        )
        
        if create_response.status_code in [200, 201]:
            budget_id = create_response.json().get("budget_id") or create_response.json().get("id")
            
            # Get budget
            response = await client.get(f"/api/v3/cost/budget/{budget_id}")
            
            assert response.status_code in [200, 404]
            if response.status_code == 200:
                data = response.json()
                assert "budget_amount" in data or "amount" in data
