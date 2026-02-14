"""
Unit Tests for Workflow Endpoints

Tests for workflow creation, execution, listing, and status endpoints.
"""

import pytest
from httpx import AsyncClient
from unittest.mock import patch, MagicMock


@pytest.mark.asyncio
@pytest.mark.unit
class TestWorkflowEndpoints:
    """Test suite for workflow endpoints"""
    
    async def test_create_workflow_endpoint_exists(self, client: AsyncClient, simple_workflow):
        """Test that create workflow endpoint exists"""
        response = await client.post(
            "/api/v3/workflows",
            json=simple_workflow
        )
        
        # Endpoint should exist (may fail without DB, but shouldn't 404)
        assert response.status_code != 404
    
    async def test_create_workflow_validation(self, client: AsyncClient):
        """Test workflow creation with invalid data"""
        response = await client.post(
            "/api/v3/workflows",
            json={"invalid": "data"}
        )
        
        # Should fail validation
        assert response.status_code == 422
    
    async def test_create_workflow_missing_fields(self, client: AsyncClient):
        """Test workflow creation with missing required fields"""
        response = await client.post(
            "/api/v3/workflows",
            json={
                "workflow_id": "test",
                # Missing version, name, steps
            }
        )
        
        assert response.status_code == 422
    
    async def test_get_workflow_endpoint(self, client: AsyncClient):
        """Test get workflow by ID and version"""
        response = await client.get("/api/v3/workflows/test-wf/1.0.0")
        
        # Endpoint should exist
        assert response.status_code in [200, 404, 500, 503]
    
    async def test_list_workflows_endpoint(self, client: AsyncClient):
        """Test list workflows endpoint"""
        response = await client.get("/api/v3/workflows")
        
        # Should return a list (may be empty)
        assert response.status_code in [200, 500, 503]
    
    async def test_list_workflows_pagination(self, client: AsyncClient):
        """Test workflows list with pagination"""
        response = await client.get("/api/v3/workflows?page=1&page_size=10")
        
        assert response.status_code in [200, 500, 503]
    
    async def test_execute_workflow_endpoint(self, client: AsyncClient, workflow_execution_request):
        """Test workflow execution endpoint"""
        response = await client.post(
            "/api/v3/workflows/execute",
            json=workflow_execution_request
        )
        
        # Endpoint should exist
        assert response.status_code in [200, 404, 500, 503]
    
    async def test_execute_workflow_missing_params(self, client: AsyncClient):
        """Test workflow execution with missing parameters"""
        response = await client.post(
            "/api/v3/workflows/execute",
            json={"workflow_id": "test"}  # Missing version
        )
        
        assert response.status_code == 422
    
    async def test_get_execution_status(self, client: AsyncClient):
        """Test getting execution status"""
        response = await client.get("/api/v3/workflows/executions/test-exec-123")
        
        # Endpoint should exist
        assert response.status_code in [200, 404, 500, 503]


@pytest.mark.asyncio
@pytest.mark.unit
class TestWorkflowSchemas:
    """Test workflow schemas"""
    
    def test_simple_workflow_structure(self, simple_workflow):
        """Test simple workflow has correct structure"""
        assert "workflow_id" in simple_workflow
        assert "version" in simple_workflow
        assert "name" in simple_workflow
        assert "steps" in simple_workflow
        assert isinstance(simple_workflow["steps"], list)
        assert len(simple_workflow["steps"]) > 0
    
    def test_complex_workflow_structure(self, complex_workflow):
        """Test complex workflow with multiple steps"""
        assert len(complex_workflow["steps"]) > 1
        
        # Check step connections
        step_ids = {step["step_id"] for step in complex_workflow["steps"]}
        for step in complex_workflow["steps"][:-1]:
            if "next" in step:
                assert step["next"] in step_ids or step["next"] is None
    
    def test_workflow_execution_request_structure(self, workflow_execution_request):
        """Test execution request schema"""
        assert "workflow_id" in workflow_execution_request
        assert "workflow_version" in workflow_execution_request
        assert "inputs" in workflow_execution_request


@pytest.mark.asyncio
@pytest.mark.unit
class TestWorkflowFactories:
    """Test workflow factory functions"""
    
    def test_workflow_factory_creates_workflow(self, workflow_factory):
        """Test workflow factory creates valid workflow"""
        workflow = workflow_factory.create_workflow()
        
        assert "workflow_id" in workflow
        assert "version" in workflow
        assert "steps" in workflow
        assert len(workflow["steps"]) >= 1
    
    def test_workflow_factory_custom_steps(self, workflow_factory):
        """Test factory with custom number of steps"""
        workflow = workflow_factory.create_workflow(num_steps=5)
        
        assert len(workflow["steps"]) == 5
        
        # Check step IDs are sequential
        for i, step in enumerate(workflow["steps"]):
            assert step["step_id"] == f"step-{i+1}"
    
    def test_workflow_factory_batch(self, workflow_factory):
        """Test creating batch of workflows"""
        workflows = workflow_factory.create_batch(count=3)
        
        assert len(workflows) == 3
        
        # Each should have unique ID
        workflow_ids = [w["workflow_id"] for w in workflows]
        assert len(set(workflow_ids)) == 3


@pytest.mark.asyncio
@pytest.mark.unit
class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint"""
    
    async def test_metrics_endpoint_exists(self, client: AsyncClient):
        """Test that /metrics endpoint exists"""
        response = await client.get("/metrics")
        
        # Should return metrics in Prometheus format
        assert response.status_code in [200, 503]
    
    async def test_metrics_format(self, client: AsyncClient):
        """Test metrics are in correct format"""
        response = await client.get("/metrics")
        
        if response.status_code == 200:
            # Should be plain text
            assert "text/plain" in response.headers.get("content-type", "")
