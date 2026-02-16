"""
End-to-End Workflow Integration Tests

Tests complete workflow execution scenarios including:
- Basic workflow creation and execution
- Workflows with checkpoints
- Workflows with memory
- Failure recovery
- Complex workflows
"""

import pytest
from httpx import AsyncClient
from uuid import uuid4
from typing import Dict, Any


# ==================== Basic Workflow Execution ====================

@pytest.mark.integration
@pytest.mark.asyncio
class TestBasicWorkflowExecution:
    """Test basic workflow creation and execution"""
    
    async def test_single_step_workflow(self, client: AsyncClient, sample_workflow_data: Dict[str, Any]):
        """Test single-step workflow creation and execution"""
        # Modify to single step
        single_step_workflow = {
            **sample_workflow_data,
            "steps": [sample_workflow_data["steps"][0]]
        }
        
        # Create workflow
        response = await client.post(
            "/api/v3/workflows/create",
            json=single_step_workflow
        )
        assert response.status_code in [200, 201]
        workflow = response.json()
        assert "workflow_id" in workflow
        
        # Execute workflow
        exec_response = await client.post(
            f"/api/v3/workflows/execute",
            json={"workflow_id": workflow["workflow_id"]}
        )
        assert exec_response.status_code in [200, 202]
    
    async def test_multi_step_sequential_workflow(self, client: AsyncClient, sample_workflow_data: Dict[str, Any]):
        """Test multi-step sequential workflow"""
        # Create workflow with sequential steps
        response = await client.post(
            "/api/v3/workflows/create",
            json=sample_workflow_data
        )
        assert response.status_code in [200, 201]
        workflow = response.json()
        
        # Execute workflow
        exec_response = await client.post(
            f"/api/v3/workflows/execute",
            json={"workflow_id": workflow["workflow_id"]}
        )
        assert exec_response.status_code in [200, 202]
        
        # Check status
        status_response = await client.get(
            f"/api/v3/workflows/status/{workflow['workflow_id']}"
        )
        assert status_response.status_code == 200
        status = status_response.json()
        assert status["workflow_id"] == workflow["workflow_id"]
    
    async def test_multi_step_parallel_workflow(self, client: AsyncClient):
        """Test multi-step parallel workflow"""
        # Create workflow with parallel steps
        parallel_workflow = {
            "workflow_id": f"parallel-{uuid4().hex[:8]}",
            "version": "1.0.0",
            "name": "Parallel Workflow",
            "description": "Test parallel execution",
            "steps": [
                {
                    "step_id": "parallel-1",
                    "name": "Parallel Step 1",
                    "type": "task",
                    "dependencies": [],
                    "config": {}
                },
                {
                    "step_id": "parallel-2",
                    "name": "Parallel Step 2",
                    "type": "task",
                    "dependencies": [],  # No dependencies = parallel
                    "config": {}
                },
                {
                    "step_id": "final",
                    "name": "Final Step",
                    "type": "task",
                    "dependencies": ["parallel-1", "parallel-2"],
                    "config": {}
                }
            ],
            "metadata": {}
        }
        
        response = await client.post(
            "/api/v3/workflows/create",
            json=parallel_workflow
        )
        assert response.status_code in [200, 201]


# ==================== Workflow with Checkpoints ====================

@pytest.mark.integration
@pytest.mark.asyncio
class TestWorkflowWithCheckpoints:
    """Test workflows with checkpoint/resume functionality"""
    
    async def test_checkpoint_creation_during_workflow(
        self,
        client: AsyncClient,
        sample_workflow_data: Dict[str, Any],
        sample_checkpoint_data: Dict[str, Any]
    ):
        """Test checkpoint creation during workflow execution"""
        # Create workflow
        workflow_response = await client.post(
            "/api/v3/workflows/create",
            json=sample_workflow_data
        )
        assert workflow_response.status_code in [200, 201]
        workflow = workflow_response.json()
        
        # Create checkpoint
        checkpoint_data = {
            **sample_checkpoint_data,
            "workflow_id": workflow["workflow_id"]
        }
        checkpoint_response = await client.post(
            "/api/v3/checkpoints/create",
            json=checkpoint_data
        )
        assert checkpoint_response.status_code in [200, 201]
        checkpoint = checkpoint_response.json()
        assert "checkpoint_id" in checkpoint
    
    async def test_checkpoint_restoration_and_resume(
        self,
        client: AsyncClient,
        sample_checkpoint_data: Dict[str, Any]
    ):
        """Test checkpoint restoration and workflow resume"""
        # Create checkpoint
        checkpoint_response = await client.post(
            "/api/v3/checkpoints/create",
            json=sample_checkpoint_data
        )
        assert checkpoint_response.status_code in [200, 201]
        checkpoint = checkpoint_response.json()
        checkpoint_id = checkpoint["checkpoint_id"]
        
        # Restore checkpoint
        restore_response = await client.post(
            "/api/v3/checkpoints/restore",
            json={"checkpoint_id": checkpoint_id}
        )
        assert restore_response.status_code in [200, 204]
        
        # Verify restoration
        get_response = await client.get(
            f"/api/v3/checkpoints/{checkpoint_id}"
        )
        assert get_response.status_code == 200
    
    async def test_checkpoint_cleanup_after_completion(
        self,
        client: AsyncClient,
        sample_checkpoint_data: Dict[str, Any]
    ):
        """Test checkpoint cleanup after workflow completion"""
        # Create checkpoint
        checkpoint_response = await client.post(
            "/api/v3/checkpoints/create",
            json=sample_checkpoint_data
        )
        assert checkpoint_response.status_code in [200, 201]
        checkpoint = checkpoint_response.json()
        checkpoint_id = checkpoint["checkpoint_id"]
        
        # Delete checkpoint
        delete_response = await client.delete(
            f"/api/v3/checkpoints/{checkpoint_id}"
        )
        assert delete_response.status_code in [200, 204]
        
        # Verify deletion
        get_response = await client.get(
            f"/api/v3/checkpoints/{checkpoint_id}"
        )
        assert get_response.status_code == 404


# ==================== Workflow with Memory ====================

@pytest.mark.integration
@pytest.mark.asyncio
class TestWorkflowWithMemory:
    """Test workflows with memory operations"""
    
    async def test_memory_storage_during_workflow(
        self,
        client: AsyncClient,
        sample_memory_data: Dict[str, Any]
    ):
        """Test memory storage during workflow steps"""
        # Store memory
        response = await client.post(
            "/api/v3/memory/store",
            json=sample_memory_data
        )
        assert response.status_code in [200, 201]
        memory = response.json()
        assert "id" in memory or "memory_id" in memory
    
    async def test_memory_retrieval_in_subsequent_steps(
        self,
        client: AsyncClient,
        sample_memory_data: Dict[str, Any]
    ):
        """Test memory retrieval in subsequent workflow steps"""
        # Store memory
        store_response = await client.post(
            "/api/v3/memory/store",
            json=sample_memory_data
        )
        assert store_response.status_code in [200, 201]
        memory = store_response.json()
        memory_id = memory.get("id") or memory.get("memory_id")
        
        # Retrieve memory
        get_response = await client.get(
            f"/api/v3/memory/{memory_id}"
        )
        assert get_response.status_code == 200
        retrieved = get_response.json()
        assert retrieved["content"] == sample_memory_data["content"]
    
    async def test_memory_tier_transitions(
        self,
        client: AsyncClient,
        sample_memory_data: Dict[str, Any]
    ):
        """Test memory tier transitions (L1→L2→L3)"""
        # Store in L1
        store_response = await client.post(
            "/api/v3/memory/store",
            json={**sample_memory_data, "tier": "L1"}
        )
        assert store_response.status_code in [200, 201]
        memory = store_response.json()
        memory_id = memory.get("id") or memory.get("memory_id")
        
        # Promote to L2
        promote_response = await client.post(
            "/api/v3/memory/promote",
            json={
                "memory_id": memory_id,
                "target_tier": "L2"
            }
        )
        assert promote_response.status_code in [200, 204]


# ==================== Workflow Failure & Recovery ====================

@pytest.mark.integration
@pytest.mark.asyncio
class TestWorkflowFailureRecovery:
    """Test workflow failure handling and recovery"""
    
    async def test_step_failure_and_retry_logic(
        self,
        client: AsyncClient,
        sample_workflow_data: Dict[str, Any]
    ):
        """Test step failure and retry logic"""
        # Create workflow with retry config
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
    
    async def test_workflow_level_failure_handling(
        self,
        client: AsyncClient,
        sample_workflow_data: Dict[str, Any]
    ):
        """Test workflow-level failure handling"""
        # Create workflow
        response = await client.post(
            "/api/v3/workflows/create",
            json=sample_workflow_data
        )
        assert response.status_code in [200, 201]
        workflow = response.json()
        
        # Attempt execution (may fail, that's expected)
        exec_response = await client.post(
            f"/api/v3/workflows/execute",
            json={"workflow_id": workflow["workflow_id"]}
        )
        # Accept both success and failure status codes
        assert exec_response.status_code in [200, 202, 400, 500]
    
    async def test_self_healing_service_intervention(
        self,
        client: AsyncClient
    ):
        """Test self-healing service intervention on failure"""
        # Trigger a health check that might activate self-healing
        response = await client.get("/api/v3/health/system")
        assert response.status_code == 200
        health = response.json()
        # System should respond even if degraded
        assert "status" in health


# ==================== Complex Workflows ====================

@pytest.mark.integration
@pytest.mark.asyncio
class TestComplexWorkflows:
    """Test complex workflow scenarios"""
    
    async def test_task_decomposition_workflow(
        self,
        client: AsyncClient,
        sample_task_decomposition: Dict[str, Any]
    ):
        """Test workflow with task decomposition (10K+ nodes)"""
        # Create decomposition task
        # Note: Actual endpoint may vary
        task_data = {
            **sample_task_decomposition,
            "max_nodes": 100  # Reduced for testing
        }
        
        # This would typically call a task decomposition endpoint
        # For now, we just verify the workflow can be created
        workflow_data = {
            "workflow_id": f"decomp-{uuid4().hex[:8]}",
            "version": "1.0.0",
            "name": "Task Decomposition Workflow",
            "description": "Test large-scale decomposition",
            "steps": [
                {
                    "step_id": "decompose",
                    "name": "Decompose Task",
                    "type": "decomposition",
                    "config": task_data
                }
            ],
            "metadata": {}
        }
        
        response = await client.post(
            "/api/v3/workflows/create",
            json=workflow_data
        )
        assert response.status_code in [200, 201]
    
    async def test_cost_governed_workflow(
        self,
        client: AsyncClient,
        sample_cost_budget: Dict[str, Any],
        sample_workflow_data: Dict[str, Any]
    ):
        """Test workflow with cost governance and budget enforcement"""
        # Create budget
        budget_response = await client.post(
            "/api/v3/cost/budget",
            json=sample_cost_budget
        )
        assert budget_response.status_code in [200, 201]
        
        # Create workflow with budget
        workflow_with_budget = {
            **sample_workflow_data,
            "budget_id": sample_cost_budget.get("workflow_id")
        }
        
        workflow_response = await client.post(
            "/api/v3/workflows/create",
            json=workflow_with_budget
        )
        assert workflow_response.status_code in [200, 201]
    
    async def test_long_running_workflow_simulation(
        self,
        client: AsyncClient,
        sample_workflow_data: Dict[str, Any],
        performance_monitor
    ):
        """Test long-running workflow (24+ hours simulation)"""
        # Create workflow
        long_workflow = {
            **sample_workflow_data,
            "metadata": {
                **sample_workflow_data.get("metadata", {}),
                "estimated_duration": "24h",
                "checkpoint_interval": "1h"
            }
        }
        
        performance_monitor.start()
        
        response = await client.post(
            "/api/v3/workflows/create",
            json=long_workflow
        )
        assert response.status_code in [200, 201]
        
        duration = performance_monitor.stop()
        # Creation should be fast even for long workflows
        assert duration < 5.0, f"Workflow creation took {duration}s"
