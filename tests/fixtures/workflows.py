"""
Test Fixtures for Workflows

Provides fixtures and factories for workflow-related testing.
"""

import pytest
from typing import Dict, Any, List
from datetime import datetime


@pytest.fixture
def simple_workflow():
    """Simple workflow with one step"""
    return {
        "workflow_id": "simple-workflow",
        "version": "1.0.0",
        "name": "Simple Workflow",
        "description": "A simple test workflow",
        "steps": [
            {
                "step_id": "step-1",
                "name": "Single Step",
                "type": "task",
                "config": {
                    "action": "log",
                    "message": "Hello World"
                }
            }
        ],
        "metadata": {
            "created_by": "test-user",
            "tags": ["test"]
        }
    }


@pytest.fixture
def complex_workflow():
    """Complex workflow with multiple steps"""
    return {
        "workflow_id": "complex-workflow",
        "version": "1.0.0",
        "name": "Complex Workflow",
        "description": "A complex multi-step workflow",
        "steps": [
            {
                "step_id": "step-1",
                "name": "Fetch Data",
                "type": "api_call",
                "config": {
                    "url": "https://api.example.com/data"
                },
                "next": "step-2"
            },
            {
                "step_id": "step-2",
                "name": "Process Data",
                "type": "transform",
                "config": {
                    "operation": "filter"
                },
                "next": "step-3"
            },
            {
                "step_id": "step-3",
                "name": "Save Results",
                "type": "database",
                "config": {
                    "operation": "insert"
                }
            }
        ],
        "metadata": {
            "created_by": "test-user",
            "tags": ["test", "complex"]
        }
    }


@pytest.fixture
def workflow_execution_request():
    """Workflow execution request"""
    return {
        "workflow_id": "test-workflow",
        "workflow_version": "1.0.0",
        "inputs": {
            "param1": "value1",
            "param2": 42
        },
        "execution_mode": "async"
    }


@pytest.fixture
def workflow_with_llm_step():
    """Workflow with LLM processing step"""
    return {
        "workflow_id": "llm-workflow",
        "version": "1.0.0",
        "name": "LLM Workflow",
        "description": "Workflow with LLM step",
        "steps": [
            {
                "step_id": "llm-step",
                "name": "AI Processing",
                "type": "llm",
                "config": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "prompt": "Analyze this data: {{ inputs.data }}"
                }
            }
        ],
        "metadata": {}
    }


class WorkflowFactory:
    """Factory for creating test workflows"""
    
    @staticmethod
    def create_workflow(
        workflow_id: str = "factory-workflow",
        version: str = "1.0.0",
        name: str = "Factory Workflow",
        num_steps: int = 1,
    ) -> Dict[str, Any]:
        """Create a workflow with specified number of steps"""
        steps = [
            {
                "step_id": f"step-{i+1}",
                "name": f"Step {i+1}",
                "type": "task",
                "config": {},
                "next": f"step-{i+2}" if i < num_steps - 1 else None
            }
            for i in range(num_steps)
        ]
        
        # Remove 'next' from last step
        if steps:
            steps[-1].pop("next", None)
        
        return {
            "workflow_id": workflow_id,
            "version": version,
            "name": name,
            "description": f"Factory-created workflow with {num_steps} steps",
            "steps": steps,
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "factory": True
            }
        }
    
    @staticmethod
    def create_batch(count: int = 5) -> List[Dict[str, Any]]:
        """Create multiple test workflows"""
        return [
            WorkflowFactory.create_workflow(
                workflow_id=f"workflow-{i}",
                name=f"Workflow {i}",
                num_steps=i % 3 + 1  # 1-3 steps
            )
            for i in range(count)
        ]


@pytest.fixture
def workflow_factory():
    """Workflow factory fixture"""
    return WorkflowFactory
