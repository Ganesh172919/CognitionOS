"""
Workflow Engine - Executor

Executes workflows step-by-step based on the DAG.
"""

import asyncio
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from ..models import (
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowExecutionStep,
    ExecutionStatus,
    WorkflowStep,
    WorkflowStepType,
    AgentRole
)


class WorkflowExecutor:
    """
    Execute workflows step-by-step.

    Handles:
    - DAG-based execution (parallel when possible)
    - Template variable substitution
    - Error handling and retries
    - Conditional execution
    """

    def __init__(self, service_clients: Dict[str, Any]):
        """
        Initialize executor with service clients.

        Args:
            service_clients: Dictionary of service clients (task_planner, agent_orchestrator, etc.)
        """
        self.service_clients = service_clients

    async def execute(
        self,
        workflow: WorkflowDefinition,
        inputs: Dict[str, Any],
        user_id: Optional[UUID] = None
    ) -> WorkflowExecution:
        """
        Execute a workflow.

        Args:
            workflow: Workflow definition
            inputs: Input values
            user_id: User executing the workflow

        Returns:
            WorkflowExecution object with results
        """
        # Create execution record
        execution = WorkflowExecution(
            workflow_id=workflow.id,
            workflow_version=workflow.version,
            inputs=inputs,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.utcnow(),
            user_id=user_id
        )

        # Build execution context
        context = {
            "inputs": inputs,
            "outputs": {},
            "steps": {}  # Step outputs
        }

        try:
            # Execute steps in topological order
            await self._execute_dag(workflow.steps, execution, context)

            # Extract workflow outputs
            execution.outputs = self._extract_outputs(workflow, context)
            execution.status = ExecutionStatus.COMPLETED
            execution.completed_at = datetime.utcnow()

        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.utcnow()
            raise

        return execution

    async def _execute_dag(
        self,
        steps: List[WorkflowStep],
        execution: WorkflowExecution,
        context: Dict[str, Any]
    ):
        """
        Execute workflow steps in DAG order.

        Executes steps in parallel when possible (no dependencies between them).
        """
        # Build dependency graph
        step_map = {step.id: step for step in steps}
        completed = set()
        running = set()

        while len(completed) < len(steps):
            # Find steps ready to execute
            ready = []
            for step in steps:
                if step.id not in completed and step.id not in running:
                    # Check if all dependencies are completed
                    if all(dep in completed for dep in step.depends_on):
                        ready.append(step)

            if not ready:
                # No steps ready - either all done or deadlock
                if running:
                    # Wait for running steps to complete
                    await asyncio.sleep(0.1)
                    continue
                else:
                    # Deadlock - should not happen with DAG validation
                    raise RuntimeError("Workflow deadlock detected")

            # Execute ready steps in parallel
            tasks = []
            for step in ready:
                running.add(step.id)
                task = self._execute_step(step, execution, context)
                tasks.append((step.id, task))

            # Wait for all ready steps to complete
            for step_id, task in tasks:
                try:
                    await task
                    completed.add(step_id)
                except Exception as e:
                    # Step failed - mark execution as failed
                    raise RuntimeError(f"Step '{step_id}' failed: {e}")
                finally:
                    running.discard(step_id)

    async def _execute_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
        context: Dict[str, Any]
    ):
        """
        Execute a single workflow step.

        Handles:
        - Template variable substitution
        - Conditional execution
        - Retries
        - Approval (if required)
        """
        # Check condition
        if step.condition:
            if not self._evaluate_condition(step.condition, context):
                # Skip this step
                return

        # Substitute template variables in parameters
        params = self._substitute_templates(step.params, context)

        # Create step execution record
        step_execution = WorkflowExecutionStep(
            execution_id=execution.id,
            step_id=step.id,
            step_type=step.type,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.utcnow()
        )

        # Execute with retries
        retry_count = 0
        while retry_count <= step.retry:
            try:
                # Check if approval required
                if step.approval_required:
                    await self._wait_for_approval(step, params)

                # Execute step
                output = await self._execute_step_type(step.type, params, step.agent_role)

                # Store output
                step_execution.output = output
                step_execution.status = ExecutionStatus.COMPLETED
                step_execution.completed_at = datetime.utcnow()

                # Add to context
                context["steps"][step.id] = output

                break  # Success - exit retry loop

            except Exception as e:
                retry_count += 1
                step_execution.retry_count = retry_count

                if retry_count > step.retry:
                    # Max retries exceeded - fail
                    step_execution.status = ExecutionStatus.FAILED
                    step_execution.error = str(e)
                    step_execution.completed_at = datetime.utcnow()
                    raise
                else:
                    # Retry after delay
                    delay_seconds = self._parse_duration(step.retry_delay or "5s")
                    await asyncio.sleep(delay_seconds)

    async def _execute_step_type(
        self,
        step_type: WorkflowStepType,
        params: Dict[str, Any],
        agent_role: Optional[AgentRole] = None
    ) -> Dict[str, Any]:
        """
        Execute specific step type.

        Routes to appropriate service based on step type.
        """
        if step_type == WorkflowStepType.EXECUTE_PYTHON:
            return await self._execute_python(params)
        elif step_type == WorkflowStepType.EXECUTE_JAVASCRIPT:
            return await self._execute_javascript(params)
        elif step_type == WorkflowStepType.HTTP_REQUEST:
            return await self._execute_http_request(params)
        elif step_type == WorkflowStepType.QUERY_DATABASE:
            return await self._execute_query_database(params)
        elif step_type == WorkflowStepType.AI_GENERATE:
            return await self._execute_ai_generate(params, agent_role)
        elif step_type == WorkflowStepType.MEMORY_STORE:
            return await self._execute_memory_store(params)
        elif step_type == WorkflowStepType.MEMORY_RETRIEVE:
            return await self._execute_memory_retrieve(params)
        else:
            raise NotImplementedError(f"Step type '{step_type}' not yet implemented")

    async def _execute_python(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python code via tool-runner service"""
        tool_runner = self.service_clients.get("tool_runner")
        if not tool_runner:
            raise ValueError("Tool runner service client not configured")

        result = await tool_runner.post("/tools/execute_python", json=params)
        return result

    async def _execute_javascript(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute JavaScript code via tool-runner service"""
        tool_runner = self.service_clients.get("tool_runner")
        if not tool_runner:
            raise ValueError("Tool runner service client not configured")

        result = await tool_runner.post("/tools/execute_javascript", json=params)
        return result

    async def _execute_http_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HTTP request via tool-runner service"""
        tool_runner = self.service_clients.get("tool_runner")
        if not tool_runner:
            raise ValueError("Tool runner service client not configured")

        result = await tool_runner.post("/tools/http_request", json=params)
        return result

    async def _execute_query_database(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute database query"""
        # This would connect to the database directly
        # For now, stub implementation
        return {"rows": [], "count": 0}

    async def _execute_ai_generate(self, params: Dict[str, Any], agent_role: Optional[AgentRole]) -> Dict[str, Any]:
        """Generate AI output via ai-runtime service"""
        ai_runtime = self.service_clients.get("ai_runtime")
        if not ai_runtime:
            raise ValueError("AI runtime service client not configured")

        result = await ai_runtime.post("/complete", json={
            "role": agent_role.value if agent_role else "executor",
            "prompt": params.get("prompt"),
            **params
        })
        return result

    async def _execute_memory_store(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Store memory via memory-service"""
        memory_service = self.service_clients.get("memory_service")
        if not memory_service:
            raise ValueError("Memory service client not configured")

        result = await memory_service.post("/memories", json=params)
        return result

    async def _execute_memory_retrieve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve memories via memory-service"""
        memory_service = self.service_clients.get("memory_service")
        if not memory_service:
            raise ValueError("Memory service client not configured")

        result = await memory_service.post("/retrieve", json=params)
        return result

    def _substitute_templates(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Substitute template variables in parameters.

        Supports:
        - ${{ inputs.repo_url }} - Input variables
        - ${{ steps.clone_repo.output }} - Step outputs
        - ${{ outputs.deployment_url }} - Output variables
        """
        def substitute(value: Any) -> Any:
            if isinstance(value, str):
                # Find all template variables
                pattern = r'\$\{\{\s*([^}]+)\s*\}\}'
                matches = re.findall(pattern, value)

                for match in matches:
                    # Parse variable path (e.g., "inputs.repo_url" or "steps.clone_repo.output")
                    parts = match.strip().split('.')

                    # Navigate context
                    current = context
                    for part in parts:
                        if isinstance(current, dict) and part in current:
                            current = current[part]
                        else:
                            raise ValueError(f"Template variable not found: {match}")

                    # Replace template with value
                    value = value.replace(f"${{{{{match}}}}}", str(current))

                return value

            elif isinstance(value, dict):
                return {k: substitute(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute(item) for item in value]
            else:
                return value

        return substitute(params)

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate conditional expression.

        Simple evaluation for now (can be enhanced with full expression parser).

        Examples:
        - ${{ steps.run_tests.output.success }} == true
        - ${{ inputs.environment }} == 'prod'
        """
        # Substitute templates
        expr = self._substitute_templates({"expr": condition}, context)["expr"]

        # Evaluate (using Python's eval - CAUTION: only for trusted workflows)
        try:
            return bool(eval(expr))
        except Exception:
            return False

    def _parse_duration(self, duration: str) -> int:
        """
        Parse duration string to seconds.

        Examples: "5s", "1m", "2h"
        """
        match = re.match(r'^(\d+)([smh])$', duration)
        if not match:
            raise ValueError(f"Invalid duration format: {duration}")

        value = int(match.group(1))
        unit = match.group(2)

        if unit == 's':
            return value
        elif unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 3600
        else:
            raise ValueError(f"Unknown duration unit: {unit}")

    def _extract_outputs(self, workflow: WorkflowDefinition, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract workflow outputs from context.

        Outputs can reference step outputs or inputs.
        """
        outputs = {}

        for output_def in workflow.outputs:
            # Outputs are typically defined as step references
            # For now, just return the context
            # In a real implementation, this would map output definitions to context values
            pass

        return outputs

    async def _wait_for_approval(self, step: WorkflowStep, params: Dict[str, Any]):
        """
        Wait for manual approval.

        In a real implementation, this would:
        1. Send notification to user
        2. Wait for user approval via API
        3. Continue execution on approval

        For now, just a placeholder.
        """
        # Stub implementation - auto-approve for now
        print(f"Step '{step.id}' requires approval. Auto-approving...")
        await asyncio.sleep(0.1)
