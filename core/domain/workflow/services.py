"""
Workflow Domain - Domain Services

Domain services contain business logic that doesn't naturally fit in entities.
These are stateless and operate on domain entities.
"""

from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4

from .entities import (
    Workflow,
    WorkflowExecution,
    WorkflowStep,
    StepExecution,
    StepId,
    ExecutionStatus
)


class WorkflowValidator:
    """
    Domain service for workflow validation.

    Validates workflow definitions beyond basic entity invariants.
    """

    @staticmethod
    def validate_dag(workflow: Workflow) -> tuple[bool, Optional[str]]:
        """
        Validate workflow DAG structure.

        Returns:
            (is_valid, error_message)
        """
        try:
            workflow.get_execution_order()
            return (True, None)
        except ValueError as e:
            return (False, str(e))

    @staticmethod
    def validate_template_references(workflow: Workflow) -> tuple[bool, Optional[str]]:
        """
        Validate that template references in steps are valid.

        Checks:
        - References to inputs exist
        - References to step outputs reference valid steps
        - References are to steps that come before in DAG

        Returns:
            (is_valid, error_message)
        """
        # Get execution order to ensure references are valid
        try:
            execution_order = workflow.get_execution_order()
        except ValueError as e:
            return (False, f"Invalid DAG: {e}")

        # Build map of step position in execution order
        step_position = {step_id: idx for idx, step_id in enumerate(execution_order)}

        # Check each step's params for template references
        for step in workflow.steps:
            # Extract template variables from params
            templates = WorkflowValidator._extract_templates(step.params)

            for template in templates:
                # Check if it's a step output reference
                if template.startswith("steps."):
                    parts = template.split(".")
                    if len(parts) < 3:
                        return (False, f"Invalid template reference in step {step.id.value}: {template}")

                    referenced_step_id = StepId(parts[1])

                    # Check step exists
                    if referenced_step_id not in step_position:
                        return (
                            False,
                            f"Step {step.id.value} references non-existent step: {referenced_step_id.value}"
                        )

                    # Check step comes before in execution order
                    if step_position[referenced_step_id] >= step_position[step.id]:
                        return (
                            False,
                            f"Step {step.id.value} references step {referenced_step_id.value} "
                            f"that comes after it in execution order"
                        )

        return (True, None)

    @staticmethod
    def _extract_templates(params: Dict) -> List[str]:
        """
        Extract template variable references from params.

        Example: "${{ steps.build.output.image }}" -> "steps.build.output.image"
        """
        import re
        templates = []

        def extract_from_value(value):
            if isinstance(value, str):
                # Match ${{ variable.path }}
                matches = re.findall(r'\$\{\{\s*([^}]+)\s*\}\}', value)
                templates.extend(matches)
            elif isinstance(value, dict):
                for v in value.values():
                    extract_from_value(v)
            elif isinstance(value, list):
                for item in value:
                    extract_from_value(item)

        extract_from_value(params)
        return templates


class WorkflowExecutionOrchestrator:
    """
    Domain service for workflow execution orchestration.

    Determines which steps can execute based on DAG and execution state.
    """

    @staticmethod
    def get_ready_steps(
        workflow: Workflow,
        execution: WorkflowExecution,
        step_executions: List[StepExecution]
    ) -> List[WorkflowStep]:
        """
        Determine which workflow steps are ready to execute.

        A step is ready if:
        1. All its dependencies have completed successfully
        2. It hasn't been executed yet
        3. It's not currently running

        Args:
            workflow: Workflow definition
            execution: Current workflow execution
            step_executions: Current step execution states

        Returns:
            List of steps ready to execute
        """
        # Build map of step execution states
        step_status = {
            step_exec.step_id: step_exec.status
            for step_exec in step_executions
        }

        # Get steps that haven't started yet
        pending_steps = [
            step for step in workflow.steps
            if step.id not in step_status or step_status[step.id] == ExecutionStatus.PENDING
        ]

        # Filter to steps whose dependencies are all completed
        ready_steps = []
        for step in pending_steps:
            dependencies_met = all(
                dep in step_status and step_status[dep] == ExecutionStatus.COMPLETED
                for dep in step.depends_on
            )
            if dependencies_met:
                ready_steps.append(step)

        return ready_steps

    @staticmethod
    def can_continue_execution(
        workflow: Workflow,
        step_executions: List[StepExecution]
    ) -> bool:
        """
        Determine if workflow execution can continue.

        Execution can continue if there are steps that:
        - Are pending or running
        - Can potentially become ready (dependencies might complete)

        Returns:
            True if execution should continue, False if terminal
        """
        # Build step status map
        step_status = {
            step_exec.step_id: step_exec.status
            for step_exec in step_executions
        }

        # Check if any steps are still running
        if any(status == ExecutionStatus.RUNNING for status in step_status.values()):
            return True

        # Check if any pending steps have all dependencies met or pending
        for step in workflow.steps:
            current_status = step_status.get(step.id, ExecutionStatus.PENDING)

            if current_status == ExecutionStatus.PENDING:
                # Check if all dependencies are completed or pending
                dependencies_ok = all(
                    step_status.get(dep, ExecutionStatus.PENDING) in [
                        ExecutionStatus.COMPLETED,
                        ExecutionStatus.PENDING,
                        ExecutionStatus.RUNNING
                    ]
                    for dep in step.depends_on
                )
                if dependencies_ok:
                    return True

        return False

    @staticmethod
    def is_execution_complete(
        workflow: Workflow,
        step_executions: List[StepExecution]
    ) -> tuple[bool, bool, Optional[str]]:
        """
        Determine if workflow execution is complete.

        Returns:
            (is_complete, is_successful, error_message)
        """
        # Build step status map
        step_status = {
            step_exec.step_id: step_exec.status
            for step_exec in step_executions
        }

        # Check if all steps have terminal status
        all_terminal = all(
            step_status.get(step.id, ExecutionStatus.PENDING) in [
                ExecutionStatus.COMPLETED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
                ExecutionStatus.SKIPPED
            ]
            for step in workflow.steps
        )

        if not all_terminal:
            return (False, False, None)

        # Check if any steps failed
        failed_steps = [
            step.id.value for step in workflow.steps
            if step_status.get(step.id) == ExecutionStatus.FAILED
        ]

        if failed_steps:
            return (
                True,
                False,
                f"Workflow failed due to failed steps: {', '.join(failed_steps)}"
            )

        # All steps completed or skipped
        return (True, True, None)

    @staticmethod
    def get_blocked_steps(
        workflow: Workflow,
        step_executions: List[StepExecution]
    ) -> List[StepId]:
        """
        Get steps that are blocked by failed dependencies.

        Args:
            workflow: Workflow definition
            step_executions: Current step execution states

        Returns:
            List of step IDs that are blocked
        """
        step_status = {
            step_exec.step_id: step_exec.status
            for step_exec in step_executions
        }

        blocked = []
        for step in workflow.steps:
            # Skip if already executed
            if step.id in step_status and step_status[step.id] != ExecutionStatus.PENDING:
                continue

            # Check if any dependency failed
            has_failed_dependency = any(
                step_status.get(dep) == ExecutionStatus.FAILED
                for dep in step.depends_on
            )

            if has_failed_dependency:
                blocked.append(step.id)

        return blocked


class WorkflowDagAnalyzer:
    """
    Domain service for analyzing workflow DAG structure.

    Provides insights into workflow complexity and structure.
    """

    @staticmethod
    def get_critical_path(workflow: Workflow) -> List[StepId]:
        """
        Calculate critical path through workflow DAG.

        Critical path is the longest path from start to finish,
        determining minimum execution time.

        Returns:
            List of step IDs on the critical path
        """
        # Build reverse dependency graph
        step_map = {step.id: step for step in workflow.steps}
        reverse_deps = {step.id: [] for step in workflow.steps}

        for step in workflow.steps:
            for dep in step.depends_on:
                reverse_deps[dep].append(step.id)

        # Calculate longest path from each step (dynamic programming)
        longest_path = {}
        path_steps = {}

        def calculate_longest_path(step_id: StepId) -> int:
            if step_id in longest_path:
                return longest_path[step_id]

            step = step_map[step_id]

            # Base case: no dependents
            if not reverse_deps[step_id]:
                longest_path[step_id] = 1
                path_steps[step_id] = [step_id]
                return 1

            # Recursive case: 1 + max of dependent paths
            max_length = 0
            max_path = []

            for dependent in reverse_deps[step_id]:
                length = calculate_longest_path(dependent)
                if length > max_length:
                    max_length = length
                    max_path = path_steps[dependent]

            longest_path[step_id] = 1 + max_length
            path_steps[step_id] = [step_id] + max_path

            return longest_path[step_id]

        # Calculate for all independent steps
        independent_steps = workflow.get_independent_steps()
        max_critical_path = []
        max_length = 0

        for step in independent_steps:
            length = calculate_longest_path(step.id)
            if length > max_length:
                max_length = length
                max_critical_path = path_steps[step.id]

        return max_critical_path

    @staticmethod
    def get_parallelism_potential(workflow: Workflow) -> int:
        """
        Calculate maximum parallelism of workflow.

        Returns the maximum number of steps that can run simultaneously.

        Returns:
            Maximum number of parallel steps
        """
        execution_order = workflow.get_execution_order()
        step_map = {step.id: step for step in workflow.steps}

        # Build dependency map
        step_status = {step_id: ExecutionStatus.PENDING for step_id in execution_order}

        max_parallel = 0

        # Simulate execution
        while any(status == ExecutionStatus.PENDING for status in step_status.values()):
            # Count steps that can run now
            ready_count = 0

            for step_id in execution_order:
                if step_status[step_id] != ExecutionStatus.PENDING:
                    continue

                step = step_map[step_id]
                dependencies_met = all(
                    step_status.get(dep) == ExecutionStatus.COMPLETED
                    for dep in step.depends_on
                )

                if dependencies_met:
                    ready_count += 1

            max_parallel = max(max_parallel, ready_count)

            # Mark ready steps as completed for next iteration
            for step_id in execution_order:
                if step_status[step_id] != ExecutionStatus.PENDING:
                    continue

                step = step_map[step_id]
                dependencies_met = all(
                    step_status.get(dep) == ExecutionStatus.COMPLETED
                    for dep in step.depends_on
                )

                if dependencies_met:
                    step_status[step_id] = ExecutionStatus.COMPLETED

        return max_parallel

    @staticmethod
    def estimate_min_execution_time(workflow: Workflow, avg_step_time_seconds: int = 60) -> int:
        """
        Estimate minimum execution time based on critical path.

        Args:
            workflow: Workflow to analyze
            avg_step_time_seconds: Average time per step

        Returns:
            Estimated minimum execution time in seconds
        """
        critical_path = WorkflowDagAnalyzer.get_critical_path(workflow)
        return len(critical_path) * avg_step_time_seconds
