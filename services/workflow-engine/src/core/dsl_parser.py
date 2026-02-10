"""
Workflow Engine - DSL Parser

Parses YAML/JSON workflow definitions into validated Pydantic models.
"""

import yaml
import json
from typing import Dict, Any
from pathlib import Path

from ..models import (
    WorkflowDefinition,
    WorkflowInput,
    WorkflowOutput,
    WorkflowStep,
    WorkflowInputType,
    WorkflowOutputType,
    WorkflowStepType,
    AgentRole
)


class WorkflowDSLParser:
    """
    Parse workflow DSL from YAML or JSON files.

    Validates the workflow definition and returns a WorkflowDefinition object.
    """

    def parse_file(self, filepath: str) -> WorkflowDefinition:
        """
        Parse workflow from YAML or JSON file.

        Args:
            filepath: Path to workflow definition file

        Returns:
            WorkflowDefinition object

        Raises:
            ValueError: If file format is invalid or validation fails
        """
        path = Path(filepath)

        if not path.exists():
            raise ValueError(f"Workflow file not found: {filepath}")

        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}. Use .yaml, .yml, or .json")

        return self.parse_dict(data)

    def parse_dict(self, data: Dict[str, Any]) -> WorkflowDefinition:
        """
        Parse workflow from dictionary (already loaded YAML/JSON).

        Args:
            data: Workflow definition as dictionary

        Returns:
            WorkflowDefinition object

        Raises:
            ValueError: If validation fails
        """
        # Extract workflow metadata
        if 'workflow' not in data:
            raise ValueError("Invalid workflow DSL: missing 'workflow' key")

        workflow_data = data['workflow']

        # Parse inputs
        inputs = []
        for input_def in workflow_data.get('inputs', []):
            inputs.append(self._parse_input(input_def))

        # Parse outputs
        outputs = []
        for output_def in workflow_data.get('outputs', []):
            outputs.append(self._parse_output(output_def))

        # Parse steps
        steps = []
        for step_def in workflow_data.get('steps', []):
            steps.append(self._parse_step(step_def))

        # Validate step dependencies (no cycles)
        self._validate_dag(steps)

        # Create WorkflowDefinition
        workflow = WorkflowDefinition(
            id=workflow_data['id'],
            version=workflow_data['version'],
            name=workflow_data.get('name', workflow_data['id']),
            description=workflow_data.get('description'),
            schedule=workflow_data.get('schedule'),
            inputs=inputs,
            outputs=outputs,
            steps=steps,
            tags=workflow_data.get('tags', []),
            created_by=workflow_data.get('created_by')
        )

        return workflow

    def _parse_input(self, input_def: Dict[str, Any]) -> WorkflowInput:
        """Parse workflow input definition"""
        return WorkflowInput(
            name=input_def['name'],
            type=WorkflowInputType(input_def['type']),
            required=input_def.get('required', True),
            default=input_def.get('default'),
            description=input_def.get('description'),
            values=input_def.get('values')
        )

    def _parse_output(self, output_def: Dict[str, Any]) -> WorkflowOutput:
        """Parse workflow output definition"""
        return WorkflowOutput(
            name=output_def['name'],
            type=WorkflowOutputType(output_def['type']),
            description=output_def.get('description')
        )

    def _parse_step(self, step_def: Dict[str, Any]) -> WorkflowStep:
        """Parse workflow step definition"""
        # Parse agent role if specified
        agent_role = None
        if 'agent_role' in step_def:
            agent_role = AgentRole(step_def['agent_role'])

        return WorkflowStep(
            id=step_def['id'],
            type=WorkflowStepType(step_def['type']),
            name=step_def.get('name'),
            description=step_def.get('description'),
            depends_on=step_def.get('depends_on', []),
            params=step_def.get('params', {}),
            agent_role=agent_role,
            timeout=step_def.get('timeout', '300s'),
            retry=step_def.get('retry', 0),
            retry_delay=step_def.get('retry_delay', '5s'),
            condition=step_def.get('condition'),
            approval_required=step_def.get('approval_required', False)
        )

    def _validate_dag(self, steps: list[WorkflowStep]):
        """
        Validate that workflow steps form a valid DAG (no cycles).

        Args:
            steps: List of workflow steps

        Raises:
            ValueError: If cycle detected
        """
        # Build dependency graph
        graph = {step.id: step.depends_on for step in steps}

        # Check each step
        for step_id in graph:
            visited = set()
            self._detect_cycle(step_id, graph, visited, set())

    def _detect_cycle(self, node: str, graph: Dict[str, list], visited: set, rec_stack: set):
        """
        Detect cycles in dependency graph using DFS.

        Args:
            node: Current node
            graph: Adjacency list
            visited: Set of visited nodes
            rec_stack: Recursion stack for cycle detection

        Raises:
            ValueError: If cycle detected
        """
        visited.add(node)
        rec_stack.add(node)

        # Visit dependencies
        for dep in graph.get(node, []):
            if dep not in visited:
                self._detect_cycle(dep, graph, visited, rec_stack)
            elif dep in rec_stack:
                raise ValueError(f"Cycle detected in workflow: {node} -> {dep}")

        rec_stack.remove(node)

    def validate_inputs(self, workflow: WorkflowDefinition, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input values against workflow input definitions.

        Args:
            workflow: Workflow definition
            inputs: Input values to validate

        Returns:
            Validated and type-coerced inputs

        Raises:
            ValueError: If validation fails
        """
        validated = {}

        # Check all required inputs are provided
        for input_def in workflow.inputs:
            if input_def.required and input_def.name not in inputs and input_def.default is None:
                raise ValueError(f"Required input '{input_def.name}' not provided")

            # Get value (from inputs or default)
            value = inputs.get(input_def.name, input_def.default)

            if value is None:
                continue

            # Type validation
            if input_def.type == WorkflowInputType.STRING:
                validated[input_def.name] = str(value)
            elif input_def.type == WorkflowInputType.INTEGER:
                validated[input_def.name] = int(value)
            elif input_def.type == WorkflowInputType.FLOAT:
                validated[input_def.name] = float(value)
            elif input_def.type == WorkflowInputType.BOOLEAN:
                validated[input_def.name] = bool(value)
            elif input_def.type == WorkflowInputType.ENUM:
                if value not in input_def.values:
                    raise ValueError(f"Invalid value for '{input_def.name}': {value}. Allowed: {input_def.values}")
                validated[input_def.name] = value
            elif input_def.type == WorkflowInputType.JSON:
                validated[input_def.name] = value  # Already dict/list

        return validated


# Example usage
if __name__ == "__main__":
    parser = WorkflowDSLParser()

    # Example workflow DSL
    workflow_yaml = """
workflow:
  id: "example-workflow"
  version: "1.0.0"
  name: "Example Workflow"
  description: "An example workflow demonstrating the DSL"

  inputs:
    - name: repo_url
      type: string
      required: true
      description: "Git repository URL"

    - name: environment
      type: enum
      values: [dev, staging, prod]
      default: dev

  outputs:
    - name: deployment_url
      type: string
      description: "URL of deployed application"

  steps:
    - id: clone_repo
      type: git_clone
      params:
        url: ${{ inputs.repo_url }}
      agent_role: executor
      timeout: 60s

    - id: run_tests
      type: execute_python
      depends_on: [clone_repo]
      params:
        script: pytest tests/
      agent_role: executor
      retry: 3

    - id: build_docker
      type: docker_build
      depends_on: [run_tests]
      params:
        dockerfile: Dockerfile
      agent_role: executor

    - id: deploy
      type: kubernetes_apply
      depends_on: [build_docker]
      params:
        manifest: k8s/deployment.yaml
        environment: ${{ inputs.environment }}
      agent_role: executor
      approval_required: true
"""

    # Parse workflow
    workflow = parser.parse_dict(yaml.safe_load(workflow_yaml))
    print("Workflow parsed successfully!")
    print(f"ID: {workflow.id}")
    print(f"Version: {workflow.version}")
    print(f"Steps: {len(workflow.steps)}")

    # Validate inputs
    inputs = {
        "repo_url": "https://github.com/example/repo.git",
        "environment": "dev"
    }
    validated = parser.validate_inputs(workflow, inputs)
    print(f"Inputs validated: {validated}")
