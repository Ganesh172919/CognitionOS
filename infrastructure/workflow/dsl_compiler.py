"""
Workflow DSL Compiler

Parses declarative YAML/JSON workflow definitions and compiles them into
executable execution plans.

DSL Format::

    name: "Data Processing Pipeline"
    version: "1.0"
    description: "Ingests, transforms and loads data"
    variables:
      source_bucket: "my-bucket"
      batch_size: 100
    steps:
      - id: ingest
        type: tool_call
        tool: "s3_reader"
        inputs:
          bucket: "{{source_bucket}}"
          limit: "{{batch_size}}"
        retry:
          max_attempts: 3
          backoff: exponential
      - id: transform
        type: llm_call
        model: "gpt-4o-mini"
        prompt: "Transform this data: {{ingest.output}}"
        depends_on: [ingest]
      - id: load
        type: tool_call
        tool: "db_writer"
        inputs:
          data: "{{transform.output}}"
        depends_on: [transform]
        on_failure: skip
    triggers:
      - type: schedule
        cron: "0 * * * *"
      - type: webhook
        path: "/trigger/data-pipeline"

Features:
- Full YAML/JSON parsing with schema validation
- Variable interpolation with {{var}} syntax
- Step dependency graph construction
- Cycle detection
- Conditional branching (if/else)
- Parallel step groups
- Retry policy parsing
- Trigger definitions (schedule, webhook, event)
- Compilation to execution-ready plan
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class StepType(str, Enum):
    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    PARALLEL = "parallel"
    CONDITION = "condition"
    LOOP = "loop"
    WAIT = "wait"
    WEBHOOK = "webhook"
    TRANSFORM = "transform"


class OnFailure(str, Enum):
    FAIL = "fail"          # Stop the workflow
    SKIP = "skip"          # Skip this step, continue
    RETRY = "retry"        # Retry with policy
    FALLBACK = "fallback"  # Use fallback step


class TriggerType(str, Enum):
    SCHEDULE = "schedule"
    WEBHOOK = "webhook"
    EVENT = "event"
    MANUAL = "manual"


class BackoffStrategy(str, Enum):
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    jitter: bool = True

    def compute_delay(self, attempt: int) -> float:
        """Compute delay for attempt N (0-indexed)"""
        if self.backoff == BackoffStrategy.FIXED:
            delay = self.initial_delay_seconds
        elif self.backoff == BackoffStrategy.LINEAR:
            delay = self.initial_delay_seconds * (attempt + 1)
        else:  # EXPONENTIAL
            delay = self.initial_delay_seconds * (2 ** attempt)
        return min(delay, self.max_delay_seconds)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetryPolicy":
        return cls(
            max_attempts=int(data.get("max_attempts", 3)),
            backoff=BackoffStrategy(data.get("backoff", "exponential")),
            initial_delay_seconds=float(data.get("initial_delay_seconds", 1.0)),
            max_delay_seconds=float(data.get("max_delay_seconds", 60.0)),
            jitter=bool(data.get("jitter", True)),
        )


@dataclass
class StepCondition:
    """Conditional execution guard"""
    expression: str    # e.g. "{{step_a.status}} == 'success'"
    target_if_true: Optional[str] = None   # Step ID to jump to
    target_if_false: Optional[str] = None  # Step ID to jump to

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepCondition":
        return cls(
            expression=data["expression"],
            target_if_true=data.get("if_true"),
            target_if_false=data.get("if_false"),
        )


@dataclass
class WorkflowStep:
    """A compiled workflow step"""
    step_id: str
    step_type: StepType
    name: str = ""
    description: str = ""

    # Execution params
    tool: Optional[str] = None        # For TOOL_CALL
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)  # output_var -> step_id.output

    # For LLM_CALL
    model: Optional[str] = None
    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024

    # For PARALLEL
    parallel_steps: List[str] = field(default_factory=list)

    # For CONDITION
    condition: Optional[StepCondition] = None

    # For LOOP
    loop_over: Optional[str] = None   # Variable to iterate
    loop_body: Optional[str] = None   # Step ID to execute per iteration

    # Control flow
    depends_on: List[str] = field(default_factory=list)
    on_failure: OnFailure = OnFailure.FAIL
    fallback_step: Optional[str] = None
    timeout_seconds: Optional[float] = None
    retry: Optional[RetryPolicy] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "type": self.step_type.value,
            "name": self.name,
            "depends_on": self.depends_on,
            "on_failure": self.on_failure.value,
        }


@dataclass
class WorkflowTrigger:
    """A workflow trigger definition"""
    trigger_type: TriggerType
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowTrigger":
        return cls(
            trigger_type=TriggerType(data.get("type", "manual")),
            config={k: v for k, v in data.items() if k != "type"},
        )


@dataclass
class CompiledWorkflow:
    """The result of compiling a workflow DSL definition"""
    workflow_id: str
    name: str
    version: str
    description: str
    variables: Dict[str, Any]
    steps: List[WorkflowStep]
    triggers: List[WorkflowTrigger]
    execution_order: List[str]          # Topologically sorted step IDs
    parallel_groups: List[List[str]]    # Groups of steps that can run in parallel
    metadata: Dict[str, Any]
    compiled_at: float = field(default_factory=time.time)

    @property
    def step_map(self) -> Dict[str, WorkflowStep]:
        return {s.step_id: s for s in self.steps}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "version": self.version,
            "step_count": len(self.steps),
            "execution_order": self.execution_order,
            "trigger_count": len(self.triggers),
            "compiled_at": self.compiled_at,
        }


@dataclass
class ValidationError:
    field: str
    message: str
    severity: str = "error"  # "error" or "warning"


class WorkflowDSLCompiler:
    """
    Compiles YAML/JSON workflow definitions into executable CompiledWorkflow objects.

    Usage::

        compiler = WorkflowDSLCompiler()
        result = compiler.compile(yaml_string)
        if result.errors:
            raise ValueError(result.errors)
        plan = result.workflow
    """

    VARIABLE_PATTERN = re.compile(r"\{\{([^}]+)\}\}")

    def compile_dict(self, definition: Dict[str, Any]) -> "CompilationResult":
        """Compile a pre-parsed workflow definition dict"""
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []

        # Extract top-level fields
        workflow_id = str(definition.get("id", definition.get("name", "unnamed"))).lower().replace(" ", "_")
        name = definition.get("name", "Unnamed Workflow")
        version = str(definition.get("version", "1.0"))
        description = definition.get("description", "")
        variables = definition.get("variables", {})
        metadata = definition.get("metadata", {})

        if not definition.get("steps"):
            errors.append(ValidationError("steps", "Workflow must define at least one step"))
            return CompilationResult(errors=errors, warnings=warnings)

        # Parse steps
        steps: List[WorkflowStep] = []
        for i, raw_step in enumerate(definition["steps"]):
            step, step_errors = self._parse_step(raw_step, i)
            errors.extend(step_errors)
            if step:
                steps.append(step)

        # Parse triggers
        triggers = [
            WorkflowTrigger.from_dict(t)
            for t in definition.get("triggers", [])
        ]

        if errors:
            return CompilationResult(errors=errors, warnings=warnings)

        # Build dependency graph and validate
        graph_errors, execution_order, parallel_groups = self._build_execution_graph(steps)
        errors.extend(graph_errors)

        if errors:
            return CompilationResult(errors=errors, warnings=warnings)

        # Validate variable references
        var_warnings = self._validate_variable_references(steps, variables)
        warnings.extend(var_warnings)

        compiled = CompiledWorkflow(
            workflow_id=workflow_id,
            name=name,
            version=version,
            description=description,
            variables=variables,
            steps=steps,
            triggers=triggers,
            execution_order=execution_order,
            parallel_groups=parallel_groups,
            metadata=metadata,
        )
        return CompilationResult(workflow=compiled, errors=errors, warnings=warnings)

    def compile_json(self, json_string: str) -> "CompilationResult":
        """Compile from a JSON string"""
        try:
            definition = json.loads(json_string)
        except json.JSONDecodeError as exc:
            return CompilationResult(errors=[
                ValidationError("json", f"Invalid JSON: {exc}")
            ])
        return self.compile_dict(definition)

    def compile_yaml(self, yaml_string: str) -> "CompilationResult":
        """Compile from a YAML string (requires PyYAML)"""
        try:
            import yaml  # type: ignore[import-untyped]
            definition = yaml.safe_load(yaml_string)
        except Exception as exc:  # noqa: BLE001
            return CompilationResult(errors=[
                ValidationError("yaml", f"Invalid YAML: {exc}")
            ])
        return self.compile_dict(definition)

    # ──────────────────────────────────────────────
    # Parsing
    # ──────────────────────────────────────────────

    def _parse_step(
        self,
        raw: Dict[str, Any],
        index: int,
    ) -> Tuple[Optional[WorkflowStep], List[ValidationError]]:
        errors: List[ValidationError] = []
        step_id = raw.get("id")
        if not step_id:
            errors.append(ValidationError(f"steps[{index}].id", "Step must have an 'id' field"))
            return None, errors

        raw_type = raw.get("type", "tool_call")
        try:
            step_type = StepType(raw_type)
        except ValueError:
            errors.append(ValidationError(
                f"steps[{step_id}].type",
                f"Unknown step type '{raw_type}'. Valid types: {[t.value for t in StepType]}",
            ))
            return None, errors

        retry_policy = None
        if "retry" in raw:
            retry_policy = RetryPolicy.from_dict(raw["retry"])

        condition = None
        if "condition" in raw:
            condition = StepCondition.from_dict(raw["condition"])

        on_failure = OnFailure.FAIL
        if "on_failure" in raw:
            try:
                on_failure = OnFailure(raw["on_failure"])
            except ValueError:
                errors.append(ValidationError(
                    f"steps[{step_id}].on_failure",
                    f"Unknown on_failure value '{raw['on_failure']}'",
                ))

        step = WorkflowStep(
            step_id=step_id,
            step_type=step_type,
            name=raw.get("name", step_id),
            description=raw.get("description", ""),
            tool=raw.get("tool"),
            inputs=raw.get("inputs", {}),
            outputs=raw.get("outputs", {}),
            model=raw.get("model"),
            prompt=raw.get("prompt"),
            system_prompt=raw.get("system_prompt"),
            temperature=float(raw.get("temperature", 0.7)),
            max_tokens=int(raw.get("max_tokens", 1024)),
            parallel_steps=raw.get("parallel_steps", []),
            condition=condition,
            loop_over=raw.get("loop_over"),
            loop_body=raw.get("loop_body"),
            depends_on=raw.get("depends_on", []),
            on_failure=on_failure,
            fallback_step=raw.get("fallback_step"),
            timeout_seconds=float(raw["timeout_seconds"]) if "timeout_seconds" in raw else None,
            retry=retry_policy,
            tags=raw.get("tags", []),
            metadata=raw.get("metadata", {}),
        )

        # Type-specific validation
        if step_type == StepType.TOOL_CALL and not step.tool:
            errors.append(ValidationError(f"steps[{step_id}].tool", "tool_call step must specify 'tool'"))
        if step_type == StepType.LLM_CALL and not step.prompt:
            errors.append(ValidationError(f"steps[{step_id}].prompt", "llm_call step must specify 'prompt'"))

        return step, errors

    # ──────────────────────────────────────────────
    # Graph analysis
    # ──────────────────────────────────────────────

    def _build_execution_graph(
        self,
        steps: List[WorkflowStep],
    ) -> Tuple[List[ValidationError], List[str], List[List[str]]]:
        """
        Build and validate the execution dependency graph.
        Returns: (errors, topological_order, parallel_groups)
        """
        errors: List[ValidationError] = []
        step_ids = {s.step_id for s in steps}

        # Validate all depends_on references exist
        for step in steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(ValidationError(
                        f"steps[{step.step_id}].depends_on",
                        f"Dependency '{dep}' references unknown step ID",
                    ))

        if errors:
            return errors, [], []

        # Topological sort (Kahn's algorithm)
        in_degree: Dict[str, int] = {s.step_id: 0 for s in steps}
        adjacency: Dict[str, List[str]] = {s.step_id: [] for s in steps}

        for step in steps:
            for dep in step.depends_on:
                adjacency[dep].append(step.step_id)
                in_degree[step.step_id] += 1

        # Kahn's BFS
        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        order: List[str] = []
        levels: List[List[str]] = []

        while queue:
            levels.append(list(queue))
            order.extend(queue)
            next_queue: List[str] = []
            for node in queue:
                for neighbor in adjacency[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            queue = next_queue

        if len(order) != len(steps):
            errors.append(ValidationError(
                "steps",
                "Workflow contains a dependency cycle",
            ))
            return errors, [], []

        # Build parallel groups (steps in the same level can run in parallel)
        parallel_groups = [group for group in levels if len(group) > 1]

        return errors, order, parallel_groups

    def _validate_variable_references(
        self,
        steps: List[WorkflowStep],
        variables: Dict[str, Any],
    ) -> List[ValidationError]:
        """Warn about variable references that might not resolve at runtime"""
        warnings: List[ValidationError] = []
        step_ids = {s.step_id for s in steps}

        def check_string(value: str, location: str) -> None:
            for match in self.VARIABLE_PATTERN.finditer(value):
                ref = match.group(1).strip()
                # Accept: plain variable, step.output, step.status
                if "." in ref:
                    step_ref, _ = ref.split(".", 1)
                    if step_ref not in step_ids and step_ref not in variables:
                        warnings.append(ValidationError(
                            location,
                            f"Variable '{{{{{ref}}}}}' references unknown step or variable '{step_ref}'",
                            severity="warning",
                        ))
                elif ref not in variables:
                    warnings.append(ValidationError(
                        location,
                        f"Variable '{{{{{ref}}}}}' not defined in workflow variables",
                        severity="warning",
                    ))

        for step in steps:
            if step.prompt:
                check_string(step.prompt, f"steps[{step.step_id}].prompt")
            for key, val in step.inputs.items():
                if isinstance(val, str):
                    check_string(val, f"steps[{step.step_id}].inputs.{key}")

        return warnings


@dataclass
class CompilationResult:
    """Result of compiling a workflow DSL"""
    workflow: Optional[CompiledWorkflow] = None
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.errors and self.workflow is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "workflow": self.workflow.to_dict() if self.workflow else None,
            "errors": [{"field": e.field, "message": e.message} for e in self.errors],
            "warnings": [{"field": w.field, "message": w.message} for w in self.warnings],
        }


# Module-level compiler instance
_compiler: Optional[WorkflowDSLCompiler] = None


def get_compiler() -> WorkflowDSLCompiler:
    global _compiler
    if _compiler is None:
        _compiler = WorkflowDSLCompiler()
    return _compiler
