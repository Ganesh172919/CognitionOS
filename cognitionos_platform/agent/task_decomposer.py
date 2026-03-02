"""
Task Decomposer - DAG from plan steps with parallelization hints.

Builds executable subtask DAG from planner output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class StepType(str, Enum):
    PLANNING = "planning"
    CODEGEN = "codegen"
    VALIDATION = "validation"
    EVALUATION = "evaluation"


# Canonical dependency order: earlier steps must complete before later
_STEP_ORDER: Dict[str, int] = {
    "plan": 0,
    "planning": 0,
    "codegen": 1,
    "validate": 2,
    "validation": 2,
    "evaluate": 3,
    "evaluation": 3,
}


@dataclass
class Subtask:
    """Executable subtask from plan step."""

    id: str
    step_type: str
    title: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    parallelizable_with: List[str] = field(default_factory=list)
    order: int = 0


@dataclass
class TaskDAG:
    """DAG of subtasks with topological order and parallel hints."""

    subtasks: List[Subtask]
    topological_order: List[str]
    parallel_groups: List[List[str]]


def decompose_plan(plan: Dict[str, Any]) -> TaskDAG:
    """
    Build DAG from plan steps.
    Returns subtasks in topological order with parallelization groups.
    """
    steps = plan.get("steps") or []
    subtasks: List[Subtask] = []
    step_ids: List[str] = []

    for i, s in enumerate(steps):
        step_id = str(s.get("id", f"step_{i}")).lower()
        step_type = str(s.get("type", "unknown")).lower()
        order = _STEP_ORDER.get(step_id, _STEP_ORDER.get(step_type, i))
        deps = []
        for j in range(i):
            prev_id = str(steps[j].get("id", f"step_{j}")).lower()
            deps.append(prev_id)
        subtask = Subtask(
            id=step_id,
            step_type=step_type,
            title=str(s.get("title", step_id)),
            description=str(s.get("description", "")),
            dependencies=deps,
            order=order,
        )
        subtasks.append(subtask)
        step_ids.append(step_id)

    topological_order = step_ids

    parallel_groups: List[List[str]] = []
    seen: Set[str] = set()
    for st in subtasks:
        if st.id not in seen:
            group = [s.id for s in subtasks if s.order == st.order]
            if group:
                parallel_groups.append(group)
                seen.update(group)

    return TaskDAG(
        subtasks=subtasks,
        topological_order=topological_order,
        parallel_groups=parallel_groups,
    )


def get_next_executable(
    dag: TaskDAG,
    completed: Set[str],
) -> List[str]:
    """
    Get IDs of subtasks that can be executed next (all deps satisfied).
    """
    executable = []
    for st in dag.subtasks:
        if st.id in completed:
            continue
        if all(d in completed for d in st.dependencies):
            executable.append(st.id)
    return executable
