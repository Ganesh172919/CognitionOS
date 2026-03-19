"""Enterprise Workflow Automation Engine package exports."""

from .workflow_engine import (
    ExecutionStatus,
    StepExecution,
    StepType,
    TriggerType,
    WorkflowDefinition,
    WorkflowEngine,
    WorkflowExecution,
    WorkflowStatus,
    WorkflowStep,
)

# Aliases for backward compatibility and external API expectations
WorkflowAutomationEngine = WorkflowEngine
WorkflowVersionManager = WorkflowEngine
StepExecutorRegistry = WorkflowEngine


from enum import Enum


class RetryStrategy(str, Enum):
    """Retry strategy for workflow steps."""

    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


WorkflowTrigger = WorkflowStep

__all__ = [
    "WorkflowAutomationEngine",
    "WorkflowDefinition",
    "WorkflowStep",
    "WorkflowTrigger",
    "WorkflowExecution",
    "StepExecution",
    "StepExecutorRegistry",
    "WorkflowVersionManager",
    "WorkflowStatus",
    "StepType",
    "TriggerType",
    "ExecutionStatus",
    "RetryStrategy",
]
