"""
Autonomous Agent System
Single powerful AI agent for autonomous code generation and task execution.
"""

from .autonomous_planner import (
    AutonomousPlanner,
    ExecutionPlan,
    RequirementAnalysis,
    TaskNode,
    TaskPriority,
    TaskStatus,
    TaskType
)
from .code_generator import (
    IntelligentCodeGenerator,
    GeneratedCode,
    CodeLanguage,
    CodeQuality,
    ValidationResult
)
from .agent_orchestrator import (
    AutonomousAgentOrchestrator,
    AgentMemory,
    HallucinationDetector,
    SafetyBoundaries
)
from .self_evaluator import (
    SelfEvaluator,
    EvaluationResult,
    QualityDimension,
    EvaluationVerdict,
)
from .context_manager import (
    ContextManager,
    ContextMessage,
    MessageRole,
    MessagePriority,
    CompressionResult,
)
from .validation_pipeline import (
    ValidationPipeline,
    PipelineResult,
    ValidationStage,
    ValidationSeverity,
    ValidationIssue,
)

__all__ = [
    # Planning
    "AutonomousPlanner",
    "ExecutionPlan",
    "RequirementAnalysis",
    "TaskNode",
    "TaskPriority",
    "TaskStatus",
    "TaskType",
    # Code Generation
    "IntelligentCodeGenerator",
    "GeneratedCode",
    "CodeLanguage",
    "CodeQuality",
    "ValidationResult",
    # Orchestration
    "AutonomousAgentOrchestrator",
    "AgentMemory",
    "HallucinationDetector",
    "SafetyBoundaries",
    # Self-Evaluation
    "SelfEvaluator",
    "EvaluationResult",
    "QualityDimension",
    "EvaluationVerdict",
    # Context Management
    "ContextManager",
    "ContextMessage",
    "MessageRole",
    "MessagePriority",
    "CompressionResult",
    # Validation Pipeline
    "ValidationPipeline",
    "PipelineResult",
    "ValidationStage",
    "ValidationSeverity",
    "ValidationIssue",
]

