"""
Single AI Agent Code Generation System

Autonomous AI agent that can:
- Accept high-level requirements
- Decompose tasks intelligently
- Generate complete modules
- Refactor existing code
- Write tests automatically
- Validate output
- Optimize performance
"""

from .agent_planner import (
    AgentPlanner,
    TaskNode,
    ExecutionPlan,
    PlanningStrategy
)
from .code_generator import (
    CodeGenerator,
    GeneratedCode,
    CodeContext,
    LanguageSupport
)
from .task_decomposer import (
    TaskDecomposer,
    TaskBreakdown,
    ComplexityAnalyzer
)
from .validation_pipeline import (
    CodeValidator,
    ValidationResult,
    TestGenerator
)
from .context_manager import (
    ContextManager,
    CodebaseContext,
    MemoryStore
)
from .self_evaluator import (
    SelfEvaluator,
    EvaluationMetrics,
    IterationEngine
)

__all__ = [
    "AgentPlanner",
    "TaskNode",
    "ExecutionPlan",
    "PlanningStrategy",
    "CodeGenerator",
    "GeneratedCode",
    "CodeContext",
    "LanguageSupport",
    "TaskDecomposer",
    "TaskBreakdown",
    "ComplexityAnalyzer",
    "CodeValidator",
    "ValidationResult",
    "TestGenerator",
    "ContextManager",
    "CodebaseContext",
    "MemoryStore",
    "SelfEvaluator",
    "EvaluationMetrics",
    "IterationEngine"
]
