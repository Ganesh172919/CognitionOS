"""
Single AI Agent Code Generation System
Autonomous agent that accepts high-level requirements and generates complete modules
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import json
import asyncio
from datetime import datetime
import hashlib


class TaskType(Enum):
    """Types of tasks the agent can handle"""
    MODULE_GENERATION = "module_generation"
    CODE_REFACTORING = "code_refactoring"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    TESTING = "testing"
    ARCHITECTURE_REVIEW = "architecture_review"


class TaskPriority(Enum):
    """Task execution priority"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    BACKGROUND = 1


class ExecutionStatus(Enum):
    """Status of task execution"""
    PENDING = "pending"
    PLANNING = "planning"
    DECOMPOSING = "decomposing"
    EXECUTING = "executing"
    VALIDATING = "validating"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class TaskNode:
    """Represents a decomposed task node"""
    task_id: str
    task_type: TaskType
    description: str
    priority: TaskPriority
    dependencies: List[str] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    status: ExecutionStatus = ExecutionStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    validation_score: float = 0.0

    def mark_completed(self, outputs: Dict[str, Any], validation_score: float = 1.0):
        """Mark task as completed"""
        self.status = ExecutionStatus.COMPLETED
        self.outputs = outputs
        self.validation_score = validation_score
        self.completed_at = datetime.utcnow()

    def mark_failed(self, error: str):
        """Mark task as failed"""
        self.status = ExecutionStatus.FAILED
        self.error_message = error
        self.retry_count += 1

    def can_retry(self) -> bool:
        """Check if task can be retried"""
        return self.retry_count < self.max_retries


@dataclass
class CodeArtifact:
    """Generated code artifact"""
    artifact_id: str
    file_path: str
    content: str
    language: str
    artifact_type: str  # module, class, function, test
    dependencies: List[str] = field(default_factory=list)
    documentation: str = ""
    tests_coverage: float = 0.0
    complexity_score: float = 0.0
    quality_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def calculate_hash(self) -> str:
        """Calculate content hash for versioning"""
        return hashlib.sha256(self.content.encode()).hexdigest()


@dataclass
class ValidationResult:
    """Result of code validation"""
    is_valid: bool
    syntax_valid: bool
    style_valid: bool
    security_valid: bool
    performance_score: float
    maintainability_score: float
    issues: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def overall_score(self) -> float:
        """Calculate overall validation score"""
        return (
            (1.0 if self.syntax_valid else 0.0) * 0.3 +
            (1.0 if self.style_valid else 0.0) * 0.2 +
            (1.0 if self.security_valid else 0.0) * 0.3 +
            self.performance_score * 0.1 +
            self.maintainability_score * 0.1
        )


@dataclass
class AgentMemory:
    """Persistent memory for the agent"""
    context: Dict[str, Any] = field(default_factory=dict)
    learned_patterns: List[Dict[str, Any]] = field(default_factory=list)
    architecture_decisions: List[Dict[str, Any]] = field(default_factory=list)
    common_mistakes: List[Dict[str, Any]] = field(default_factory=list)
    best_practices: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    def add_pattern(self, pattern: Dict[str, Any]):
        """Add learned pattern"""
        self.learned_patterns.append({
            **pattern,
            "learned_at": datetime.utcnow().isoformat()
        })

    def add_architecture_decision(self, decision: Dict[str, Any]):
        """Record architecture decision"""
        self.architecture_decisions.append({
            **decision,
            "decided_at": datetime.utcnow().isoformat()
        })

    def update_metric(self, metric_name: str, value: float):
        """Update performance metric"""
        self.performance_metrics[metric_name] = value


class TaskDecompositionEngine:
    """Intelligent task decomposition engine"""

    def __init__(self, memory: AgentMemory):
        self.memory = memory
        self.decomposition_strategies = {
            TaskType.MODULE_GENERATION: self._decompose_module_generation,
            TaskType.CODE_REFACTORING: self._decompose_refactoring,
            TaskType.VALIDATION: self._decompose_validation,
            TaskType.OPTIMIZATION: self._decompose_optimization,
            TaskType.TESTING: self._decompose_testing,
        }

    async def decompose_task(
        self,
        requirement: str,
        task_type: TaskType,
        context: Dict[str, Any]
    ) -> List[TaskNode]:
        """Decompose high-level requirement into subtasks"""
        strategy = self.decomposition_strategies.get(task_type)
        if not strategy:
            raise ValueError(f"No decomposition strategy for {task_type}")

        # Analyze requirement complexity
        complexity = self._analyze_complexity(requirement, context)

        # Apply decomposition strategy
        subtasks = await strategy(requirement, context, complexity)

        # Optimize task order based on dependencies
        optimized_tasks = self._optimize_task_order(subtasks)

        return optimized_tasks

    def _analyze_complexity(
        self,
        requirement: str,
        context: Dict[str, Any]
    ) -> float:
        """Analyze requirement complexity"""
        # Simple heuristic - can be enhanced with ML
        factors = {
            "length": len(requirement) / 1000.0,
            "keywords": len([w for w in requirement.split() if len(w) > 5]) / 50.0,
            "dependencies": len(context.get("dependencies", [])) / 10.0,
            "files_affected": len(context.get("files", [])) / 20.0,
        }
        return min(sum(factors.values()) / len(factors), 1.0)

    async def _decompose_module_generation(
        self,
        requirement: str,
        context: Dict[str, Any],
        complexity: float
    ) -> List[TaskNode]:
        """Decompose module generation task"""
        tasks = []
        base_id = hashlib.md5(requirement.encode()).hexdigest()[:8]

        # Task 1: Analyze requirements and design
        tasks.append(TaskNode(
            task_id=f"{base_id}_design",
            task_type=TaskType.MODULE_GENERATION,
            description="Analyze requirements and design module architecture",
            priority=TaskPriority.CRITICAL,
            inputs={"requirement": requirement, "context": context}
        ))

        # Task 2: Generate core module code
        tasks.append(TaskNode(
            task_id=f"{base_id}_core",
            task_type=TaskType.MODULE_GENERATION,
            description="Generate core module implementation",
            priority=TaskPriority.HIGH,
            dependencies=[f"{base_id}_design"],
            inputs={"design": "from_previous"}
        ))

        # Task 3: Generate supporting utilities
        tasks.append(TaskNode(
            task_id=f"{base_id}_utils",
            task_type=TaskType.MODULE_GENERATION,
            description="Generate utility functions and helpers",
            priority=TaskPriority.MEDIUM,
            dependencies=[f"{base_id}_design"],
            inputs={"design": "from_previous"}
        ))

        # Task 4: Generate tests
        tasks.append(TaskNode(
            task_id=f"{base_id}_tests",
            task_type=TaskType.TESTING,
            description="Generate comprehensive test suite",
            priority=TaskPriority.HIGH,
            dependencies=[f"{base_id}_core", f"{base_id}_utils"],
            inputs={"code": "from_previous"}
        ))

        # Task 5: Validate and optimize
        tasks.append(TaskNode(
            task_id=f"{base_id}_validate",
            task_type=TaskType.VALIDATION,
            description="Validate code quality and optimize",
            priority=TaskPriority.CRITICAL,
            dependencies=[f"{base_id}_tests"],
            inputs={"code": "from_previous", "tests": "from_previous"}
        ))

        return tasks

    async def _decompose_refactoring(
        self,
        requirement: str,
        context: Dict[str, Any],
        complexity: float
    ) -> List[TaskNode]:
        """Decompose refactoring task"""
        tasks = []
        base_id = hashlib.md5(requirement.encode()).hexdigest()[:8]

        # Analyze existing code
        tasks.append(TaskNode(
            task_id=f"{base_id}_analyze",
            task_type=TaskType.CODE_REFACTORING,
            description="Analyze existing code structure",
            priority=TaskPriority.HIGH,
            inputs={"files": context.get("files", [])}
        ))

        # Plan refactoring
        tasks.append(TaskNode(
            task_id=f"{base_id}_plan",
            task_type=TaskType.CODE_REFACTORING,
            description="Create refactoring plan",
            priority=TaskPriority.HIGH,
            dependencies=[f"{base_id}_analyze"]
        ))

        # Execute refactoring
        tasks.append(TaskNode(
            task_id=f"{base_id}_execute",
            task_type=TaskType.CODE_REFACTORING,
            description="Execute refactoring changes",
            priority=TaskPriority.CRITICAL,
            dependencies=[f"{base_id}_plan"]
        ))

        # Validate refactored code
        tasks.append(TaskNode(
            task_id=f"{base_id}_validate",
            task_type=TaskType.VALIDATION,
            description="Validate refactored code",
            priority=TaskPriority.CRITICAL,
            dependencies=[f"{base_id}_execute"]
        ))

        return tasks

    async def _decompose_validation(
        self,
        requirement: str,
        context: Dict[str, Any],
        complexity: float
    ) -> List[TaskNode]:
        """Decompose validation task"""
        # Single comprehensive validation task
        base_id = hashlib.md5(requirement.encode()).hexdigest()[:8]
        return [TaskNode(
            task_id=f"{base_id}_validate",
            task_type=TaskType.VALIDATION,
            description="Comprehensive code validation",
            priority=TaskPriority.CRITICAL,
            inputs=context
        )]

    async def _decompose_optimization(
        self,
        requirement: str,
        context: Dict[str, Any],
        complexity: float
    ) -> List[TaskNode]:
        """Decompose optimization task"""
        tasks = []
        base_id = hashlib.md5(requirement.encode()).hexdigest()[:8]

        # Profile performance
        tasks.append(TaskNode(
            task_id=f"{base_id}_profile",
            task_type=TaskType.OPTIMIZATION,
            description="Profile code performance",
            priority=TaskPriority.HIGH,
            inputs=context
        ))

        # Apply optimizations
        tasks.append(TaskNode(
            task_id=f"{base_id}_optimize",
            task_type=TaskType.OPTIMIZATION,
            description="Apply performance optimizations",
            priority=TaskPriority.CRITICAL,
            dependencies=[f"{base_id}_profile"]
        ))

        # Validate improvements
        tasks.append(TaskNode(
            task_id=f"{base_id}_validate",
            task_type=TaskType.VALIDATION,
            description="Validate performance improvements",
            priority=TaskPriority.HIGH,
            dependencies=[f"{base_id}_optimize"]
        ))

        return tasks

    async def _decompose_testing(
        self,
        requirement: str,
        context: Dict[str, Any],
        complexity: float
    ) -> List[TaskNode]:
        """Decompose testing task"""
        tasks = []
        base_id = hashlib.md5(requirement.encode()).hexdigest()[:8]

        # Generate unit tests
        tasks.append(TaskNode(
            task_id=f"{base_id}_unit",
            task_type=TaskType.TESTING,
            description="Generate unit tests",
            priority=TaskPriority.HIGH,
            inputs=context
        ))

        # Generate integration tests
        tasks.append(TaskNode(
            task_id=f"{base_id}_integration",
            task_type=TaskType.TESTING,
            description="Generate integration tests",
            priority=TaskPriority.MEDIUM,
            inputs=context
        ))

        # Generate load tests
        tasks.append(TaskNode(
            task_id=f"{base_id}_load",
            task_type=TaskType.TESTING,
            description="Generate load tests",
            priority=TaskPriority.LOW,
            inputs=context
        ))

        return tasks

    def _optimize_task_order(self, tasks: List[TaskNode]) -> List[TaskNode]:
        """Optimize task execution order using topological sort"""
        # Build dependency graph
        task_map = {task.task_id: task for task in tasks}
        in_degree = {task.task_id: 0 for task in tasks}

        for task in tasks:
            for dep in task.dependencies:
                if dep in in_degree:
                    in_degree[task.task_id] += 1

        # Topological sort with priority
        result = []
        queue = [(task.priority.value, task) for task in tasks if in_degree[task.task_id] == 0]
        queue.sort(reverse=True)

        while queue:
            _, task = queue.pop(0)
            result.append(task)

            # Update dependencies
            for other_task in tasks:
                if task.task_id in other_task.dependencies:
                    in_degree[other_task.task_id] -= 1
                    if in_degree[other_task.task_id] == 0:
                        queue.append((other_task.priority.value, other_task))
                        queue.sort(reverse=True)

        return result


class CodeValidator:
    """Advanced code validation system"""

    def __init__(self, memory: AgentMemory):
        self.memory = memory

    async def validate_code(
        self,
        artifact: CodeArtifact,
        context: Dict[str, Any]
    ) -> ValidationResult:
        """Comprehensive code validation"""
        # Run parallel validation checks
        syntax_result = await self._validate_syntax(artifact)
        style_result = await self._validate_style(artifact)
        security_result = await self._validate_security(artifact)
        performance_score = await self._analyze_performance(artifact)
        maintainability_score = await self._analyze_maintainability(artifact)

        issues = []
        suggestions = []

        if not syntax_result["valid"]:
            issues.extend(syntax_result["issues"])

        if not style_result["valid"]:
            issues.extend(style_result["issues"])
            suggestions.extend(style_result["suggestions"])

        if not security_result["valid"]:
            issues.extend(security_result["issues"])
            suggestions.extend(security_result["suggestions"])

        return ValidationResult(
            is_valid=syntax_result["valid"] and security_result["valid"],
            syntax_valid=syntax_result["valid"],
            style_valid=style_result["valid"],
            security_valid=security_result["valid"],
            performance_score=performance_score,
            maintainability_score=maintainability_score,
            issues=issues,
            suggestions=suggestions
        )

    async def _validate_syntax(self, artifact: CodeArtifact) -> Dict[str, Any]:
        """Validate code syntax"""
        issues = []

        try:
            if artifact.language == "python":
                import ast
                ast.parse(artifact.content)
            # Add support for other languages

            return {"valid": True, "issues": []}
        except SyntaxError as e:
            issues.append({
                "type": "syntax_error",
                "line": e.lineno,
                "message": str(e),
                "severity": "critical"
            })
            return {"valid": False, "issues": issues}

    async def _validate_style(self, artifact: CodeArtifact) -> Dict[str, Any]:
        """Validate code style"""
        issues = []
        suggestions = []

        # Check line length
        lines = artifact.content.split('\n')
        for i, line in enumerate(lines, 1):
            if len(line) > 100:
                issues.append({
                    "type": "style",
                    "line": i,
                    "message": "Line exceeds 100 characters",
                    "severity": "warning"
                })

        # Check for common style issues
        if "  " in artifact.content:  # Multiple spaces
            suggestions.append("Consider using consistent indentation")

        return {
            "valid": len([i for i in issues if i["severity"] == "critical"]) == 0,
            "issues": issues,
            "suggestions": suggestions
        }

    async def _validate_security(self, artifact: CodeArtifact) -> Dict[str, Any]:
        """Validate code security"""
        issues = []
        suggestions = []

        dangerous_patterns = [
            ("eval(", "Avoid using eval()"),
            ("exec(", "Avoid using exec()"),
            ("__import__", "Avoid dynamic imports"),
            ("pickle.loads", "Pickle can execute arbitrary code"),
            ("yaml.load", "Use yaml.safe_load instead"),
        ]

        for pattern, message in dangerous_patterns:
            if pattern in artifact.content:
                issues.append({
                    "type": "security",
                    "message": message,
                    "severity": "critical"
                })
                suggestions.append(f"Replace {pattern} with safer alternative")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions
        }

    async def _analyze_performance(self, artifact: CodeArtifact) -> float:
        """Analyze code performance"""
        # Simple heuristics - can be enhanced with profiling
        score = 1.0

        # Check for inefficient patterns
        if "for" in artifact.content and "append" in artifact.content:
            score -= 0.1  # List comprehension might be better

        if artifact.content.count("for") > 3:  # Nested loops
            score -= 0.2

        return max(score, 0.0)

    async def _analyze_maintainability(self, artifact: CodeArtifact) -> float:
        """Analyze code maintainability"""
        score = 1.0
        lines = artifact.content.split('\n')

        # Check function length
        avg_function_length = len(lines) / max(artifact.content.count('def '), 1)
        if avg_function_length > 50:
            score -= 0.3

        # Check documentation
        if '"""' not in artifact.content and "'''" not in artifact.content:
            score -= 0.2

        # Check complexity
        complexity_indicators = artifact.content.count('if') + artifact.content.count('for') + artifact.content.count('while')
        if complexity_indicators > 10:
            score -= 0.2

        return max(score, 0.0)


class CodeOptimizer:
    """Intelligent code optimizer"""

    def __init__(self, memory: AgentMemory):
        self.memory = memory

    async def optimize_code(
        self,
        artifact: CodeArtifact,
        optimization_goals: List[str]
    ) -> CodeArtifact:
        """Optimize code based on goals"""
        optimized_content = artifact.content

        for goal in optimization_goals:
            if goal == "performance":
                optimized_content = await self._optimize_performance(optimized_content)
            elif goal == "memory":
                optimized_content = await self._optimize_memory(optimized_content)
            elif goal == "readability":
                optimized_content = await self._optimize_readability(optimized_content)

        # Create optimized artifact
        optimized_artifact = CodeArtifact(
            artifact_id=f"{artifact.artifact_id}_optimized",
            file_path=artifact.file_path,
            content=optimized_content,
            language=artifact.language,
            artifact_type=artifact.artifact_type,
            dependencies=artifact.dependencies,
            documentation=artifact.documentation
        )

        return optimized_artifact

    async def _optimize_performance(self, content: str) -> str:
        """Apply performance optimizations"""
        # Replace inefficient patterns
        optimized = content

        # Example: Convert simple loops to list comprehensions
        # This is simplified - real implementation would use AST
        return optimized

    async def _optimize_memory(self, content: str) -> str:
        """Apply memory optimizations"""
        # Use generators instead of lists where appropriate
        return content

    async def _optimize_readability(self, content: str) -> str:
        """Improve code readability"""
        # Add proper spacing, comments, etc.
        return content


class SingleAgentCodeGenerator:
    """
    Main Single AI Agent Code Generation System
    Autonomous agent that generates complete modules from high-level requirements
    """

    def __init__(self):
        self.memory = AgentMemory()
        self.decomposition_engine = TaskDecompositionEngine(self.memory)
        self.validator = CodeValidator(self.memory)
        self.optimizer = CodeOptimizer(self.memory)
        self.task_queue: List[TaskNode] = []
        self.completed_tasks: Dict[str, TaskNode] = {}
        self.generated_artifacts: Dict[str, CodeArtifact] = {}

    async def generate_module(
        self,
        requirement: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point: Generate complete module from high-level requirement
        """
        context = context or {}

        # Step 1: Decompose into subtasks
        print(f"Decomposing requirement: {requirement}")
        tasks = await self.decomposition_engine.decompose_task(
            requirement=requirement,
            task_type=TaskType.MODULE_GENERATION,
            context=context
        )

        self.task_queue.extend(tasks)
        print(f"Created {len(tasks)} subtasks")

        # Step 2: Execute tasks in order
        execution_results = await self._execute_task_queue()

        # Step 3: Validate all generated artifacts
        validation_results = await self._validate_all_artifacts()

        # Step 4: Optimize if needed
        if any(not result.is_valid for result in validation_results):
            print("Validation failed, applying optimizations...")
            await self._optimize_failed_artifacts(validation_results)

        # Step 5: Self-evaluate and iterate if needed
        should_iterate = await self._self_evaluate()

        if should_iterate:
            print("Self-evaluation suggests improvements, iterating...")
            return await self.generate_module(requirement, context)

        # Return results
        return {
            "status": "success",
            "tasks_completed": len(self.completed_tasks),
            "artifacts_generated": len(self.generated_artifacts),
            "artifacts": [
                {
                    "file_path": artifact.file_path,
                    "language": artifact.language,
                    "type": artifact.artifact_type,
                    "quality_score": artifact.quality_score,
                    "content": artifact.content
                }
                for artifact in self.generated_artifacts.values()
            ],
            "validation_results": [
                {
                    "artifact_id": k,
                    "is_valid": v.is_valid,
                    "overall_score": v.overall_score()
                }
                for k, v in zip(self.generated_artifacts.keys(), validation_results)
            ]
        }

    async def _execute_task_queue(self) -> List[Dict[str, Any]]:
        """Execute all tasks in the queue"""
        results = []

        while self.task_queue:
            task = self.task_queue.pop(0)

            # Check if dependencies are met
            if not all(dep in self.completed_tasks for dep in task.dependencies):
                self.task_queue.append(task)  # Re-queue
                continue

            print(f"Executing task: {task.description}")
            task.status = ExecutionStatus.EXECUTING

            try:
                result = await self._execute_task(task)
                task.mark_completed(result, validation_score=0.9)
                self.completed_tasks[task.task_id] = task
                results.append(result)
                print(f"Task completed: {task.task_id}")
            except Exception as e:
                print(f"Task failed: {task.task_id} - {str(e)}")
                task.mark_failed(str(e))

                if task.can_retry():
                    task.status = ExecutionStatus.RETRYING
                    self.task_queue.append(task)
                else:
                    results.append({"error": str(e)})

        return results

    async def _execute_task(self, task: TaskNode) -> Dict[str, Any]:
        """Execute individual task"""
        # Simulate task execution - in real implementation, this would call LLM
        await asyncio.sleep(0.1)  # Simulate processing

        if task.task_type == TaskType.MODULE_GENERATION:
            return await self._generate_code(task)
        elif task.task_type == TaskType.VALIDATION:
            return await self._validate_task(task)
        elif task.task_type == TaskType.TESTING:
            return await self._generate_tests(task)
        elif task.task_type == TaskType.OPTIMIZATION:
            return await self._optimize_task(task)

        return {"status": "completed"}

    async def _generate_code(self, task: TaskNode) -> Dict[str, Any]:
        """Generate code for task"""
        # This would call LLM in real implementation
        artifact_id = f"artifact_{task.task_id}"

        # Generate sample code
        content = f'''"""
Generated module for: {task.description}
"""

class GeneratedModule:
    """Auto-generated module"""

    def __init__(self):
        self.initialized = True

    async def execute(self):
        """Execute module logic"""
        return {{"status": "success"}}
'''

        artifact = CodeArtifact(
            artifact_id=artifact_id,
            file_path=f"generated/{artifact_id}.py",
            content=content,
            language="python",
            artifact_type="module"
        )

        self.generated_artifacts[artifact_id] = artifact

        return {"artifact_id": artifact_id, "status": "generated"}

    async def _validate_task(self, task: TaskNode) -> Dict[str, Any]:
        """Validate code"""
        return {"status": "validated", "score": 0.9}

    async def _generate_tests(self, task: TaskNode) -> Dict[str, Any]:
        """Generate tests"""
        artifact_id = f"test_{task.task_id}"

        content = f'''"""
Generated tests for: {task.description}
"""

import pytest

class TestGeneratedModule:
    """Auto-generated tests"""

    def test_initialization(self):
        """Test module initialization"""
        module = GeneratedModule()
        assert module.initialized is True

    @pytest.mark.asyncio
    async def test_execution(self):
        """Test module execution"""
        module = GeneratedModule()
        result = await module.execute()
        assert result["status"] == "success"
'''

        artifact = CodeArtifact(
            artifact_id=artifact_id,
            file_path=f"generated/test_{artifact_id}.py",
            content=content,
            language="python",
            artifact_type="test"
        )

        self.generated_artifacts[artifact_id] = artifact

        return {"artifact_id": artifact_id, "status": "generated"}

    async def _optimize_task(self, task: TaskNode) -> Dict[str, Any]:
        """Optimize code"""
        return {"status": "optimized"}

    async def _validate_all_artifacts(self) -> List[ValidationResult]:
        """Validate all generated artifacts"""
        results = []

        for artifact in self.generated_artifacts.values():
            result = await self.validator.validate_code(artifact, {})
            artifact.quality_score = result.overall_score()
            results.append(result)

        return results

    async def _optimize_failed_artifacts(self, validation_results: List[ValidationResult]):
        """Optimize artifacts that failed validation"""
        for artifact_id, result in zip(self.generated_artifacts.keys(), validation_results):
            if not result.is_valid:
                artifact = self.generated_artifacts[artifact_id]
                optimized = await self.optimizer.optimize_code(
                    artifact,
                    optimization_goals=["performance", "readability"]
                )
                self.generated_artifacts[artifact_id] = optimized

    async def _self_evaluate(self) -> bool:
        """Self-evaluate and decide if iteration is needed"""
        # Check overall quality
        avg_quality = sum(
            artifact.quality_score
            for artifact in self.generated_artifacts.values()
        ) / len(self.generated_artifacts)

        # Iterate if quality is below threshold
        return avg_quality < 0.7

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get agent memory statistics"""
        return {
            "learned_patterns": len(self.memory.learned_patterns),
            "architecture_decisions": len(self.memory.architecture_decisions),
            "common_mistakes": len(self.memory.common_mistakes),
            "best_practices": len(self.memory.best_practices),
            "performance_metrics": self.memory.performance_metrics
        }
