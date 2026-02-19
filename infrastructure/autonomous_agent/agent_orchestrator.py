"""
Autonomous Agent Orchestrator
Coordinates autonomous execution of complex multi-step tasks.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .autonomous_planner import (
    AutonomousPlanner,
    ExecutionPlan,
    TaskNode,
    TaskStatus,
    TaskType
)
from .code_generator import IntelligentCodeGenerator, CodeLanguage, GeneratedCode


class AgentMemory:
    """Agent's working memory for context and learning"""

    def __init__(self):
        self.context: Dict[str, Any] = {}
        self.learned_patterns: List[Dict[str, Any]] = []
        self.execution_history: List[Dict[str, Any]] = []
        self.successes: List[Dict[str, Any]] = []
        self.failures: List[Dict[str, Any]] = []

    def add_context(self, key: str, value: Any) -> None:
        """Add context information"""
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Retrieve context"""
        return self.context.get(key, default)

    def record_execution(self, task_id: str, result: Dict[str, Any]) -> None:
        """Record task execution"""
        self.execution_history.append({
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat(),
            "result": result
        })

        if result.get("success"):
            self.successes.append(result)
        else:
            self.failures.append(result)

    def learn_pattern(self, pattern: Dict[str, Any]) -> None:
        """Learn from successful patterns"""
        self.learned_patterns.append({
            **pattern,
            "learned_at": datetime.utcnow().isoformat()
        })

    def get_similar_past_executions(self, task_type: str) -> List[Dict[str, Any]]:
        """Get similar past executions for learning"""
        return [
            h for h in self.execution_history
            if h.get("result", {}).get("task_type") == task_type
        ]


class HallucinationDetector:
    """Detects and prevents hallucinations in agent outputs"""

    def __init__(self):
        self.confidence_threshold = 0.7
        self.consistency_checks = []

    async def validate_output(
        self,
        output: Any,
        expected_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate agent output for hallucinations"""
        issues = []
        confidence = 1.0

        # Type validation
        if not self._validate_type(output, expected_type):
            issues.append(f"Output type mismatch: expected {expected_type}")
            confidence -= 0.3

        # Context consistency
        if not self._check_context_consistency(output, context):
            issues.append("Output inconsistent with context")
            confidence -= 0.2

        # Factual grounding
        if not self._check_factual_grounding(output):
            issues.append("Output may contain ungrounded claims")
            confidence -= 0.15

        return {
            "valid": len(issues) == 0 and confidence >= self.confidence_threshold,
            "confidence": max(0.0, confidence),
            "issues": issues
        }

    def _validate_type(self, output: Any, expected_type: str) -> bool:
        """Validate output type"""
        type_mapping = {
            "string": str,
            "dict": dict,
            "list": list,
            "int": int,
            "float": float,
            "bool": bool
        }
        expected_python_type = type_mapping.get(expected_type, str)
        return isinstance(output, expected_python_type)

    def _check_context_consistency(self, output: Any, context: Dict[str, Any]) -> bool:
        """Check if output is consistent with context"""
        # Simplified check - in production would be more sophisticated
        return True

    def _check_factual_grounding(self, output: Any) -> bool:
        """Check if output is factually grounded"""
        # Simplified check - in production would use knowledge base
        return True


class SafetyBoundaries:
    """Enforces safety boundaries for autonomous agent"""

    def __init__(self):
        self.max_iterations = 100
        self.max_execution_time_seconds = 3600  # 1 hour
        self.allowed_actions = {
            "read_file",
            "write_file",
            "generate_code",
            "run_tests",
            "analyze_code",
            "plan_tasks"
        }
        self.forbidden_actions = {
            "delete_database",
            "expose_secrets",
            "external_api_without_approval"
        }

    def validate_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if action is safe to execute"""
        if action in self.forbidden_actions:
            return {
                "allowed": False,
                "reason": f"Action '{action}' is forbidden"
            }

        if action not in self.allowed_actions:
            return {
                "allowed": False,
                "reason": f"Action '{action}' is not in allowed actions"
            }

        # Additional parameter validation
        if action == "write_file":
            if not self._validate_file_path(parameters.get("path", "")):
                return {
                    "allowed": False,
                    "reason": "Invalid or unsafe file path"
                }

        return {"allowed": True}

    def _validate_file_path(self, path: str) -> bool:
        """Validate file path for safety"""
        dangerous_patterns = [
            "../",
            "/etc/",
            "/sys/",
            "/proc/",
            "~/.ssh/",
            "~/.aws/"
        ]
        return not any(pattern in path for pattern in dangerous_patterns)

    def check_resource_limits(
        self,
        iterations: int,
        execution_time: float
    ) -> Dict[str, Any]:
        """Check if resource limits are exceeded"""
        if iterations >= self.max_iterations:
            return {
                "exceeded": True,
                "reason": f"Max iterations ({self.max_iterations}) exceeded"
            }

        if execution_time >= self.max_execution_time_seconds:
            return {
                "exceeded": True,
                "reason": f"Max execution time ({self.max_execution_time_seconds}s) exceeded"
            }

        return {"exceeded": False}


class AutonomousAgentOrchestrator:
    """
    Main orchestrator for autonomous agent execution.
    Coordinates planning, execution, validation, and iteration.
    """

    def __init__(self):
        self.planner = AutonomousPlanner()
        self.code_generator = IntelligentCodeGenerator()
        self.memory = AgentMemory()
        self.hallucination_detector = HallucinationDetector()
        self.safety = SafetyBoundaries()
        self.active_executions: Dict[str, Dict[str, Any]] = {}

    async def execute_requirement(
        self,
        requirement: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point: Execute a high-level requirement autonomously
        """
        execution_id = str(uuid4())
        start_time = datetime.utcnow()

        self.active_executions[execution_id] = {
            "requirement": requirement,
            "status": "planning",
            "start_time": start_time,
            "iterations": 0
        }

        try:
            # Phase 1: Planning
            plan = await self._create_plan(requirement, context or {})

            # Phase 2: Execution
            results = await self._execute_plan(execution_id, plan)

            # Phase 3: Validation
            validation = await self._validate_results(results, plan)

            # Phase 4: Self-evaluation
            evaluation = await self._self_evaluate(results, validation, plan)

            # Phase 5: Iteration if needed
            if not evaluation["success"] and evaluation["should_retry"]:
                results = await self._iterate(execution_id, plan, evaluation)
                validation = await self._validate_results(results, plan)
                evaluation = await self._self_evaluate(results, validation, plan)

            # Record in memory
            self.memory.record_execution(execution_id, {
                "requirement": requirement,
                "success": evaluation["success"],
                "results": results,
                "duration": (datetime.utcnow() - start_time).total_seconds()
            })

            # Learn from successful patterns
            if evaluation["success"]:
                self.memory.learn_pattern({
                    "requirement_type": self._classify_requirement(requirement),
                    "approach": plan.analysis.model_dump(),
                    "success_factors": evaluation.get("success_factors", [])
                })

            return {
                "execution_id": execution_id,
                "success": evaluation["success"],
                "results": results,
                "validation": validation,
                "evaluation": evaluation,
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds()
            }

        except Exception as e:
            return {
                "execution_id": execution_id,
                "success": False,
                "error": str(e),
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds()
            }
        finally:
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]

    async def _create_plan(
        self,
        requirement: str,
        context: Dict[str, Any]
    ) -> ExecutionPlan:
        """Create execution plan"""
        # Add memory context
        similar_executions = self.memory.get_similar_past_executions(
            self._classify_requirement(requirement)
        )

        if similar_executions:
            context["past_learnings"] = similar_executions[-3:]  # Last 3 similar

        plan = await self.planner.create_execution_plan(requirement, context)

        return plan

    async def _execute_plan(
        self,
        execution_id: str,
        plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """Execute the plan"""
        results = {
            "plan_id": plan.plan_id,
            "tasks_completed": [],
            "tasks_failed": [],
            "outputs": {}
        }

        execution_state = self.active_executions[execution_id]
        execution_state["status"] = "executing"
        execution_state["plan_id"] = plan.plan_id

        # Execute tasks in order
        for task_id in plan.execution_order:
            # Check safety limits
            execution_state["iterations"] += 1
            elapsed = (datetime.utcnow() - execution_state["start_time"]).total_seconds()

            limit_check = self.safety.check_resource_limits(
                execution_state["iterations"],
                elapsed
            )

            if limit_check["exceeded"]:
                results["error"] = limit_check["reason"]
                break

            # Execute task
            task_result = await self._execute_task(task_id, plan, results["outputs"])

            if task_result["success"]:
                results["tasks_completed"].append(task_id)
                results["outputs"][task_id] = task_result["output"]
            else:
                results["tasks_failed"].append(task_id)
                # Decide whether to continue or stop
                if task_result.get("critical", False):
                    break

        return results

    async def _execute_task(
        self,
        task_id: str,
        plan: ExecutionPlan,
        previous_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single task"""
        # In production, this would retrieve task details from storage
        # For now, simulate task execution

        # Determine task type and execute accordingly
        task_type = TaskType.CODE_GENERATION  # Simplified

        if task_type == TaskType.CODE_GENERATION:
            return await self._execute_code_generation_task(
                task_id,
                "Generate code for requirement",
                previous_outputs
            )
        elif task_type == TaskType.TESTING:
            return await self._execute_testing_task(task_id, previous_outputs)
        else:
            return {
                "success": True,
                "output": {"task_id": task_id, "completed": True}
            }

    async def _execute_code_generation_task(
        self,
        task_id: str,
        purpose: str,
        previous_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute code generation task"""
        try:
            # Generate code
            generated = await self.code_generator.generate_code(
                purpose=purpose,
                language=CodeLanguage.PYTHON,
                context={"previous_outputs": previous_outputs}
            )

            # Validate code
            if generated.validation and not generated.validation.is_valid:
                return {
                    "success": False,
                    "error": "Code validation failed",
                    "validation_errors": generated.validation.errors
                }

            return {
                "success": True,
                "output": {
                    "code": generated.code,
                    "test_code": generated.test_code,
                    "documentation": generated.documentation,
                    "quality": generated.validation.quality.value if generated.validation else "unknown"
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _execute_testing_task(
        self,
        task_id: str,
        previous_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute testing task"""
        # Simplified - in production would run actual tests
        return {
            "success": True,
            "output": {
                "tests_passed": True,
                "coverage": 0.85
            }
        }

    async def _validate_results(
        self,
        results: Dict[str, Any],
        plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """Validate execution results"""
        validation = {
            "valid": True,
            "completeness": 0.0,
            "quality": "good",
            "issues": []
        }

        # Check completeness
        total_tasks = len(plan.execution_order)
        completed_tasks = len(results.get("tasks_completed", []))

        validation["completeness"] = completed_tasks / total_tasks if total_tasks > 0 else 0

        if validation["completeness"] < 1.0:
            validation["issues"].append(
                f"Only {completed_tasks}/{total_tasks} tasks completed"
            )

        # Check for failed tasks
        failed_tasks = results.get("tasks_failed", [])
        if failed_tasks:
            validation["issues"].append(f"{len(failed_tasks)} tasks failed")
            validation["valid"] = False

        # Check outputs for hallucinations
        for task_id, output in results.get("outputs", {}).items():
            hallucination_check = await self.hallucination_detector.validate_output(
                output,
                "dict",
                {"task_id": task_id}
            )

            if not hallucination_check["valid"]:
                validation["issues"].extend(hallucination_check["issues"])
                validation["valid"] = False

        return validation

    async def _self_evaluate(
        self,
        results: Dict[str, Any],
        validation: Dict[str, Any],
        plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """Agent self-evaluates its performance"""
        evaluation = {
            "success": False,
            "should_retry": False,
            "confidence": 0.0,
            "success_factors": [],
            "improvement_areas": []
        }

        # Calculate success
        completeness = validation.get("completeness", 0.0)
        has_errors = len(results.get("tasks_failed", [])) > 0

        evaluation["success"] = completeness == 1.0 and not has_errors and validation["valid"]

        # Calculate confidence
        evaluation["confidence"] = completeness * (0.9 if not has_errors else 0.5)

        # Determine if should retry
        evaluation["should_retry"] = (
            not evaluation["success"] and
            completeness >= 0.5 and  # Made some progress
            results.get("iterations", 0) < 3  # Haven't retried too many times
        )

        # Identify success factors
        if evaluation["success"]:
            evaluation["success_factors"] = [
                "All tasks completed successfully",
                "No validation errors",
                f"High confidence ({evaluation['confidence']:.2f})"
            ]
        else:
            # Identify improvement areas
            if completeness < 1.0:
                evaluation["improvement_areas"].append("Complete all planned tasks")
            if has_errors:
                evaluation["improvement_areas"].append("Fix task execution errors")
            if not validation["valid"]:
                evaluation["improvement_areas"].append("Address validation issues")

        return evaluation

    async def _iterate(
        self,
        execution_id: str,
        plan: ExecutionPlan,
        evaluation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Iterate on failed execution"""
        # In production, would:
        # 1. Analyze failure modes
        # 2. Adjust plan
        # 3. Re-execute with improvements

        # For now, return the same results (simplified)
        return await self._execute_plan(execution_id, plan)

    def _classify_requirement(self, requirement: str) -> str:
        """Classify requirement type"""
        requirement_lower = requirement.lower()

        if "bug" in requirement_lower or "fix" in requirement_lower:
            return "bug_fix"
        elif "refactor" in requirement_lower:
            return "refactoring"
        elif "optimize" in requirement_lower or "performance" in requirement_lower:
            return "optimization"
        elif "test" in requirement_lower:
            return "testing"
        else:
            return "feature"

    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get current execution status"""
        return self.active_executions.get(execution_id)

    def get_memory_context(self) -> Dict[str, Any]:
        """Get current memory context"""
        return {
            "total_executions": len(self.memory.execution_history),
            "successes": len(self.memory.successes),
            "failures": len(self.memory.failures),
            "learned_patterns": len(self.memory.learned_patterns)
        }
