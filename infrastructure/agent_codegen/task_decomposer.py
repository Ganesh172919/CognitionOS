"""
Task Decomposer - Intelligent Task Breakdown Engine

Decomposes complex tasks into smaller, manageable subtasks with
intelligent complexity analysis.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import logging
import re

logger = logging.getLogger(__name__)


class ComplexityLevel(str, Enum):
    """Task complexity levels"""
    TRIVIAL = "trivial"  # < 30 min
    SIMPLE = "simple"  # 30min - 2hr
    MODERATE = "moderate"  # 2hr - 1 day
    COMPLEX = "complex"  # 1-3 days
    VERY_COMPLEX = "very_complex"  # > 3 days


@dataclass
class TaskBreakdown:
    """Result of task decomposition"""
    original_task: str
    complexity: ComplexityLevel
    subtasks: List[Dict[str, Any]]
    estimated_total_time_hours: float
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    risks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class ComplexityAnalyzer:
    """
    Analyzes task complexity using multiple factors
    """

    # Complexity indicators (word patterns that suggest complexity)
    COMPLEXITY_INDICATORS = {
        "high": [
            "distributed", "scalable", "fault-tolerant", "multi-tenant",
            "real-time", "high-performance", "optimization", "migration",
            "refactor", "redesign", "architecture", "security"
        ],
        "medium": [
            "integration", "api", "database", "authentication",
            "validation", "testing", "deployment", "monitoring"
        ],
        "low": [
            "fix", "update", "add", "remove", "modify",
            "format", "style", "documentation", "comment"
        ]
    }

    def analyze(self, task_description: str) -> tuple[ComplexityLevel, int]:
        """
        Analyze task complexity

        Returns:
            Tuple of (complexity_level, estimated_hours)
        """
        task_lower = task_description.lower()
        score = 0

        # Check for complexity indicators
        for indicator in self.COMPLEXITY_INDICATORS["high"]:
            if indicator in task_lower:
                score += 3

        for indicator in self.COMPLEXITY_INDICATORS["medium"]:
            if indicator in task_lower:
                score += 2

        for indicator in self.COMPLEXITY_INDICATORS["low"]:
            if indicator in task_lower:
                score += 1

        # Check for scope indicators
        if any(word in task_lower for word in ["entire", "complete", "full", "whole"]):
            score += 2

        if any(word in task_lower for word in ["multiple", "several", "various"]):
            score += 2

        # Estimate complexity level
        if score <= 2:
            complexity = ComplexityLevel.TRIVIAL
            hours = 0.5
        elif score <= 5:
            complexity = ComplexityLevel.SIMPLE
            hours = 2
        elif score <= 10:
            complexity = ComplexityLevel.MODERATE
            hours = 8
        elif score <= 15:
            complexity = ComplexityLevel.COMPLEX
            hours = 24
        else:
            complexity = ComplexityLevel.VERY_COMPLEX
            hours = 40

        return complexity, hours


class TaskDecomposer:
    """
    Intelligent task decomposition engine

    Breaks down complex tasks into manageable subtasks with proper
    dependency management.
    """

    def __init__(self, llm_provider: Optional[Any] = None):
        self.llm_provider = llm_provider
        self.complexity_analyzer = ComplexityAnalyzer()

    async def decompose(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        max_subtask_complexity: ComplexityLevel = ComplexityLevel.SIMPLE
    ) -> TaskBreakdown:
        """
        Decompose task into subtasks

        Args:
            task: High-level task description
            context: Optional context (existing code, constraints, etc.)
            max_subtask_complexity: Maximum complexity for each subtask

        Returns:
            TaskBreakdown with subtasks and metadata
        """
        logger.info(f"Decomposing task: {task[:50]}...")

        # Analyze overall complexity
        complexity, estimated_hours = self.complexity_analyzer.analyze(task)

        # If task is already simple enough, don't decompose
        if complexity in [ComplexityLevel.TRIVIAL, ComplexityLevel.SIMPLE]:
            return TaskBreakdown(
                original_task=task,
                complexity=complexity,
                subtasks=[{
                    "id": "task_1",
                    "description": task,
                    "estimated_hours": estimated_hours,
                    "complexity": complexity.value
                }],
                estimated_total_time_hours=estimated_hours
            )

        # Decompose complex task
        if self.llm_provider:
            subtasks = await self._decompose_with_llm(task, context, max_subtask_complexity)
        else:
            subtasks = self._decompose_heuristic(task, complexity)

        # Build dependency graph
        dependencies = self._identify_dependencies(subtasks)

        # Identify risks
        risks = self._identify_risks(task, subtasks, complexity)

        # Generate recommendations
        recommendations = self._generate_recommendations(subtasks, complexity)

        total_hours = sum(st.get("estimated_hours", 1) for st in subtasks)

        return TaskBreakdown(
            original_task=task,
            complexity=complexity,
            subtasks=subtasks,
            estimated_total_time_hours=total_hours,
            dependencies=dependencies,
            risks=risks,
            recommendations=recommendations
        )

    async def _decompose_with_llm(
        self,
        task: str,
        context: Optional[Dict[str, Any]],
        max_complexity: ComplexityLevel
    ) -> List[Dict[str, Any]]:
        """Use LLM for intelligent decomposition"""

        prompt = f"""Decompose the following software development task into subtasks:

Task: {task}

Context: {context if context else 'New implementation'}

Requirements:
- Each subtask should be completable in < 2 hours
- Include all necessary steps (design, implementation, testing, documentation)
- Consider dependencies between subtasks
- Be specific and actionable

Return JSON array with format:
[
  {{
    "id": "task_1",
    "description": "Subtask description",
    "estimated_hours": 1.5,
    "depends_on": [],
    "complexity": "simple"
  }},
  ...
]
"""

        response = await self.llm_provider.generate(prompt)

        # Parse JSON response
        import json
        try:
            subtasks = json.loads(response)
            return subtasks
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response, using heuristic decomposition")
            return self._decompose_heuristic(task, ComplexityLevel.MODERATE)

    def _decompose_heuristic(
        self,
        task: str,
        complexity: ComplexityLevel
    ) -> List[Dict[str, Any]]:
        """Heuristic-based decomposition (fallback)"""

        subtasks = []

        # Standard software development phases
        phases = [
            ("Design and Planning", 0.5),
            ("Core Implementation", 2.0),
            ("Testing", 1.0),
            ("Documentation", 0.5)
        ]

        # Adjust based on complexity
        if complexity == ComplexityLevel.VERY_COMPLEX:
            phases.insert(1, ("Architecture Design", 2.0))
            phases.append(("Performance Optimization", 1.5))

        # Detect specific components in task description
        task_lower = task.lower()

        if "api" in task_lower or "endpoint" in task_lower:
            subtasks.append({
                "id": f"task_{len(subtasks) + 1}",
                "description": "Define API specification and endpoints",
                "estimated_hours": 1.0,
                "complexity": "simple"
            })

        if "database" in task_lower or "schema" in task_lower:
            subtasks.append({
                "id": f"task_{len(subtasks) + 1}",
                "description": "Design database schema",
                "estimated_hours": 1.5,
                "complexity": "simple"
            })

        if "ui" in task_lower or "frontend" in task_lower:
            subtasks.append({
                "id": f"task_{len(subtasks) + 1}",
                "description": "Design UI components",
                "estimated_hours": 2.0,
                "complexity": "moderate"
            })

        # Add standard phases
        for phase, hours in phases:
            subtasks.append({
                "id": f"task_{len(subtasks) + 1}",
                "description": f"{phase} for: {task[:40]}...",
                "estimated_hours": hours,
                "complexity": "simple"
            })

        return subtasks

    def _identify_dependencies(
        self,
        subtasks: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Identify dependencies between subtasks"""

        dependencies = {}

        # Simple rule-based dependency detection
        for i, task in enumerate(subtasks):
            task_id = task["id"]
            task_desc = task["description"].lower()

            # Dependencies based on common patterns
            if i > 0:
                prev_task_id = subtasks[i-1]["id"]

                # Testing depends on implementation
                if "test" in task_desc:
                    dependencies[task_id] = [prev_task_id]

                # Documentation depends on implementation
                elif "document" in task_desc:
                    impl_tasks = [
                        st["id"] for st in subtasks
                        if "implement" in st["description"].lower()
                    ]
                    dependencies[task_id] = impl_tasks

                # Implementation depends on design
                elif "implement" in task_desc:
                    design_tasks = [
                        st["id"] for st in subtasks[:i]
                        if "design" in st["description"].lower()
                    ]
                    if design_tasks:
                        dependencies[task_id] = design_tasks

        return dependencies

    def _identify_risks(
        self,
        task: str,
        subtasks: List[Dict[str, Any]],
        complexity: ComplexityLevel
    ) -> List[str]:
        """Identify potential risks"""

        risks = []

        # Complexity-based risks
        if complexity == ComplexityLevel.VERY_COMPLEX:
            risks.append("High complexity may lead to scope creep")
            risks.append("Consider breaking into multiple phases")

        # Time-based risks
        total_hours = sum(st.get("estimated_hours", 1) for st in subtasks)
        if total_hours > 40:
            risks.append(f"Large time estimate ({total_hours}h) increases risk of delays")

        # Dependency risks
        if len(subtasks) > 10:
            risks.append("Many subtasks may complicate coordination")

        # Specific pattern risks
        task_lower = task.lower()
        if "migration" in task_lower:
            risks.append("Data migration requires careful planning and rollback strategy")
        if "security" in task_lower:
            risks.append("Security features require thorough testing and audit")
        if "performance" in task_lower:
            risks.append("Performance optimization requires benchmarking and monitoring")

        return risks

    def _generate_recommendations(
        self,
        subtasks: List[Dict[str, Any]],
        complexity: ComplexityLevel
    ) -> List[str]:
        """Generate recommendations for task execution"""

        recommendations = []

        # Complexity-based recommendations
        if complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.VERY_COMPLEX]:
            recommendations.append("Start with proof-of-concept for critical components")
            recommendations.append("Set up monitoring and logging early")

        # Number of subtasks
        if len(subtasks) > 8:
            recommendations.append("Consider parallel execution where possible")

        # Testing recommendations
        if any("test" in st["description"].lower() for st in subtasks):
            recommendations.append("Use TDD approach for better code quality")

        # General recommendations
        recommendations.append("Regular code reviews after each major subtask")
        recommendations.append("Document key decisions and trade-offs")

        return recommendations

    async def refine_breakdown(
        self,
        breakdown: TaskBreakdown,
        feedback: str
    ) -> TaskBreakdown:
        """
        Refine task breakdown based on feedback

        Args:
            breakdown: Original breakdown
            feedback: Feedback on the breakdown

        Returns:
            Refined breakdown
        """
        logger.info("Refining task breakdown based on feedback")

        if self.llm_provider:
            prompt = f"""Refine the following task breakdown:

Original Task: {breakdown.original_task}

Current Subtasks:
{self._format_subtasks(breakdown.subtasks)}

Feedback: {feedback}

Provide refined subtasks as JSON array with same format."""

            response = await self.llm_provider.generate(prompt)

            import json
            try:
                refined_subtasks = json.loads(response)
                breakdown.subtasks = refined_subtasks
                breakdown.estimated_total_time_hours = sum(
                    st.get("estimated_hours", 1) for st in refined_subtasks
                )
            except json.JSONDecodeError:
                logger.error("Failed to parse refinement response")

        return breakdown

    def _format_subtasks(self, subtasks: List[Dict[str, Any]]) -> str:
        """Format subtasks for display"""
        lines = []
        for st in subtasks:
            lines.append(f"- {st['description']} ({st.get('estimated_hours', 1)}h)")
        return "\n".join(lines)
