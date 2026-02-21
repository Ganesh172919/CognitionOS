"""
Self-Evaluator - Agent Self-Evaluation and Iteration Engine

Evaluates generated code quality and iterates to improve it.
Implements self-critique and refinement loops.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EvaluationCriteria(str, Enum):
    """Criteria for code evaluation"""
    CORRECTNESS = "correctness"
    READABILITY = "readability"
    EFFICIENCY = "efficiency"
    MAINTAINABILITY = "maintainability"
    TESTABILITY = "testability"
    SECURITY = "security"
    SCALABILITY = "scalability"


@dataclass
class EvaluationMetrics:
    """Metrics for code evaluation"""
    overall_score: float  # 0-1
    criteria_scores: Dict[EvaluationCriteria, float] = field(default_factory=dict)

    # Specific metrics
    code_quality: float = 0.0
    test_coverage: float = 0.0
    documentation_completeness: float = 0.0
    performance_score: float = 0.0

    # Improvement suggestions
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)

    # Metadata
    evaluated_at: datetime = field(default_factory=datetime.utcnow)

    def is_acceptable(self, threshold: float = 0.7) -> bool:
        """Check if code meets quality threshold"""
        return self.overall_score >= threshold


@dataclass
class IterationResult:
    """Result of a code improvement iteration"""
    iteration_number: int
    improved_code: str
    metrics: EvaluationMetrics
    changes_made: List[str] = field(default_factory=list)
    rationale: str = ""


class SelfEvaluator:
    """
    Self-evaluation and improvement engine

    Evaluates generated code and iteratively improves it until
    it meets quality standards.
    """

    def __init__(self, llm_provider: Optional[Any] = None):
        self.llm_provider = llm_provider

    async def evaluate(
        self,
        code: str,
        language: str,
        requirements: str,
        validation_result: Optional[Any] = None
    ) -> EvaluationMetrics:
        """
        Evaluate code quality

        Args:
            code: Code to evaluate
            language: Programming language
            requirements: Original requirements
            validation_result: Optional validation result

        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating code quality")

        metrics = EvaluationMetrics(overall_score=0.0)

        # Evaluate each criterion
        for criterion in EvaluationCriteria:
            score = await self._evaluate_criterion(
                code, language, requirements, criterion, validation_result
            )
            metrics.criteria_scores[criterion] = score

        # Calculate overall score
        metrics.overall_score = sum(metrics.criteria_scores.values()) / len(metrics.criteria_scores)

        # Generate strengths and weaknesses
        metrics.strengths = self._identify_strengths(code, metrics.criteria_scores)
        metrics.weaknesses = self._identify_weaknesses(code, metrics.criteria_scores)

        # Generate improvement suggestions
        metrics.improvement_suggestions = await self._generate_suggestions(
            code, language, metrics.weaknesses
        )

        logger.info(f"Evaluation complete: overall score = {metrics.overall_score:.2f}")
        return metrics

    async def _evaluate_criterion(
        self,
        code: str,
        language: str,
        requirements: str,
        criterion: EvaluationCriteria,
        validation_result: Optional[Any]
    ) -> float:
        """Evaluate a specific criterion"""

        if criterion == EvaluationCriteria.CORRECTNESS:
            return self._evaluate_correctness(code, requirements, validation_result)
        elif criterion == EvaluationCriteria.READABILITY:
            return self._evaluate_readability(code, language)
        elif criterion == EvaluationCriteria.EFFICIENCY:
            return self._evaluate_efficiency(code, language)
        elif criterion == EvaluationCriteria.MAINTAINABILITY:
            return self._evaluate_maintainability(code, language)
        elif criterion == EvaluationCriteria.TESTABILITY:
            return self._evaluate_testability(code, language)
        elif criterion == EvaluationCriteria.SECURITY:
            return self._evaluate_security(code, validation_result)
        elif criterion == EvaluationCriteria.SCALABILITY:
            return self._evaluate_scalability(code, language)

        return 0.5  # Default middle score

    def _evaluate_correctness(
        self,
        code: str,
        requirements: str,
        validation_result: Optional[Any]
    ) -> float:
        """Evaluate correctness"""
        score = 0.5

        # Syntax validity
        if validation_result and validation_result.syntax_valid:
            score += 0.3

        # Type safety
        if validation_result and validation_result.type_safe:
            score += 0.2

        return min(score, 1.0)

    def _evaluate_readability(self, code: str, language: str) -> float:
        """Evaluate readability"""
        score = 0.5

        # Check for comments/docstrings
        if '"""' in code or "'''" in code or '//' in code or '/*' in code:
            score += 0.2

        # Check for reasonable line length
        lines = code.split('\n')
        long_lines = sum(1 for line in lines if len(line) > 100)
        if long_lines / max(len(lines), 1) < 0.1:
            score += 0.2

        # Check for descriptive names (not single letters)
        import re
        single_letter_vars = len(re.findall(r'\b[a-z]\b', code))
        if single_letter_vars < 5:
            score += 0.1

        return min(score, 1.0)

    def _evaluate_efficiency(self, code: str, language: str) -> float:
        """Evaluate efficiency"""
        score = 0.7  # Default to good

        # Check for obvious inefficiencies
        if 'for' in code and 'for' in code:
            # Nested loops might indicate O(n²) complexity
            score -= 0.2

        # Check for unnecessary operations
        if code.count('append') > 10:
            # Many appends might be inefficient
            score -= 0.1

        return max(score, 0.3)

    def _evaluate_maintainability(self, code: str, language: str) -> float:
        """Evaluate maintainability"""
        score = 0.5

        # Check complexity
        lines = len([l for l in code.split('\n') if l.strip()])
        if lines < 100:
            score += 0.2

        # Check for modular design (functions/classes)
        if 'def ' in code or 'function ' in code or 'class ' in code:
            score += 0.2

        # Check for error handling
        if 'try' in code or 'catch' in code or 'except' in code:
            score += 0.1

        return min(score, 1.0)

    def _evaluate_testability(self, code: str, language: str) -> float:
        """Evaluate testability"""
        score = 0.5

        # Check for pure functions (easier to test)
        if language == "python":
            import re
            functions = re.findall(r'def\s+(\w+)\s*\(', code)
            if functions:
                score += 0.2

        # Check for dependency injection patterns
        if '__init__' in code and 'self' in code:
            score += 0.2

        return min(score, 1.0)

    def _evaluate_security(
        self,
        code: str,
        validation_result: Optional[Any]
    ) -> float:
        """Evaluate security"""
        score = 0.9  # Start high, deduct for issues

        # Check validation result for security issues
        if validation_result and validation_result.security_issues > 0:
            score -= 0.2 * validation_result.security_issues

        # Check for dangerous patterns
        dangerous = ['eval(', 'exec(', 'subprocess.call']
        for pattern in dangerous:
            if pattern in code:
                score -= 0.2

        return max(score, 0.0)

    def _evaluate_scalability(self, code: str, language: str) -> float:
        """Evaluate scalability"""
        score = 0.6  # Default moderate

        # Check for async/await (good for scalability)
        if 'async ' in code or 'await ' in code:
            score += 0.2

        # Check for caching
        if 'cache' in code.lower():
            score += 0.1

        return min(score, 1.0)

    def _identify_strengths(
        self,
        code: str,
        scores: Dict[EvaluationCriteria, float]
    ) -> List[str]:
        """Identify code strengths"""
        strengths = []

        for criterion, score in scores.items():
            if score >= 0.8:
                strengths.append(f"Strong {criterion.value}: {score:.2f}")

        return strengths

    def _identify_weaknesses(
        self,
        code: str,
        scores: Dict[EvaluationCriteria, float]
    ) -> List[str]:
        """Identify code weaknesses"""
        weaknesses = []

        for criterion, score in scores.items():
            if score < 0.6:
                weaknesses.append(f"Weak {criterion.value}: {score:.2f}")

        return weaknesses

    async def _generate_suggestions(
        self,
        code: str,
        language: str,
        weaknesses: List[str]
    ) -> List[str]:
        """Generate improvement suggestions"""

        if self.llm_provider and weaknesses:
            prompt = f"""Analyze this {language} code and provide specific improvement suggestions:

Code:
```{language}
{code}
```

Identified weaknesses:
{chr(10).join(f"- {w}" for w in weaknesses)}

Provide 3-5 specific, actionable suggestions to improve the code."""

            response = await self.llm_provider.generate(prompt)
            # Parse suggestions from response
            suggestions = [s.strip() for s in response.split('\n') if s.strip().startswith('-')]
            return suggestions[:5]

        # Fallback: generic suggestions based on weaknesses
        suggestions = []
        for weakness in weaknesses:
            if 'readability' in weakness.lower():
                suggestions.append("Add descriptive variable names and comments")
            if 'security' in weakness.lower():
                suggestions.append("Add input validation and sanitization")
            if 'efficiency' in weakness.lower():
                suggestions.append("Optimize loops and reduce complexity")

        return suggestions


class IterationEngine:
    """
    Iterative improvement engine

    Repeatedly evaluates and improves code until quality threshold is met.
    """

    def __init__(
        self,
        evaluator: SelfEvaluator,
        code_generator: Optional[Any] = None
    ):
        self.evaluator = evaluator
        self.code_generator = code_generator
        self.max_iterations = 5

    async def improve_iteratively(
        self,
        initial_code: str,
        language: str,
        requirements: str,
        quality_threshold: float = 0.8,
        validator: Optional[Any] = None
    ) -> tuple[str, List[IterationResult]]:
        """
        Iteratively improve code

        Args:
            initial_code: Starting code
            language: Programming language
            requirements: Original requirements
            quality_threshold: Minimum acceptable quality
            validator: Optional code validator

        Returns:
            Tuple of (final_code, iteration_history)
        """
        logger.info("Starting iterative improvement process")

        current_code = initial_code
        iterations = []

        for i in range(self.max_iterations):
            logger.info(f"Iteration {i + 1}/{self.max_iterations}")

            # Validate code
            validation_result = None
            if validator:
                validation_result = await validator.validate(current_code, language)

            # Evaluate current code
            metrics = await self.evaluator.evaluate(
                current_code,
                language,
                requirements,
                validation_result
            )

            # Check if quality threshold met
            if metrics.is_acceptable(quality_threshold):
                logger.info(f"Quality threshold met at iteration {i + 1}")
                break

            # Generate improvements
            if self.code_generator and metrics.improvement_suggestions:
                improved_code = await self._apply_improvements(
                    current_code,
                    language,
                    metrics.improvement_suggestions
                )

                if improved_code and improved_code != current_code:
                    iteration_result = IterationResult(
                        iteration_number=i + 1,
                        improved_code=improved_code,
                        metrics=metrics,
                        changes_made=metrics.improvement_suggestions,
                        rationale=f"Applied {len(metrics.improvement_suggestions)} improvements"
                    )
                    iterations.append(iteration_result)
                    current_code = improved_code
                else:
                    logger.warning("No improvements generated, stopping iteration")
                    break
            else:
                logger.warning("Cannot improve further without code generator")
                break

        logger.info(f"Completed {len(iterations)} iterations")
        return current_code, iterations

    async def _apply_improvements(
        self,
        code: str,
        language: str,
        suggestions: List[str]
    ) -> Optional[str]:
        """Apply improvement suggestions to code"""

        if not self.code_generator:
            return None

        improvement_description = "\n".join(f"- {s}" for s in suggestions)

        # Use code generator to refactor
        from .code_generator import CodeContext, CodeStyle
        context = CodeContext(
            language=language,
            style=CodeStyle.OBJECT_ORIENTED
        )

        result = await self.code_generator.refactor_code(
            existing_code=code,
            refactoring_goal=f"Apply improvements:\n{improvement_description}",
            context=context
        )

        return result.code
