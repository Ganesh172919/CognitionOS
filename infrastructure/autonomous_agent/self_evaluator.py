"""
Agent Self-Evaluator — CognitionOS AI Intelligence Layer

Multi-dimensional output quality scoring system that evaluates
agent-generated code and content across:
- Correctness (syntax, compilation, logic)
- Completeness (requirement coverage)
- Style (formatting, conventions, readability)
- Performance (complexity analysis)
- Security (common vulnerability patterns)

Triggers re-iteration when output quality falls below configurable thresholds.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class QualityDimension(str, Enum):
    """Dimensions of output quality."""
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    STYLE = "style"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"


class EvaluationVerdict(str, Enum):
    """Overall evaluation result."""
    EXCELLENT = "excellent"    # 90-100
    GOOD = "good"              # 75-89
    ACCEPTABLE = "acceptable"  # 60-74
    NEEDS_WORK = "needs_work"  # 40-59
    REJECT = "reject"          # 0-39


@dataclass
class DimensionScore:
    """Score for a single quality dimension."""
    dimension: QualityDimension
    score: float  # 0-100
    max_score: float = 100.0
    details: str = ""
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    @property
    def normalized(self) -> float:
        return min(self.score / self.max_score, 1.0) if self.max_score > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "score": round(self.score, 1),
            "max_score": self.max_score,
            "normalized": round(self.normalized, 3),
            "details": self.details,
            "issues": self.issues,
            "suggestions": self.suggestions,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result from the self-evaluator."""
    evaluation_id: str
    composite_score: float
    verdict: EvaluationVerdict
    dimension_scores: Dict[QualityDimension, DimensionScore] = field(default_factory=dict)
    should_iterate: bool = False
    iteration_guidance: str = ""
    evaluated_at: float = field(default_factory=time.time)
    evaluation_duration_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "composite_score": round(self.composite_score, 1),
            "verdict": self.verdict.value,
            "should_iterate": self.should_iterate,
            "iteration_guidance": self.iteration_guidance,
            "dimensions": {
                dim.value: score.to_dict()
                for dim, score in self.dimension_scores.items()
            },
            "evaluated_at": self.evaluated_at,
            "evaluation_duration_ms": round(self.evaluation_duration_ms, 1),
        }


# ── Dimension Analyzers ──

class CorrectnessAnalyzer:
    """Analyze code correctness via AST parsing and pattern matching."""

    def analyze(self, code: str, language: str = "python") -> DimensionScore:
        issues = []
        suggestions = []
        score = 100.0

        if language == "python":
            score, issues, suggestions = self._analyze_python(code)

        return DimensionScore(
            dimension=QualityDimension.CORRECTNESS,
            score=max(0, score),
            details=f"Analyzed {len(code)} characters of {language} code",
            issues=issues,
            suggestions=suggestions,
        )

    def _analyze_python(self, code: str) -> Tuple[float, List[str], List[str]]:
        score = 100.0
        issues = []
        suggestions = []

        # Try AST parse
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            issues.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return 10.0, issues, ["Fix the syntax error before proceeding"]

        # Check for common issues
        for node in ast.walk(tree):
            # Bare except
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append("Bare 'except:' clause found — catches too broadly")
                score -= 5
                suggestions.append("Use specific exception types")

            # Mutable default arguments
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append(
                            f"Mutable default argument in '{node.name}()'"
                        )
                        score -= 3
                        suggestions.append("Use None as default, set inside function")

            # assert in production code
            if isinstance(node, ast.Assert):
                issues.append("Assert statement found — removed in optimized mode")
                score -= 2

            # Global statement
            if isinstance(node, ast.Global):
                issues.append("Global statement found — consider dependency injection")
                score -= 2

        # Check for undefined references (basic)
        defined_names: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined_names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                defined_names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_names.add(target.id)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    defined_names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    defined_names.add(alias.asname or alias.name)

        return max(0, score), issues, suggestions


class CompletenessAnalyzer:
    """Analyze whether output meets the original requirements."""

    def analyze(
        self,
        code: str,
        requirements: List[str],
        language: str = "python",
    ) -> DimensionScore:
        issues = []
        suggestions = []
        met_requirements = 0

        if not requirements:
            return DimensionScore(
                dimension=QualityDimension.COMPLETENESS,
                score=100.0,
                details="No requirements specified",
            )

        code_lower = code.lower()

        for req in requirements:
            # Check if requirement keywords appear in code
            keywords = self._extract_keywords(req)
            matched = sum(1 for kw in keywords if kw.lower() in code_lower)
            if matched >= len(keywords) * 0.5:
                met_requirements += 1
            else:
                issues.append(f"Requirement possibly unmet: '{req}'")
                suggestions.append(f"Ensure '{req}' is fully implemented")

        coverage = (met_requirements / len(requirements)) * 100 if requirements else 100

        return DimensionScore(
            dimension=QualityDimension.COMPLETENESS,
            score=coverage,
            details=f"{met_requirements}/{len(requirements)} requirements covered",
            issues=issues,
            suggestions=suggestions,
        )

    def _extract_keywords(self, requirement: str) -> List[str]:
        """Extract meaningful keywords from a requirement."""
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "shall", "should", "may", "might", "must", "can",
            "could", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "and", "or", "but", "not",
            "this", "that", "it", "its", "each", "every", "all", "any",
        }
        words = re.findall(r'\b\w+\b', requirement.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]


class StyleAnalyzer:
    """Analyze code style, formatting, and readability."""

    def analyze(self, code: str, language: str = "python") -> DimensionScore:
        issues = []
        suggestions = []
        score = 100.0

        lines = code.split("\n")

        # Line length
        long_lines = sum(1 for line in lines if len(line) > 120)
        if long_lines > 0:
            penalty = min(long_lines * 2, 15)
            score -= penalty
            issues.append(f"{long_lines} lines exceed 120 characters")
            suggestions.append("Break long lines for readability")

        # Docstrings (for Python)
        if language == "python":
            try:
                tree = ast.parse(code)
                functions = [
                    n for n in ast.walk(tree)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                undocumented = 0
                for func in functions:
                    if not (func.body and isinstance(func.body[0], ast.Expr)
                            and isinstance(func.body[0].value, ast.Constant)
                            and isinstance(func.body[0].value.value, str)):
                        if not func.name.startswith("_"):
                            undocumented += 1
                if undocumented > 0 and functions:
                    doc_ratio = 1 - (undocumented / len(functions))
                    if doc_ratio < 0.5:
                        score -= 10
                        issues.append(
                            f"{undocumented}/{len(functions)} public functions lack docstrings"
                        )
                        suggestions.append("Add docstrings to public functions")
            except SyntaxError:
                pass

        # Blank line consistency
        consecutive_blanks = 0
        max_blanks = 0
        for line in lines:
            if line.strip() == "":
                consecutive_blanks += 1
                max_blanks = max(max_blanks, consecutive_blanks)
            else:
                consecutive_blanks = 0
        if max_blanks > 3:
            score -= 5
            issues.append(f"Excessive blank lines (max consecutive: {max_blanks})")

        # TODO/FIXME/HACK comments
        todo_count = sum(
            1 for line in lines
            if any(marker in line.upper() for marker in ["TODO", "FIXME", "HACK", "XXX"])
        )
        if todo_count > 3:
            score -= min(todo_count, 10)
            issues.append(f"{todo_count} TODO/FIXME/HACK comments found")
            suggestions.append("Resolve pending TODO items")

        return DimensionScore(
            dimension=QualityDimension.STYLE,
            score=max(0, score),
            details=f"Analyzed {len(lines)} lines",
            issues=issues,
            suggestions=suggestions,
        )


class PerformanceAnalyzer:
    """Analyze code performance characteristics."""

    def analyze(self, code: str, language: str = "python") -> DimensionScore:
        issues = []
        suggestions = []
        score = 100.0

        if language == "python":
            try:
                tree = ast.parse(code)
                score, issues, suggestions = self._analyze_python_perf(tree, code)
            except SyntaxError:
                pass

        return DimensionScore(
            dimension=QualityDimension.PERFORMANCE,
            score=max(0, score),
            details="Complexity and performance pattern analysis",
            issues=issues,
            suggestions=suggestions,
        )

    def _analyze_python_perf(
        self, tree: ast.AST, code: str
    ) -> Tuple[float, List[str], List[str]]:
        score = 100.0
        issues = []
        suggestions = []

        for node in ast.walk(tree):
            # Nested loops (O(n²) or worse)
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if child is not node and isinstance(child, (ast.For, ast.While)):
                        issues.append("Nested loop detected — potential O(n²) complexity")
                        score -= 10
                        suggestions.append(
                            "Consider dict/set lookups or algorithmic optimization"
                        )
                        break

            # String concatenation in loop
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if (isinstance(child, ast.AugAssign)
                            and isinstance(child.op, ast.Add)):
                        if isinstance(child.target, ast.Name):
                            issues.append(
                                "String concatenation in loop — use list join"
                            )
                            score -= 5
                            break

            # Recursive function without memoization hint
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                for child in ast.walk(node):
                    if (isinstance(child, ast.Call)
                            and isinstance(child.func, ast.Name)
                            and child.func.id == func_name):
                        # Check for @lru_cache or @cache decorator
                        has_cache = any(
                            isinstance(d, ast.Name) and d.id in ("cache", "lru_cache")
                            or isinstance(d, ast.Attribute) and d.attr in ("cache", "lru_cache")
                            for d in node.decorator_list
                        )
                        if not has_cache:
                            issues.append(
                                f"Recursive function '{func_name}' without memoization"
                            )
                            score -= 5
                            suggestions.append(
                                f"Consider adding @lru_cache to '{func_name}'"
                            )
                        break

        # Global analysis: large file
        lines = code.split("\n")
        if len(lines) > 500:
            score -= 5
            issues.append(f"Large file ({len(lines)} lines) — consider splitting")

        return max(0, score), issues, suggestions


class SecurityAnalyzer:
    """Analyze code for common security vulnerabilities."""

    DANGEROUS_PATTERNS = [
        (r'eval\s*\(', "Use of eval() — code injection risk"),
        (r'exec\s*\(', "Use of exec() — code injection risk"),
        (r'__import__\s*\(', "Dynamic import — could load malicious modules"),
        (r'subprocess\.call\s*\(.*shell\s*=\s*True', "Shell injection via subprocess"),
        (r'os\.system\s*\(', "Shell command execution — injection risk"),
        (r'pickle\.loads?\s*\(', "Pickle deserialization — arbitrary code execution"),
        (r'yaml\.load\s*\([^)]*\)', "Unsafe YAML load — use safe_load instead"),
        (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected"),
        (r'(api_key|secret|token)\s*=\s*["\'][^"\']+["\']', "Hardcoded secret detected"),
        (r'verify\s*=\s*False', "SSL verification disabled"),
    ]

    def analyze(self, code: str, language: str = "python") -> DimensionScore:
        issues = []
        suggestions = []
        score = 100.0

        for pattern, description in self.DANGEROUS_PATTERNS:
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                count = len(matches)
                severity = 15 if "injection" in description.lower() else 10
                score -= severity * count
                issues.append(f"{description} ({count} occurrence{'s' if count > 1 else ''})")
                suggestions.append(f"Remove or fix: {description}")

        # Check for SQL injection patterns
        sql_patterns = [
            r'f["\'].*(?:SELECT|INSERT|UPDATE|DELETE).*\{',
            r'\.format\(.*(?:SELECT|INSERT|UPDATE|DELETE)',
            r'%\s*(?:SELECT|INSERT|UPDATE|DELETE)',
        ]
        for pattern in sql_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                score -= 20
                issues.append("Potential SQL injection — use parameterized queries")
                suggestions.append("Use SQLAlchemy ORM or parameterized queries")
                break

        return DimensionScore(
            dimension=QualityDimension.SECURITY,
            score=max(0, score),
            details="Security vulnerability pattern analysis",
            issues=issues,
            suggestions=suggestions,
        )


# ── Main Evaluator ──

class SelfEvaluator:
    """
    Agent output quality evaluator.

    Scores generated code across multiple dimensions and provides
    actionable feedback for self-improvement iteration.
    """

    DEFAULT_WEIGHTS = {
        QualityDimension.CORRECTNESS: 0.30,
        QualityDimension.COMPLETENESS: 0.25,
        QualityDimension.STYLE: 0.15,
        QualityDimension.PERFORMANCE: 0.15,
        QualityDimension.SECURITY: 0.15,
    }

    VERDICT_THRESHOLDS = [
        (90, EvaluationVerdict.EXCELLENT),
        (75, EvaluationVerdict.GOOD),
        (60, EvaluationVerdict.ACCEPTABLE),
        (40, EvaluationVerdict.NEEDS_WORK),
        (0, EvaluationVerdict.REJECT),
    ]

    def __init__(
        self,
        *,
        iteration_threshold: float = 60.0,
        max_iterations: int = 5,
        weights: Optional[Dict[QualityDimension, float]] = None,
    ):
        self._iteration_threshold = iteration_threshold
        self._max_iterations = max_iterations
        self._weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._evaluation_count = 0
        self._total_score_sum = 0.0
        self._iteration_triggers = 0
        self._history: List[EvaluationResult] = []

        # Analyzers
        self._correctness = CorrectnessAnalyzer()
        self._completeness = CompletenessAnalyzer()
        self._style = StyleAnalyzer()
        self._performance = PerformanceAnalyzer()
        self._security = SecurityAnalyzer()

        logger.info(
            "SelfEvaluator initialized (threshold=%.0f, max_iter=%d)",
            iteration_threshold, max_iterations,
        )

    def evaluate(
        self,
        code: str,
        *,
        requirements: Optional[List[str]] = None,
        language: str = "python",
        iteration: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """
        Evaluate agent output quality.

        Args:
            code: The generated code to evaluate.
            requirements: Original requirements to check completeness against.
            language: Programming language.
            iteration: Current iteration number.
            metadata: Additional context.

        Returns:
            EvaluationResult with scores and iteration guidance.
        """
        start = time.perf_counter()

        # Run all analyzers
        scores: Dict[QualityDimension, DimensionScore] = {}
        scores[QualityDimension.CORRECTNESS] = self._correctness.analyze(code, language)
        scores[QualityDimension.COMPLETENESS] = self._completeness.analyze(
            code, requirements or [], language
        )
        scores[QualityDimension.STYLE] = self._style.analyze(code, language)
        scores[QualityDimension.PERFORMANCE] = self._performance.analyze(code, language)
        scores[QualityDimension.SECURITY] = self._security.analyze(code, language)

        # Compute weighted composite score
        composite = 0.0
        for dim, weight in self._weights.items():
            if dim in scores:
                composite += scores[dim].score * weight

        # Determine verdict
        verdict = EvaluationVerdict.REJECT
        for threshold, v in self.VERDICT_THRESHOLDS:
            if composite >= threshold:
                verdict = v
                break

        # Should iterate?
        should_iterate = (
            composite < self._iteration_threshold
            and iteration < self._max_iterations
        )

        # Build iteration guidance
        guidance = ""
        if should_iterate:
            worst_dims = sorted(
                scores.items(), key=lambda x: x[1].score
            )[:2]
            guidance_parts = []
            for dim, score in worst_dims:
                if score.issues:
                    guidance_parts.append(
                        f"{dim.value}: {'; '.join(score.issues[:2])}"
                    )
                if score.suggestions:
                    guidance_parts.append(
                        f"  → {'; '.join(score.suggestions[:2])}"
                    )
            guidance = "\n".join(guidance_parts)
            self._iteration_triggers += 1

        duration_ms = (time.perf_counter() - start) * 1000
        eval_id = hashlib.md5(
            f"{code[:100]}{time.time()}".encode()
        ).hexdigest()[:12]

        result = EvaluationResult(
            evaluation_id=eval_id,
            composite_score=composite,
            verdict=verdict,
            dimension_scores=scores,
            should_iterate=should_iterate,
            iteration_guidance=guidance,
            evaluation_duration_ms=duration_ms,
            metadata=metadata or {},
        )

        # Track history
        self._evaluation_count += 1
        self._total_score_sum += composite
        self._history.append(result)
        if len(self._history) > 100:
            self._history = self._history[-100:]

        logger.info(
            "Evaluation %s: score=%.1f verdict=%s iterate=%s (%.1fms)",
            eval_id, composite, verdict.value, should_iterate, duration_ms,
        )

        return result

    def get_improvement_trend(self, last_n: int = 10) -> Dict[str, Any]:
        """Analyze score improvement trend across recent evaluations."""
        recent = self._history[-last_n:] if self._history else []
        if len(recent) < 2:
            return {"trend": "insufficient_data", "evaluations": len(recent)}

        scores = [r.composite_score for r in recent]
        avg_first_half = sum(scores[:len(scores)//2]) / (len(scores)//2)
        avg_second_half = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
        delta = avg_second_half - avg_first_half

        return {
            "trend": "improving" if delta > 2 else "declining" if delta < -2 else "stable",
            "delta": round(delta, 1),
            "avg_score": round(sum(scores) / len(scores), 1),
            "min_score": round(min(scores), 1),
            "max_score": round(max(scores), 1),
            "evaluations": len(recent),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return evaluator stats for monitoring."""
        avg = (
            self._total_score_sum / self._evaluation_count
            if self._evaluation_count > 0 else 0
        )
        return {
            "total_evaluations": self._evaluation_count,
            "average_score": round(avg, 1),
            "iteration_triggers": self._iteration_triggers,
            "iteration_threshold": self._iteration_threshold,
            "max_iterations": self._max_iterations,
            "weights": {d.value: w for d, w in self._weights.items()},
            "trend": self.get_improvement_trend(),
        }
