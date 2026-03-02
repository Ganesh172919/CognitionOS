"""
Validation Pipeline — CognitionOS AI Intelligence Layer

Multi-stage code validation pipeline:
1. Syntax Check — AST parsing
2. Type Analysis — Basic type hint validation
3. Lint Check — Code quality patterns
4. Security Scan — Vulnerability pattern detection
5. Architecture Rules — Convention enforcement
6. Complexity Analysis — Cyclomatic/cognitive complexity
"""

from __future__ import annotations

import ast
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationStage(str, Enum):
    SYNTAX = "syntax"
    TYPE_CHECK = "type_check"
    LINT = "lint"
    SECURITY = "security"
    ARCHITECTURE = "architecture"
    COMPLEXITY = "complexity"


@dataclass
class ValidationIssue:
    stage: ValidationStage
    severity: ValidationSeverity
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    rule_id: str = ""
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "stage": self.stage.value, "severity": self.severity.value,
            "message": self.message,
        }
        if self.line:
            result["line"] = self.line
        if self.rule_id:
            result["rule_id"] = self.rule_id
        if self.suggestion:
            result["suggestion"] = self.suggestion
        return result


@dataclass
class StageResult:
    stage: ValidationStage
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    duration_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value, "passed": self.passed,
            "errors": self.error_count, "warnings": self.warning_count,
            "duration_ms": round(self.duration_ms, 1),
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass
class PipelineResult:
    passed: bool
    stages: List[StageResult] = field(default_factory=list)
    total_errors: int = 0
    total_warnings: int = 0
    total_duration_ms: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "stages": [s.to_dict() for s in self.stages],
        }


class ValidationPipeline:
    """
    Multi-stage validation pipeline for agent-generated code.
    Runs each stage in order; optionally fails fast on first error.
    """

    def __init__(self, *, fail_fast: bool = False,
                 max_line_length: int = 120, max_complexity: int = 15,
                 max_function_length: int = 80):
        self._fail_fast = fail_fast
        self._max_line_length = max_line_length
        self._max_complexity = max_complexity
        self._max_function_length = max_function_length
        self._custom_rules: List[Callable[[str], List[ValidationIssue]]] = []
        self._validations_run = 0
        self._total_errors_found = 0
        logger.info("ValidationPipeline initialized (fail_fast=%s)", fail_fast)

    def add_custom_rule(self, rule: Callable[[str], List[ValidationIssue]]):
        self._custom_rules.append(rule)

    def validate(self, code: str, *, language: str = "python") -> PipelineResult:
        """Run the full validation pipeline on the given code."""
        start = time.perf_counter()
        stages: List[StageResult] = []

        pipeline_stages = [
            (ValidationStage.SYNTAX, self._check_syntax),
            (ValidationStage.TYPE_CHECK, self._check_types),
            (ValidationStage.LINT, self._check_lint),
            (ValidationStage.SECURITY, self._check_security),
            (ValidationStage.ARCHITECTURE, self._check_architecture),
            (ValidationStage.COMPLEXITY, self._check_complexity),
        ]

        all_passed = True
        for stage_name, checker in pipeline_stages:
            stage_start = time.perf_counter()
            issues = checker(code, language)
            duration = (time.perf_counter() - stage_start) * 1000

            has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
            result = StageResult(
                stage=stage_name, passed=not has_errors,
                issues=issues, duration_ms=duration,
            )
            stages.append(result)

            if has_errors:
                all_passed = False
                if self._fail_fast:
                    break

        total_errors = sum(s.error_count for s in stages)
        total_warnings = sum(s.warning_count for s in stages)
        total_duration = (time.perf_counter() - start) * 1000

        self._validations_run += 1
        self._total_errors_found += total_errors

        return PipelineResult(
            passed=all_passed, stages=stages,
            total_errors=total_errors, total_warnings=total_warnings,
            total_duration_ms=total_duration,
        )

    def _check_syntax(self, code: str, language: str) -> List[ValidationIssue]:
        issues = []
        if language != "python":
            return issues
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(ValidationIssue(
                stage=ValidationStage.SYNTAX, severity=ValidationSeverity.ERROR,
                message=f"Syntax error: {e.msg}", line=e.lineno,
                column=e.offset, rule_id="E001",
            ))
        return issues

    def _check_types(self, code: str, language: str) -> List[ValidationIssue]:
        issues = []
        if language != "python":
            return issues
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.returns and not node.name.startswith("_"):
                    issues.append(ValidationIssue(
                        stage=ValidationStage.TYPE_CHECK,
                        severity=ValidationSeverity.WARNING,
                        message=f"Function '{node.name}' missing return type annotation",
                        line=node.lineno, rule_id="T001",
                        suggestion="Add return type annotation",
                    ))
                untyped = sum(1 for a in node.args.args
                              if a.annotation is None and a.arg != "self")
                if untyped > 0:
                    issues.append(ValidationIssue(
                        stage=ValidationStage.TYPE_CHECK,
                        severity=ValidationSeverity.INFO,
                        message=f"Function '{node.name}' has {untyped} untyped parameter(s)",
                        line=node.lineno, rule_id="T002",
                    ))
        return issues

    def _check_lint(self, code: str, language: str) -> List[ValidationIssue]:
        issues = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            if len(line.rstrip()) > self._max_line_length:
                issues.append(ValidationIssue(
                    stage=ValidationStage.LINT, severity=ValidationSeverity.WARNING,
                    message=f"Line too long ({len(line.rstrip())} > {self._max_line_length})",
                    line=i, rule_id="L001",
                ))
            if line.rstrip() != line.rstrip("\n") and line.endswith(" "):
                issues.append(ValidationIssue(
                    stage=ValidationStage.LINT, severity=ValidationSeverity.INFO,
                    message="Trailing whitespace", line=i, rule_id="L002",
                ))

        if language == "python":
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ExceptHandler) and node.type is None:
                        issues.append(ValidationIssue(
                            stage=ValidationStage.LINT, severity=ValidationSeverity.WARNING,
                            message="Bare except clause", line=node.lineno,
                            rule_id="L003", suggestion="Catch specific exceptions",
                        ))
                    if isinstance(node, ast.Global):
                        issues.append(ValidationIssue(
                            stage=ValidationStage.LINT, severity=ValidationSeverity.WARNING,
                            message="Global statement used", line=node.lineno,
                            rule_id="L004", suggestion="Use dependency injection",
                        ))
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name == "*":
                                issues.append(ValidationIssue(
                                    stage=ValidationStage.LINT,
                                    severity=ValidationSeverity.ERROR,
                                    message="Wildcard import", line=node.lineno,
                                    rule_id="L005",
                                ))
            except SyntaxError:
                pass
        return issues

    def _check_security(self, code: str, language: str) -> List[ValidationIssue]:
        issues = []
        patterns = [
            (r'\beval\s*\(', "E", "S001", "Use of eval() — code injection risk"),
            (r'\bexec\s*\(', "E", "S002", "Use of exec() — code injection risk"),
            (r'\bos\.system\s*\(', "E", "S003", "os.system() — use subprocess with shell=False"),
            (r'pickle\.loads?\s*\(', "W", "S004", "Pickle deserialization risk"),
            (r'verify\s*=\s*False', "W", "S005", "SSL verification disabled"),
            (r'(password|secret|api_key)\s*=\s*["\'][^"\']{3,}["\']', "E", "S006",
             "Hardcoded credential detected"),
        ]
        for pattern, severity, rule_id, msg in patterns:
            for match in re.finditer(pattern, code, re.IGNORECASE):
                line_no = code[:match.start()].count("\n") + 1
                issues.append(ValidationIssue(
                    stage=ValidationStage.SECURITY,
                    severity=ValidationSeverity.ERROR if severity == "E" else ValidationSeverity.WARNING,
                    message=msg, line=line_no, rule_id=rule_id,
                ))
        return issues

    def _check_architecture(self, code: str, language: str) -> List[ValidationIssue]:
        issues = []
        if language != "python":
            return issues
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues

        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        for cls in classes:
            methods = [n for n in ast.walk(cls)
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if len(methods) > 25:
                issues.append(ValidationIssue(
                    stage=ValidationStage.ARCHITECTURE,
                    severity=ValidationSeverity.WARNING,
                    message=f"Class '{cls.name}' has {len(methods)} methods (consider splitting)",
                    line=cls.lineno, rule_id="A001",
                ))

        functions = [n for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for func in functions:
            body_lines = (func.end_lineno or func.lineno) - func.lineno
            if body_lines > self._max_function_length:
                issues.append(ValidationIssue(
                    stage=ValidationStage.ARCHITECTURE,
                    severity=ValidationSeverity.WARNING,
                    message=f"Function '{func.name}' is {body_lines} lines (max {self._max_function_length})",
                    line=func.lineno, rule_id="A002",
                ))

        return issues

    def _check_complexity(self, code: str, language: str) -> List[ValidationIssue]:
        issues = []
        if language != "python":
            return issues
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = self._calculate_complexity(node)
                if complexity > self._max_complexity:
                    issues.append(ValidationIssue(
                        stage=ValidationStage.COMPLEXITY,
                        severity=ValidationSeverity.WARNING,
                        message=f"Function '{node.name}' complexity={complexity} (max {self._max_complexity})",
                        line=node.lineno, rule_id="C001",
                        suggestion="Break into smaller functions",
                    ))
        return issues

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For,
                                  ast.ExceptHandler, ast.With,
                                  ast.Assert, ast.comprehension)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def get_stats(self) -> Dict[str, Any]:
        return {
            "validations_run": self._validations_run,
            "total_errors_found": self._total_errors_found,
            "fail_fast": self._fail_fast,
            "custom_rules": len(self._custom_rules),
        }
