"""
Architecture Validator - Check generated code against project conventions.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Violation:
    """Architecture violation."""

    rule: str
    message: str
    path: str
    line: Optional[int] = None
    severity: str = "warn"


@dataclass
class ValidationResult:
    """Result of architecture validation."""

    ok: bool
    violations: List[Violation] = field(default_factory=list)

    def add_violation(self, v: Violation) -> None:
        self.violations.append(v)
        self.ok = False


# Default conventions
DEFAULT_RULES = {
    "no_exec": True,
    "no_subprocess_shell": True,
    "no_eval": True,
    "no_absolute_imports": False,
    "max_line_length": 120,
    "require_type_hints": False,
}


class ArchitectureValidator:
    """
    Validates generated code against project conventions.
    """

    def __init__(self, rules: Optional[Dict[str, Any]] = None):
        self.rules = {**DEFAULT_RULES, **(rules or {})}

    def validate(
        self,
        files: List[Dict[str, Any]],
        mode: str = "warn",
    ) -> ValidationResult:
        """
        Validate generated files.
        Returns ValidationResult with any violations.
        """
        result = ValidationResult(ok=True)
        for f in files:
            path = str(f.get("path", "unknown"))
            content = f.get("content", "")
            if not isinstance(content, str):
                result.add_violation(
                    Violation("invalid_content", "content must be string", path, severity=mode)
                )
                continue
            for v in self._validate_file(path, content):
                result.add_violation(v)
        return result

    def _validate_file(self, path: str, content: str) -> List[Violation]:
        violations = []
        try:
            tree = ast.parse(content, filename=path)
        except SyntaxError as e:
            violations.append(
                Violation("syntax_error", str(e), path, line=e.lineno, severity="error")
            )
            return violations
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                v = self._check_call(node, path, content)
                if v:
                    violations.append(v)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    v = self._check_import(alias.name, path)
                    if v:
                        violations.append(v)
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                v = self._check_import(mod, path)
                if v:
                    violations.append(v)
        if self.rules.get("max_line_length"):
            for i, line in enumerate(content.split("\n"), 1):
                if len(line) > self.rules["max_line_length"]:
                    violations.append(
                        Violation(
                            "line_too_long",
                            f"Line exceeds {self.rules['max_line_length']} chars",
                            path,
                            line=i,
                        )
                    )
        return violations

    def _check_call(self, node: ast.Call, path: str, content: str) -> Optional[Violation]:
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
            if self.rules.get("no_exec") and name == "exec":
                return Violation("forbidden_call", "exec() is not allowed", path)
            if self.rules.get("no_eval") and name == "eval":
                return Violation("forbidden_call", "eval() is not allowed", path)
            if self.rules.get("no_subprocess_shell") and name == "shell" and (
                getattr(node.func.value, "attr", None) == "call"
                or (isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess")
            ):
                return Violation("forbidden_call", "subprocess with shell=True not allowed", path)
        if isinstance(node.func, ast.Name):
            if self.rules.get("no_exec") and node.func.id == "exec":
                return Violation("forbidden_call", "exec() is not allowed", path)
            if self.rules.get("no_eval") and node.func.id == "eval":
                return Violation("forbidden_call", "eval() is not allowed", path)
        return None

    def _check_import(self, module: str, path: str) -> Optional[Violation]:
        forbidden = ["os.system", "subprocess", "eval"]
        for fb in forbidden:
            if module.startswith(fb.replace(".", "")) or module == fb:
                if "subprocess" in module and self.rules.get("no_subprocess_shell"):
                    return Violation("forbidden_import", f"Import of {module} not allowed", path)
        return None
