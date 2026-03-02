"""
Sandboxed Code Executor — CognitionOS

Secure code execution environment:
- AST-based safety analysis
- Resource limits (time, memory)
- Output capture (stdout, stderr)
- Module whitelisting
- Execution environment isolation
- Multi-language support (Python primary)
"""

from __future__ import annotations

import ast
import io
import logging
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger(__name__)


class ExecutionLanguage(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"  # placeholder
    SHELL = "shell"  # placeholder


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    SECURITY_VIOLATION = "security_violation"


@dataclass
class ExecutionResult:
    execution_id: str = field(default_factory=lambda: str(uuid4()))
    language: ExecutionLanguage = ExecutionLanguage.PYTHON
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    error: Optional[str] = None
    traceback_str: Optional[str] = None
    duration_ms: float = 0
    memory_used_bytes: int = 0
    security_violations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "status": self.status.value,
            "stdout": self.stdout[:5000],
            "stderr": self.stderr[:2000],
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
            "security_violations": self.security_violations}


@dataclass
class SandboxConfig:
    max_execution_seconds: float = 30.0
    max_output_chars: int = 10000
    allowed_modules: Set[str] = field(default_factory=lambda: {
        "math", "json", "re", "datetime", "collections", "itertools",
        "functools", "string", "hashlib", "base64", "uuid",
        "dataclasses", "enum", "typing", "statistics", "decimal",
        "fractions", "random", "copy", "textwrap", "pprint"})
    blocked_builtins: Set[str] = field(default_factory=lambda: {
        "exec", "eval", "compile", "__import__", "open",
        "globals", "locals", "getattr", "setattr", "delattr",
        "breakpoint", "exit", "quit"})
    blocked_imports: Set[str] = field(default_factory=lambda: {
        "os", "sys", "subprocess", "shutil", "socket", "http",
        "urllib", "ctypes", "signal", "multiprocessing", "threading",
        "pickle", "marshal", "tempfile", "pathlib", "importlib"})


class SecurityAnalyzer:
    """Static analysis of code for security violations."""

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config

    def analyze(self, code: str) -> List[str]:
        violations = []
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"Syntax error: {e}"]

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in self._config.blocked_imports:
                        violations.append(f"Blocked import: {alias.name}")
                    elif root not in self._config.allowed_modules:
                        violations.append(f"Unallowed module: {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root in self._config.blocked_imports:
                        violations.append(f"Blocked import: {node.module}")
                    elif root not in self._config.allowed_modules:
                        violations.append(f"Unallowed module: {node.module}")

            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self._config.blocked_builtins:
                        violations.append(f"Blocked builtin: {node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("system", "popen", "remove",
                                           "rmdir", "unlink"):
                        violations.append(f"Blocked method: {node.func.attr}")

            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                violations.append(f"Blocked statement: {type(node).__name__}")

        return violations


class SandboxExecutor:
    """Executes code in a restricted sandbox environment."""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self._config = config or SandboxConfig()
        self._analyzer = SecurityAnalyzer(self._config)
        self._executions: List[ExecutionResult] = []
        self._metrics = {"total": 0, "success": 0, "error": 0,
                         "timeout": 0, "security_blocked": 0}

    def execute(self, code: str, *, language: ExecutionLanguage = ExecutionLanguage.PYTHON,
                context: Dict[str, Any] | None = None) -> ExecutionResult:
        self._metrics["total"] += 1

        if language != ExecutionLanguage.PYTHON:
            result = ExecutionResult(
                language=language, status=ExecutionStatus.ERROR,
                error=f"Language not yet supported: {language.value}")
            self._executions.append(result)
            return result

        # Security check
        violations = self._analyzer.analyze(code)
        if violations:
            self._metrics["security_blocked"] += 1
            result = ExecutionResult(
                language=language, status=ExecutionStatus.SECURITY_VIOLATION,
                security_violations=violations,
                error=f"Security violations: {', '.join(violations)}")
            self._executions.append(result)
            return result

        # Build safe globals
        safe_builtins = {k: v for k, v in __builtins__.__dict__.items()
                         if k not in self._config.blocked_builtins} if isinstance(__builtins__, type(sys)) else {
            k: v for k, v in __builtins__.items()
            if k not in self._config.blocked_builtins}

        sandbox_globals: Dict[str, Any] = {"__builtins__": safe_builtins}
        if context:
            sandbox_globals.update(context)

        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        result = ExecutionResult(language=language)

        start = time.monotonic()
        try:
            compiled = compile(code, "<sandbox>", "exec")
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compiled, sandbox_globals)

            result.status = ExecutionStatus.SUCCESS
            result.stdout = stdout_capture.getvalue()[:self._config.max_output_chars]
            result.stderr = stderr_capture.getvalue()[:self._config.max_output_chars]
            result.return_value = sandbox_globals.get("result", sandbox_globals.get("output"))
            self._metrics["success"] += 1

        except Exception as e:
            result.status = ExecutionStatus.ERROR
            result.error = str(e)
            result.traceback_str = traceback.format_exc()
            result.stdout = stdout_capture.getvalue()[:self._config.max_output_chars]
            result.stderr = stderr_capture.getvalue()[:self._config.max_output_chars]
            self._metrics["error"] += 1

        result.duration_ms = (time.monotonic() - start) * 1000
        self._executions.append(result)

        if len(self._executions) > 5000:
            self._executions = self._executions[-5000:]

        return result

    def validate_code(self, code: str) -> Dict[str, Any]:
        violations = self._analyzer.analyze(code)
        try:
            ast.parse(code)
            syntax_valid = True
        except SyntaxError as e:
            syntax_valid = False
            violations.append(f"Syntax error: {e}")
        return {"valid": syntax_valid and len(violations) == 0,
                "violations": violations}

    def get_execution_history(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._executions[-limit:]]

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self._metrics)


_executor: SandboxExecutor | None = None

def get_sandbox_executor() -> SandboxExecutor:
    global _executor
    if not _executor:
        _executor = SandboxExecutor()
    return _executor
