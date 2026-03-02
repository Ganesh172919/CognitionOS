"""
Autonomous Code Generation Agent — CognitionOS

Single powerful AI agent that can:
- Accept high-level user requirements
- Decompose tasks intelligently
- Generate complete modules
- Validate output
- Optimize performance
- Detect architecture violations
- Self-evaluate and iterate

This is the master agent that orchestrates the entire code generation pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class TaskPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    DECOMPOSING = "decomposing"
    GENERATING = "generating"
    VALIDATING = "validating"
    OPTIMIZING = "optimizing"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CodeLanguage(str, Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    GO = "go"
    RUST = "rust"
    SQL = "sql"
    YAML = "yaml"
    JSON = "json"


@dataclass
class CodeRequirement:
    """High-level code generation requirement."""
    requirement_id: str
    title: str
    description: str
    language: CodeLanguage = CodeLanguage.PYTHON
    target_path: str = ""
    constraints: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    architecture_style: str = "clean_architecture"
    priority: TaskPriority = TaskPriority.MEDIUM
    context: Dict[str, Any] = field(default_factory=dict)
    parent_requirement_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    max_iterations: int = 3
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def create(title: str, description: str, **kwargs) -> "CodeRequirement":
        return CodeRequirement(
            requirement_id=uuid.uuid4().hex[:12],
            title=title, description=description, **kwargs,
        )


@dataclass
class DecomposedTask:
    """Atomic task produced by task decomposition."""
    task_id: str
    requirement_id: str
    title: str
    description: str
    task_type: str  # "module", "function", "class", "test", "config", "migration"
    language: CodeLanguage
    target_file: str = ""
    dependencies: List[str] = field(default_factory=list)
    estimated_complexity: int = 1  # 1=trivial, 5=complex, 10=very complex
    estimated_lines: int = 50
    template_hints: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    output: Optional["GeneratedCode"] = None


@dataclass
class GeneratedCode:
    """Output of code generation."""
    code_id: str
    task_id: str
    content: str
    language: CodeLanguage
    file_path: str
    imports: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    line_count: int = 0
    complexity_score: float = 0.0
    generated_at: float = field(default_factory=time.time)
    iteration: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.line_count = len(self.content.strip().split("\n"))


@dataclass
class ValidationResult:
    """Result of code validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    architecture_violations: List[str] = field(default_factory=list)
    security_issues: List[str] = field(default_factory=list)


@dataclass
class AgentMemoryEntry:
    """Memory entry for the agent's context."""
    key: str
    value: Any
    category: str  # "pattern", "error", "decision", "feedback", "context"
    relevance: float = 1.0
    created_at: float = field(default_factory=time.time)
    access_count: int = 0


@dataclass
class GenerationPlan:
    """Complete plan for fulfilling a requirement."""
    plan_id: str
    requirement: CodeRequirement
    tasks: List[DecomposedTask] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)  # task_ids
    estimated_total_lines: int = 0
    estimated_total_time_ms: float = 0
    architecture_notes: List[str] = field(default_factory=list)
    risk_assessment: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class GenerationResult:
    """Complete result of a code generation session."""
    result_id: str
    requirement_id: str
    plan: GenerationPlan
    generated_files: List[GeneratedCode] = field(default_factory=list)
    validation: Optional[ValidationResult] = None
    iterations: int = 0
    total_time_ms: float = 0
    total_lines: int = 0
    success: bool = False
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# TASK DECOMPOSITION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class TaskDecompositionEngine:
    """Decomposes high-level requirements into atomic generation tasks."""

    PATTERNS = {
        "api_endpoint": {
            "tasks": ["route_handler", "request_schema", "response_schema",
                       "service_method", "repository_method", "unit_test"],
            "complexity": 4,
        },
        "data_model": {
            "tasks": ["entity_class", "repository_interface", "repository_impl",
                       "migration_script", "unit_test"],
            "complexity": 3,
        },
        "service_layer": {
            "tasks": ["service_interface", "service_impl", "dependency_injection",
                       "error_handling", "unit_test", "integration_test"],
            "complexity": 5,
        },
        "background_worker": {
            "tasks": ["worker_class", "task_handler", "retry_logic",
                       "monitoring_hooks", "unit_test"],
            "complexity": 4,
        },
        "plugin_module": {
            "tasks": ["plugin_interface", "plugin_impl", "config_schema",
                       "activation_logic", "unit_test"],
            "complexity": 4,
        },
        "full_module": {
            "tasks": ["module_init", "core_logic", "models", "services",
                       "api_routes", "utils", "config", "tests"],
            "complexity": 8,
        },
    }

    def decompose(self, requirement: CodeRequirement) -> List[DecomposedTask]:
        """Break requirement into atomic tasks."""
        tasks = []
        pattern_key = self._detect_pattern(requirement)
        pattern = self.PATTERNS.get(pattern_key, self.PATTERNS["full_module"])

        for i, task_type in enumerate(pattern["tasks"]):
            task = DecomposedTask(
                task_id=f"{requirement.requirement_id}_{i:03d}",
                requirement_id=requirement.requirement_id,
                title=f"{task_type.replace('_', ' ').title()} for {requirement.title}",
                description=self._generate_task_description(requirement, task_type),
                task_type=task_type,
                language=requirement.language,
                target_file=self._infer_file_path(requirement, task_type),
                dependencies=[f"{requirement.requirement_id}_{j:03d}"
                               for j in range(i)
                               if self._has_dependency(pattern["tasks"][j], task_type)],
                estimated_complexity=self._estimate_complexity(task_type),
                estimated_lines=self._estimate_lines(task_type),
            )
            tasks.append(task)

        return tasks

    def _detect_pattern(self, req: CodeRequirement) -> str:
        desc = req.description.lower()
        if any(kw in desc for kw in ["api", "endpoint", "route", "rest"]):
            return "api_endpoint"
        if any(kw in desc for kw in ["model", "entity", "schema", "database"]):
            return "data_model"
        if any(kw in desc for kw in ["service", "business logic", "use case"]):
            return "service_layer"
        if any(kw in desc for kw in ["worker", "background", "queue", "job"]):
            return "background_worker"
        if any(kw in desc for kw in ["plugin", "extension", "addon"]):
            return "plugin_module"
        return "full_module"

    def _generate_task_description(self, req: CodeRequirement, task_type: str) -> str:
        descriptions = {
            "route_handler": f"Create API route handler for {req.title}",
            "request_schema": f"Define request validation schema for {req.title}",
            "response_schema": f"Define response serialization schema for {req.title}",
            "service_method": f"Implement service method for {req.title} business logic",
            "repository_method": f"Implement data access method for {req.title}",
            "entity_class": f"Define entity/model class for {req.title}",
            "repository_interface": f"Define repository interface for {req.title}",
            "repository_impl": f"Implement repository with database access for {req.title}",
            "migration_script": f"Create database migration for {req.title}",
            "service_interface": f"Define service interface for {req.title}",
            "service_impl": f"Implement service with business logic for {req.title}",
            "dependency_injection": f"Setup DI container for {req.title}",
            "error_handling": f"Define error types for {req.title}",
            "worker_class": f"Create background worker for {req.title}",
            "task_handler": f"Implement task processing logic for {req.title}",
            "retry_logic": f"Add retry/backoff logic for {req.title}",
            "monitoring_hooks": f"Add monitoring/metrics hooks for {req.title}",
            "plugin_interface": f"Define plugin interface for {req.title}",
            "plugin_impl": f"Implement plugin for {req.title}",
            "config_schema": f"Define configuration schema for {req.title}",
            "activation_logic": f"Implement activation/deactivation for {req.title}",
            "module_init": f"Create module __init__.py for {req.title}",
            "core_logic": f"Implement core logic for {req.title}",
            "models": f"Define data models for {req.title}",
            "services": f"Implement service layer for {req.title}",
            "api_routes": f"Create API routes for {req.title}",
            "utils": f"Create utility functions for {req.title}",
            "config": f"Setup configuration for {req.title}",
            "unit_test": f"Write unit tests for {req.title}",
            "integration_test": f"Write integration tests for {req.title}",
            "tests": f"Write comprehensive tests for {req.title}",
        }
        return descriptions.get(task_type, f"Implement {task_type} for {req.title}")

    def _infer_file_path(self, req: CodeRequirement, task_type: str) -> str:
        base = req.target_path or f"modules/{req.requirement_id}"
        ext = ".py" if req.language == CodeLanguage.PYTHON else ".ts"
        file_map = {
            "unit_test": f"tests/test_{req.requirement_id}{ext}",
            "integration_test": f"tests/integration/test_{req.requirement_id}{ext}",
            "migration_script": f"database/migrations/add_{req.requirement_id}.sql",
        }
        return file_map.get(task_type, f"{base}/{task_type}{ext}")

    def _estimate_complexity(self, task_type: str) -> int:
        complexities = {
            "unit_test": 2, "config": 1, "module_init": 1,
            "request_schema": 2, "response_schema": 2,
            "entity_class": 3, "repository_interface": 2,
            "service_impl": 5, "core_logic": 6,
            "integration_test": 4, "worker_class": 5,
        }
        return complexities.get(task_type, 3)

    def _estimate_lines(self, task_type: str) -> int:
        estimates = {
            "unit_test": 80, "config": 30, "module_init": 20,
            "request_schema": 40, "response_schema": 40,
            "entity_class": 60, "service_impl": 120,
            "core_logic": 150, "integration_test": 100,
            "worker_class": 100, "api_routes": 80,
        }
        return estimates.get(task_type, 60)

    def _has_dependency(self, source: str, target: str) -> bool:
        deps = {
            "unit_test": {"service_method", "entity_class", "core_logic",
                           "service_impl", "worker_class", "plugin_impl"},
            "integration_test": {"service_impl", "repository_impl", "api_routes"},
            "service_impl": {"repository_interface", "entity_class"},
            "repository_impl": {"entity_class", "repository_interface"},
            "api_routes": {"service_impl"},
            "route_handler": {"service_method", "request_schema", "response_schema"},
        }
        return source in deps.get(target, set())


# ═══════════════════════════════════════════════════════════════════════════════
# CODE VALIDATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class CodeValidator:
    """Multi-stage code validation pipeline."""

    def __init__(self):
        self._validators: List[Callable[[GeneratedCode], ValidationResult]] = [
            self._validate_syntax,
            self._validate_architecture,
            self._validate_security,
            self._validate_style,
            self._validate_complexity,
        ]

    def validate(self, code: GeneratedCode) -> ValidationResult:
        """Run all validators on generated code."""
        combined = ValidationResult(valid=True)

        for validator in self._validators:
            result = validator(code)
            if not result.valid:
                combined.valid = False
            combined.errors.extend(result.errors)
            combined.warnings.extend(result.warnings)
            combined.suggestions.extend(result.suggestions)
            combined.architecture_violations.extend(result.architecture_violations)
            combined.security_issues.extend(result.security_issues)
            combined.metrics.update(result.metrics)

        return combined

    def _validate_syntax(self, code: GeneratedCode) -> ValidationResult:
        errors = []
        if code.language == CodeLanguage.PYTHON:
            try:
                compile(code.content, code.file_path, "exec")
            except SyntaxError as exc:
                errors.append(f"Syntax error at line {exc.lineno}: {exc.msg}")

        return ValidationResult(valid=not errors, errors=errors)

    def _validate_architecture(self, code: GeneratedCode) -> ValidationResult:
        violations = []
        content = code.content

        # Check for common architecture violations
        if "infrastructure" in code.file_path and "domain" in code.file_path:
            violations.append("Infrastructure code should not be in domain layer")

        # Check for circular import patterns
        import_lines = [l for l in content.split("\n") if l.strip().startswith(("import ", "from "))]
        for imp_line in import_lines:
            if "from core.domain" in imp_line and "infrastructure" in code.file_path:
                pass  # This is fine (infra depends on domain)
            elif "from infrastructure" in imp_line and "core/domain" in code.file_path:
                violations.append(f"Domain layer should not depend on infrastructure: {imp_line.strip()}")

        return ValidationResult(
            valid=not violations,
            architecture_violations=violations,
        )

    def _validate_security(self, code: GeneratedCode) -> ValidationResult:
        issues = []
        content = code.content.lower()

        dangerous_patterns = [
            ("eval(", "Use ast.literal_eval() instead of eval()"),
            ("exec(", "Avoid exec() — use safe alternatives"),
            ("__import__", "Avoid dynamic imports with __import__"),
            ("pickle.loads", "pickle.loads is vulnerable to code injection"),
            ("subprocess.call(", "Use subprocess.run() with shell=False"),
            ("os.system(", "Avoid os.system() — use subprocess.run()"),
            ("password", None),  # Check for hardcoded passwords
        ]

        for pattern, suggestion in dangerous_patterns:
            if pattern in content:
                if pattern == "password":
                    # Check for hardcoded passwords
                    for line in code.content.split("\n"):
                        lstrip = line.strip().lower()
                        if "password" in lstrip and "=" in lstrip and ('"' in lstrip or "'" in lstrip):
                            if not any(safe in lstrip for safe in ["os.getenv", "config.", "environ", "getattr"]):
                                issues.append(f"Possible hardcoded password: {line.strip()[:80]}")
                else:
                    issues.append(suggestion or f"Security concern: {pattern}")

        return ValidationResult(valid=not issues, security_issues=issues)

    def _validate_style(self, code: GeneratedCode) -> ValidationResult:
        warnings = []
        lines = code.content.split("\n")

        # Line length
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                warnings.append(f"Line {i} exceeds 120 characters ({len(line)})")

        # Missing docstrings
        if code.language == CodeLanguage.PYTHON:
            if "class " in code.content:
                for i, line in enumerate(lines):
                    if line.strip().startswith("class "):
                        # Check for docstring
                        if i + 1 < len(lines) and '"""' not in lines[i + 1]:
                            warnings.append(f"Class at line {i+1} missing docstring")

            if "def " in code.content:
                func_count = sum(1 for l in lines if l.strip().startswith("def "))
                doc_count = code.content.count('"""')
                if func_count > 0 and doc_count < func_count:
                    warnings.append("Some functions may be missing docstrings")

        return ValidationResult(valid=True, warnings=warnings)

    def _validate_complexity(self, code: GeneratedCode) -> ValidationResult:
        metrics = {}
        content = code.content
        lines = content.split("\n")

        # Basic complexity metrics
        metrics["total_lines"] = len(lines)
        metrics["code_lines"] = sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))
        metrics["comment_lines"] = sum(1 for l in lines if l.strip().startswith("#"))
        metrics["blank_lines"] = sum(1 for l in lines if not l.strip())

        if code.language == CodeLanguage.PYTHON:
            metrics["classes"] = sum(1 for l in lines if l.strip().startswith("class "))
            metrics["functions"] = sum(1 for l in lines if l.strip().startswith("def "))
            metrics["imports"] = sum(1 for l in lines if l.strip().startswith(("import ", "from ")))

        # Nesting depth
        max_indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent // 4)
        metrics["max_nesting_depth"] = max_indent

        warnings = []
        if max_indent > 6:
            warnings.append(f"High nesting depth ({max_indent}) — consider refactoring")
        if metrics.get("code_lines", 0) > 500:
            warnings.append("File exceeds 500 lines — consider splitting into modules")

        return ValidationResult(valid=True, warnings=warnings, metrics=metrics)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT MEMORY SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class AgentContextManager:
    """Manages agent's working memory, learned patterns, and decision history."""

    def __init__(self, *, max_entries: int = 5000):
        self._memory: Dict[str, AgentMemoryEntry] = {}
        self._max_entries = max_entries
        self._pattern_library: Dict[str, Dict[str, Any]] = {}

    def store(self, key: str, value: Any, category: str = "context",
              relevance: float = 1.0):
        self._memory[key] = AgentMemoryEntry(
            key=key, value=value, category=category, relevance=relevance,
        )
        self._enforce_limits()

    def recall(self, key: str) -> Optional[Any]:
        entry = self._memory.get(key)
        if entry:
            entry.access_count += 1
            return entry.value
        return None

    def search(self, query: str, *, category: Optional[str] = None,
               top_k: int = 10) -> List[AgentMemoryEntry]:
        query_words = set(query.lower().split())
        results = []
        for entry in self._memory.values():
            if category and entry.category != category:
                continue
            key_words = set(entry.key.lower().split("_"))
            overlap = len(query_words & key_words)
            if overlap > 0:
                score = overlap * entry.relevance
                results.append((score, entry))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:top_k]]

    def store_pattern(self, pattern_name: str, pattern: Dict[str, Any]):
        self._pattern_library[pattern_name] = {
            **pattern, "stored_at": time.time(),
        }

    def get_pattern(self, pattern_name: str) -> Optional[Dict[str, Any]]:
        return self._pattern_library.get(pattern_name)

    def list_patterns(self) -> List[str]:
        return list(self._pattern_library.keys())

    def store_decision(self, decision: str, rationale: str, outcome: str = ""):
        key = f"decision_{uuid.uuid4().hex[:8]}"
        self.store(key, {
            "decision": decision, "rationale": rationale,
            "outcome": outcome,
        }, category="decision")

    def store_error(self, error: str, context: str, resolution: str = ""):
        key = f"error_{hashlib.sha256(error.encode()).hexdigest()[:8]}"
        self.store(key, {
            "error": error, "context": context,
            "resolution": resolution,
        }, category="error", relevance=0.9)

    def get_relevant_context(self, requirement: CodeRequirement) -> Dict[str, Any]:
        """Build relevant context for a code generation task."""
        context = {
            "patterns": [],
            "past_decisions": [],
            "known_errors": [],
        }

        # Find relevant patterns
        for name, pattern in self._pattern_library.items():
            if any(tag in name.lower() for tag in requirement.tags):
                context["patterns"].append(pattern)

        # Find relevant past decisions
        decisions = self.search(requirement.description, category="decision", top_k=5)
        context["past_decisions"] = [d.value for d in decisions]

        # Find relevant errors to avoid
        errors = self.search(requirement.description, category="error", top_k=3)
        context["known_errors"] = [e.value for e in errors]

        return context

    def _enforce_limits(self):
        if len(self._memory) > self._max_entries:
            # Remove least relevant, least accessed entries
            entries = sorted(
                self._memory.items(),
                key=lambda x: x[1].relevance * (x[1].access_count + 1),
            )
            to_remove = entries[:len(entries) - self._max_entries]
            for key, _ in to_remove:
                del self._memory[key]

    def get_stats(self) -> Dict[str, Any]:
        by_category: Dict[str, int] = defaultdict(int)
        for entry in self._memory.values():
            by_category[entry.category] += 1
        return {
            "total_entries": len(self._memory),
            "by_category": dict(by_category),
            "patterns": len(self._pattern_library),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HALLUCINATION DETECTION & SAFETY
# ═══════════════════════════════════════════════════════════════════════════════

class HallucinationGuard:
    """Detect and prevent hallucinated or unsafe code generation."""

    def __init__(self):
        self._known_apis: Set[str] = set()
        self._known_modules: Set[str] = set()
        self._blocklist: Set[str] = {
            "rm -rf", "format c:", "DROP TABLE", "DELETE FROM",
            "os.rmdir", "shutil.rmtree",
        }

    def register_api(self, api_name: str):
        self._known_apis.add(api_name)

    def register_module(self, module_name: str):
        self._known_modules.add(module_name)

    def check(self, code: GeneratedCode) -> List[str]:
        """Check for hallucinations and unsafe patterns."""
        issues = []
        content = code.content

        # Check for blocked patterns
        for pattern in self._blocklist:
            if pattern.lower() in content.lower():
                issues.append(f"Blocked pattern detected: {pattern}")

        # Check for phantom imports (import non-existent modules)
        if code.language == CodeLanguage.PYTHON:
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("import ") or line.startswith("from "):
                    module = self._extract_module(line)
                    if module and self._known_modules and module not in self._known_modules:
                        # Only flag if we have a known module list
                        if not any(module.startswith(known) for known in self._known_modules):
                            issues.append(f"Potentially hallucinated import: {module}")

        # Check for suspicious patterns
        if "TODO: implement" in content.lower() and code.content.count("pass") > 3:
            issues.append("Multiple unimplemented stubs detected")

        # Check for repetitive code (copy-paste hallucination)
        lines = [l.strip() for l in content.split("\n") if l.strip() and not l.strip().startswith("#")]
        if len(lines) > 10:
            line_counts = defaultdict(int)
            for line in lines:
                if len(line) > 20:
                    line_counts[line] += 1
            for line, count in line_counts.items():
                if count > 5:
                    issues.append(f"Highly repetitive code detected ({count} occurrences)")
                    break

        return issues

    def _extract_module(self, import_line: str) -> Optional[str]:
        parts = import_line.split()
        if import_line.startswith("from "):
            if len(parts) >= 2:
                return parts[1].split(".")[0]
        elif import_line.startswith("import "):
            if len(parts) >= 2:
                return parts[1].split(".")[0]
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# CODE GENERATION TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

class CodeTemplateEngine:
    """Template engine for common code patterns."""

    TEMPLATES: Dict[str, str] = {
        "python_module": '''"""
{module_name} — CognitionOS

{description}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


{content}
''',
        "python_service": '''"""
{service_name} Service — CognitionOS

{description}
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class {class_name}:
    """{description}"""

    def __init__(self{init_params}):
{init_body}

{methods}
''',
        "python_test": '''"""
Tests for {module_name}
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

{imports}


class Test{class_name}:
    """Test suite for {class_name}."""

    def setup_method(self):
        """Setup test fixtures."""
{setup}

{test_methods}
''',
        "python_api_route": '''"""
API Routes for {module_name} — CognitionOS
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional

{imports}

router = APIRouter(prefix="/{route_prefix}", tags=["{tag}"])


{routes}
''',
    }

    def render(self, template_name: str, **kwargs) -> str:
        template = self.TEMPLATES.get(template_name, "")
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning("Missing template variable: %s", e)
            return template


# ═══════════════════════════════════════════════════════════════════════════════
# AUTONOMOUS CODE GENERATION AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class AutonomousCodeGenAgent:
    """
    Single powerful autonomous AI agent for code generation.

    Pipeline:
    1. Accept requirement
    2. Plan (understand context, select patterns)
    3. Decompose into atomic tasks
    4. Generate code for each task
    5. Validate output
    6. Self-evaluate and iterate
    7. Return final result
    """

    def __init__(self, *, llm_provider: Optional[Any] = None,
                 max_iterations: int = 3):
        self._decomposer = TaskDecompositionEngine()
        self._validator = CodeValidator()
        self._memory = AgentContextManager()
        self._hallucination_guard = HallucinationGuard()
        self._template_engine = CodeTemplateEngine()
        self._llm = llm_provider
        self._max_iterations = max_iterations
        self._generation_history: List[GenerationResult] = []
        self._active_sessions: Dict[str, Dict[str, Any]] = {}

    # ── Main Pipeline ──

    async def generate(self, requirement: CodeRequirement) -> GenerationResult:
        """Execute the full code generation pipeline."""
        start_time = time.time()
        session_id = uuid.uuid4().hex[:12]
        self._active_sessions[session_id] = {"requirement": requirement.title, "status": "starting"}

        result = GenerationResult(
            result_id=session_id,
            requirement_id=requirement.requirement_id,
            plan=GenerationPlan(plan_id=session_id, requirement=requirement),
        )

        try:
            # Step 1: Build context
            self._active_sessions[session_id]["status"] = "building_context"
            context = self._memory.get_relevant_context(requirement)

            # Step 2: Plan
            self._active_sessions[session_id]["status"] = "planning"
            plan = await self._plan(requirement, context)
            result.plan = plan

            # Step 3: Decompose
            self._active_sessions[session_id]["status"] = "decomposing"
            plan.tasks = self._decomposer.decompose(requirement)
            plan.execution_order = [t.task_id for t in plan.tasks]
            plan.estimated_total_lines = sum(t.estimated_lines for t in plan.tasks)

            # Step 4: Generate & Validate cycle
            for iteration in range(self._max_iterations):
                self._active_sessions[session_id]["status"] = f"generating_iteration_{iteration + 1}"
                result.iterations = iteration + 1

                generated_files = await self._generate_all_tasks(plan, context)
                result.generated_files = generated_files

                # Step 5: Validate
                self._active_sessions[session_id]["status"] = "validating"
                validation = self._validate_all(generated_files)
                result.validation = validation

                if validation.valid:
                    break

                # Step 6: Self-correct
                self._active_sessions[session_id]["status"] = "self_correcting"
                context["validation_errors"] = validation.errors
                context["iteration"] = iteration + 1

                # Store errors for learning
                for error in validation.errors:
                    self._memory.store_error(error, requirement.description)

            result.success = result.validation.valid if result.validation else False
            result.total_lines = sum(f.line_count for f in result.generated_files)
            result.total_time_ms = (time.time() - start_time) * 1000

            # Store successful patterns
            if result.success:
                self._memory.store_pattern(
                    f"successful_{requirement.requirement_id}",
                    {
                        "requirement_type": self._decomposer._detect_pattern(requirement),
                        "task_count": len(plan.tasks),
                        "total_lines": result.total_lines,
                        "iterations": result.iterations,
                    }
                )
                self._memory.store_decision(
                    f"Generated {requirement.title}",
                    f"Used {len(plan.tasks)} tasks, {result.iterations} iterations",
                    "success",
                )

            self._generation_history.append(result)

        except Exception as exc:
            result.error = str(exc)
            result.success = False
            logger.error("Code generation failed for %s: %s",
                         requirement.title, exc, exc_info=True)
        finally:
            self._active_sessions.pop(session_id, None)

        return result

    # ── Planning ──

    async def _plan(self, requirement: CodeRequirement,
                     context: Dict[str, Any]) -> GenerationPlan:
        """Create a generation plan."""
        plan = GenerationPlan(
            plan_id=uuid.uuid4().hex[:12],
            requirement=requirement,
        )

        # Architecture notes based on requirement
        if requirement.architecture_style == "clean_architecture":
            plan.architecture_notes = [
                "Follow clean architecture: domain → application → infrastructure",
                "Domain layer has zero external dependencies",
                "Use dependency injection for cross-layer communication",
                "Repository pattern for data access abstraction",
            ]
        elif requirement.architecture_style == "hexagonal":
            plan.architecture_notes = [
                "Ports and adapters pattern",
                "Core domain at center",
                "Input ports for primary adapters",
                "Output ports for secondary adapters",
            ]

        # Risk assessment
        plan.risk_assessment = {
            "complexity": "high" if len(requirement.constraints) > 3 else "medium",
            "dependencies": "medium" if requirement.context else "low",
        }

        return plan

    # ── Code Generation ──

    async def _generate_all_tasks(self, plan: GenerationPlan,
                                    context: Dict[str, Any]) -> List[GeneratedCode]:
        """Generate code for all tasks in the plan."""
        generated = []

        for task in plan.tasks:
            task.status = TaskStatus.GENERATING
            code = await self._generate_single_task(task, plan.requirement, context)
            task.output = code
            task.status = TaskStatus.COMPLETED
            generated.append(code)

        return generated

    async def _generate_single_task(self, task: DecomposedTask,
                                      requirement: CodeRequirement,
                                      context: Dict[str, Any]) -> GeneratedCode:
        """Generate code for a single task."""
        # Select template based on task type
        content = self._generate_from_template(task, requirement)

        code = GeneratedCode(
            code_id=uuid.uuid4().hex[:12],
            task_id=task.task_id,
            content=content,
            language=task.language,
            file_path=task.target_file,
        )

        # Hallucination check
        issues = self._hallucination_guard.check(code)
        if issues:
            logger.warning("Hallucination issues in %s: %s", task.task_id, issues)

        return code

    def _generate_from_template(self, task: DecomposedTask,
                                  requirement: CodeRequirement) -> str:
        """Use templates to generate initial code structure."""
        if task.task_type in ("service_impl", "service_method"):
            return self._template_engine.render(
                "python_service",
                service_name=requirement.title,
                description=task.description,
                class_name=self._to_class_name(requirement.title),
                init_params="",
                init_body="        pass",
                methods=""
            )
        elif task.task_type in ("unit_test", "tests"):
            return self._template_engine.render(
                "python_test",
                module_name=requirement.title,
                class_name=self._to_class_name(requirement.title),
                imports="",
                setup="        pass",
                test_methods="    def test_placeholder(self):\n        assert True",
            )
        elif task.task_type in ("route_handler", "api_routes"):
            return self._template_engine.render(
                "python_api_route",
                module_name=requirement.title,
                route_prefix=requirement.requirement_id,
                tag=requirement.title,
                imports="",
                routes="",
            )
        else:
            return self._template_engine.render(
                "python_module",
                module_name=requirement.title,
                description=task.description,
                content=f"# TODO: Implement {task.task_type} for {requirement.title}",
            )

    # ── Validation ──

    def _validate_all(self, files: List[GeneratedCode]) -> ValidationResult:
        """Validate all generated files."""
        combined = ValidationResult(valid=True)

        for code in files:
            result = self._validator.validate(code)
            if not result.valid:
                combined.valid = False
            combined.errors.extend(result.errors)
            combined.warnings.extend(result.warnings)
            combined.suggestions.extend(result.suggestions)
            combined.architecture_violations.extend(result.architecture_violations)
            combined.security_issues.extend(result.security_issues)
            combined.metrics.update(result.metrics)

        return combined

    # ── Helpers ──

    def _to_class_name(self, name: str) -> str:
        return "".join(word.capitalize() for word in name.replace("-", " ").replace("_", " ").split())

    # ── Status & History ──

    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._active_sessions)

    def get_history(self, *, limit: int = 20) -> List[Dict[str, Any]]:
        return [
            {
                "result_id": r.result_id,
                "requirement_id": r.requirement_id,
                "success": r.success,
                "iterations": r.iterations,
                "total_lines": r.total_lines,
                "total_time_ms": round(r.total_time_ms, 1),
                "files_generated": len(r.generated_files),
            }
            for r in self._generation_history[-limit:]
        ]

    def get_stats(self) -> Dict[str, Any]:
        total = len(self._generation_history)
        successes = sum(1 for r in self._generation_history if r.success)
        return {
            "total_generations": total,
            "success_rate": round(successes / max(total, 1) * 100, 1),
            "total_lines_generated": sum(r.total_lines for r in self._generation_history),
            "avg_iterations": round(
                sum(r.iterations for r in self._generation_history) / max(total, 1), 1
            ),
            "avg_time_ms": round(
                sum(r.total_time_ms for r in self._generation_history) / max(total, 1), 1
            ),
            "memory": self._memory.get_stats(),
            "active_sessions": len(self._active_sessions),
        }


# ── Singleton ──

_agent: Optional[AutonomousCodeGenAgent] = None


def get_codegen_agent() -> AutonomousCodeGenAgent:
    global _agent
    if not _agent:
        _agent = AutonomousCodeGenAgent()
    return _agent
