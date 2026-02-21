"""
Validation Pipeline - Code Validation and Test Generation

Validates generated code for syntax, style, and correctness.
Automatically generates tests.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import logging
import ast
import re

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation strictness levels"""
    BASIC = "basic"  # Syntax only
    STANDARD = "standard"  # Syntax + style
    STRICT = "strict"  # Syntax + style + type checking
    PRODUCTION = "production"  # All checks + security


@dataclass
class ValidationIssue:
    """Single validation issue"""
    severity: str  # error, warning, info
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    rule: Optional[str] = None
    suggested_fix: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of code validation"""
    is_valid: bool
    language: str
    issues: List[ValidationIssue] = field(default_factory=list)

    # Metrics
    syntax_valid: bool = True
    style_compliant: bool = True
    type_safe: bool = True
    security_issues: int = 0

    # Details
    lines_of_code: int = 0
    complexity_score: int = 0

    def add_issue(
        self,
        severity: str,
        message: str,
        line: Optional[int] = None,
        rule: Optional[str] = None
    ):
        """Add validation issue"""
        self.issues.append(ValidationIssue(
            severity=severity,
            message=message,
            line=line,
            rule=rule
        ))

        if severity == "error":
            self.is_valid = False


class CodeValidator:
    """
    Multi-language code validator

    Validates code for syntax, style, type safety, and security issues.
    """

    def __init__(self):
        self._validators = {
            "python": self._validate_python,
            "typescript": self._validate_typescript,
            "javascript": self._validate_javascript
        }

    async def validate(
        self,
        code: str,
        language: str,
        level: ValidationLevel = ValidationLevel.STANDARD
    ) -> ValidationResult:
        """
        Validate code

        Args:
            code: Code to validate
            language: Programming language
            level: Validation strictness

        Returns:
            Validation result with issues
        """
        logger.info(f"Validating {language} code (level: {level.value})")

        result = ValidationResult(
            is_valid=True,
            language=language,
            lines_of_code=len(code.split('\n'))
        )

        # Get validator for language
        validator_func = self._validators.get(language.lower())
        if not validator_func:
            result.add_issue("warning", f"No validator available for {language}")
            return result

        # Run validation
        await validator_func(code, level, result)

        logger.info(f"Validation complete: {len(result.issues)} issues found")
        return result

    async def _validate_python(
        self,
        code: str,
        level: ValidationLevel,
        result: ValidationResult
    ):
        """Validate Python code"""

        # Syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            result.syntax_valid = False
            result.add_issue(
                "error",
                f"Syntax error: {e.msg}",
                line=e.lineno,
                rule="syntax"
            )
            return  # Can't continue if syntax is invalid

        # Style check (PEP 8)
        if level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.PRODUCTION]:
            await self._check_python_style(code, result)

        # Type checking
        if level in [ValidationLevel.STRICT, ValidationLevel.PRODUCTION]:
            await self._check_python_types(code, result)

        # Security check
        if level == ValidationLevel.PRODUCTION:
            await self._check_python_security(code, result)

    async def _check_python_style(self, code: str, result: ValidationResult):
        """Check Python style (PEP 8)"""

        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 88:
                result.add_issue(
                    "warning",
                    f"Line too long ({len(line)} > 88 characters)",
                    line=i,
                    rule="E501"
                )

            # Check trailing whitespace
            if line.rstrip() != line:
                result.add_issue(
                    "warning",
                    "Trailing whitespace",
                    line=i,
                    rule="W291"
                )

            # Check indentation (should be 4 spaces)
            if line and line[0] == ' ':
                leading_spaces = len(line) - len(line.lstrip(' '))
                if leading_spaces % 4 != 0:
                    result.add_issue(
                        "warning",
                        "Indentation is not a multiple of 4",
                        line=i,
                        rule="E111"
                    )

        # Check for missing docstrings
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if not docstring:
                    result.add_issue(
                        "info",
                        f"Missing docstring for {node.name}",
                        line=node.lineno,
                        rule="D100"
                    )

    async def _check_python_types(self, code: str, result: ValidationResult):
        """Check Python type hints"""

        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for return type annotation
                if not node.returns:
                    result.add_issue(
                        "warning",
                        f"Missing return type annotation for {node.name}",
                        line=node.lineno,
                        rule="ANN201"
                    )

                # Check for parameter type annotations
                for arg in node.args.args:
                    if not arg.annotation:
                        result.add_issue(
                            "warning",
                            f"Missing type annotation for parameter {arg.arg}",
                            line=node.lineno,
                            rule="ANN001"
                        )

    async def _check_python_security(self, code: str, result: ValidationResult):
        """Check for security issues"""

        # Check for dangerous functions
        dangerous_patterns = [
            (r'\beval\s*\(', "Use of eval() is dangerous"),
            (r'\bexec\s*\(', "Use of exec() is dangerous"),
            (r'__import__\s*\(', "Dynamic imports can be dangerous"),
            (r'pickle\.loads', "Unpickling untrusted data is dangerous"),
            (r'subprocess\..*shell\s*=\s*True', "Shell injection vulnerability")
        ]

        for pattern, message in dangerous_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                result.security_issues += 1
                line_num = code[:match.start()].count('\n') + 1
                result.add_issue(
                    "error",
                    message,
                    line=line_num,
                    rule="security"
                )

    async def _validate_typescript(
        self,
        code: str,
        level: ValidationLevel,
        result: ValidationResult
    ):
        """Validate TypeScript code"""

        # Basic syntax patterns
        if not self._check_balanced_braces(code):
            result.syntax_valid = False
            result.add_issue("error", "Unbalanced braces", rule="syntax")

        # Check for common issues
        if "any" in code and level in [ValidationLevel.STRICT, ValidationLevel.PRODUCTION]:
            result.add_issue(
                "warning",
                "Avoid using 'any' type, use specific types",
                rule="@typescript-eslint/no-explicit-any"
            )

    async def _validate_javascript(
        self,
        code: str,
        level: ValidationLevel,
        result: ValidationResult
    ):
        """Validate JavaScript code"""

        # Basic syntax patterns
        if not self._check_balanced_braces(code):
            result.syntax_valid = False
            result.add_issue("error", "Unbalanced braces", rule="syntax")

        # Check for console.log (should be removed in production)
        if level == ValidationLevel.PRODUCTION and "console.log" in code:
            result.add_issue(
                "warning",
                "Remove console.log statements in production",
                rule="no-console"
            )

    def _check_balanced_braces(self, code: str) -> bool:
        """Check if braces are balanced"""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}

        for char in code:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack or pairs[stack.pop()] != char:
                    return False

        return len(stack) == 0


class TestGenerator:
    """
    Automatic test generation

    Generates unit tests for code automatically.
    """

    def __init__(self, llm_provider: Optional[Any] = None):
        self.llm_provider = llm_provider

    async def generate_tests(
        self,
        code: str,
        language: str,
        test_framework: Optional[str] = None
    ) -> str:
        """
        Generate tests for code

        Args:
            code: Code to test
            language: Programming language
            test_framework: Test framework to use

        Returns:
            Generated test code
        """
        logger.info(f"Generating tests for {language} code")

        # Determine test framework
        if not test_framework:
            test_framework = self._default_framework(language)

        if self.llm_provider:
            tests = await self._generate_tests_with_llm(
                code, language, test_framework
            )
        else:
            tests = self._generate_tests_template(
                code, language, test_framework
            )

        return tests

    async def _generate_tests_with_llm(
        self,
        code: str,
        language: str,
        framework: str
    ) -> str:
        """Generate tests using LLM"""

        prompt = f"""Generate comprehensive unit tests for the following {language} code:

```{language}
{code}
```

Requirements:
- Use {framework} testing framework
- Test all public functions/methods
- Include edge cases and error conditions
- Test boundary conditions
- Include setup and teardown if needed
- Add descriptive test names

Return only the test code in a code block."""

        response = await self.llm_provider.generate(prompt)

        # Extract code from response
        pattern = f"```{language}\\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        return response

    def _generate_tests_template(
        self,
        code: str,
        language: str,
        framework: str
    ) -> str:
        """Generate tests from template"""

        if language == "python":
            return self._generate_python_test_template(code, framework)
        elif language in ["typescript", "javascript"]:
            return self._generate_js_test_template(code, framework)

        return f"# TODO: Generate tests for {language}"

    def _generate_python_test_template(self, code: str, framework: str) -> str:
        """Generate Python test template"""

        # Extract function names
        try:
            tree = ast.parse(code)
            functions = [
                node.name for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and not node.name.startswith('_')
            ]
        except:
            functions = []

        tests = []
        tests.append("import pytest")
        tests.append("from module import *")
        tests.append("")

        for func_name in functions:
            tests.append(f"def test_{func_name}():")
            tests.append(f'    """Test {func_name} function"""')
            tests.append("    # TODO: Implement test")
            tests.append("    assert True")
            tests.append("")

        return "\n".join(tests)

    def _generate_js_test_template(self, code: str, framework: str) -> str:
        """Generate JavaScript test template"""

        # Extract function names (simple regex)
        pattern = r'function\s+(\w+)|const\s+(\w+)\s*=.*?=>'
        matches = re.findall(pattern, code)
        functions = [m[0] or m[1] for m in matches]

        tests = []
        tests.append("import { describe, it, expect } from 'vitest';")
        tests.append("import { " + ", ".join(functions) + " } from './module';")
        tests.append("")

        for func_name in functions:
            tests.append(f"describe('{func_name}', () => {{")
            tests.append(f"  it('should work correctly', () => {{")
            tests.append("    // TODO: Implement test")
            tests.append("    expect(true).toBe(true);")
            tests.append("  });")
            tests.append("});")
            tests.append("")

        return "\n".join(tests)

    def _default_framework(self, language: str) -> str:
        """Get default test framework for language"""
        frameworks = {
            "python": "pytest",
            "typescript": "vitest",
            "javascript": "jest",
            "go": "testing",
            "rust": "cargo test",
            "java": "junit"
        }
        return frameworks.get(language, "unknown")
