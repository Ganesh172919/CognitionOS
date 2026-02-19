"""
Intelligent Code Generation System
Generates, validates, and optimizes code with safety checks.
"""

import ast
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field


class CodeLanguage(str, Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    SQL = "sql"


class CodeQuality(str, Enum):
    """Code quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"


@dataclass
class ValidationResult:
    """Result of code validation"""
    is_valid: bool
    quality: CodeQuality
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    metrics: Dict[str, Any]


class CodeTemplate(BaseModel):
    """Template for code generation"""
    name: str
    language: CodeLanguage
    template: str
    variables: List[str] = Field(default_factory=list)
    description: str = ""


class GeneratedCode(BaseModel):
    """Generated code with metadata"""
    code: str
    language: CodeLanguage
    purpose: str
    dependencies: List[str] = Field(default_factory=list)
    test_code: Optional[str] = None
    documentation: Optional[str] = None
    validation: Optional[ValidationResult] = None


class IntelligentCodeGenerator:
    """
    Generates production-grade code with validation and optimization.
    """

    def __init__(self):
        self.templates = self._initialize_templates()
        self.safety_patterns = self._initialize_safety_patterns()

    def _initialize_templates(self) -> Dict[str, CodeTemplate]:
        """Initialize code templates for common patterns"""
        return {
            "python_api_endpoint": CodeTemplate(
                name="FastAPI Endpoint",
                language=CodeLanguage.PYTHON,
                template='''
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from pydantic import BaseModel

router = APIRouter()

class {model_name}Request(BaseModel):
    {request_fields}

class {model_name}Response(BaseModel):
    {response_fields}

@router.{method}("/{endpoint_path}")
async def {function_name}(
    {parameters}
) -> {model_name}Response:
    """
    {docstring}
    """
    try:
        # Implementation
        {implementation}

        return {model_name}Response({return_values})
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
''',
                variables=["model_name", "method", "endpoint_path", "function_name",
                          "request_fields", "response_fields", "parameters",
                          "docstring", "implementation", "return_values"],
                description="FastAPI REST endpoint with error handling"
            ),
            "python_repository": CodeTemplate(
                name="Repository Pattern",
                language=CodeLanguage.PYTHON,
                template='''
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

class {entity_name}Repository:
    """Repository for {entity_name} operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, entity: {entity_name}) -> {entity_name}:
        """Create a new {entity_name}"""
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def get_by_id(self, entity_id: str) -> Optional[{entity_name}]:
        """Get {entity_name} by ID"""
        result = await self.session.execute(
            select({entity_name}).where({entity_name}.id == entity_id)
        )
        return result.scalar_one_or_none()

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[{entity_name}]:
        """Get all {entity_name}s with pagination"""
        result = await self.session.execute(
            select({entity_name})
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def update(self, entity_id: str, **kwargs) -> Optional[{entity_name}]:
        """Update {entity_name}"""
        await self.session.execute(
            update({entity_name})
            .where({entity_name}.id == entity_id)
            .values(**kwargs)
        )
        return await self.get_by_id(entity_id)

    async def delete(self, entity_id: str) -> bool:
        """Delete {entity_name}"""
        result = await self.session.execute(
            delete({entity_name}).where({entity_name}.id == entity_id)
        )
        return result.rowcount > 0
''',
                variables=["entity_name"],
                description="SQLAlchemy async repository with CRUD operations"
            ),
            "python_service_class": CodeTemplate(
                name="Service Class",
                language=CodeLanguage.PYTHON,
                template='''
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class {service_name}:
    """
    {service_description}
    """

    def __init__(self, {dependencies}):
        {dependency_assignments}
        logger.info(f"{service_name} initialized")

    async def {primary_method}(self, {method_parameters}) -> {return_type}:
        """
        {method_description}
        """
        try:
            logger.debug(f"Executing {primary_method} with parameters: {method_parameters}")

            # Validation
            {validation_logic}

            # Core logic
            {core_logic}

            # Post-processing
            {post_processing}

            logger.info(f"{primary_method} completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error in {primary_method}: {{str(e)}}")
            raise
''',
                variables=["service_name", "service_description", "dependencies",
                          "dependency_assignments", "primary_method", "method_parameters",
                          "return_type", "method_description", "validation_logic",
                          "core_logic", "post_processing"],
                description="Service class with logging and error handling"
            )
        }

    def _initialize_safety_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for safety checking"""
        return {
            "dangerous_imports": [
                "os.system",
                "subprocess.call",
                "eval",
                "exec",
                "__import__",
                "compile"
            ],
            "sql_injection_patterns": [
                r"execute\([^)]*%s[^)]*\)",
                r"execute\([^)]*\+[^)]*\)",
                r"WHERE.*=.*\+",
                r"raw\([^)]*\+[^)]*\)"
            ],
            "security_issues": [
                r"pickle\.loads",
                r"yaml\.load\(",
                r"password\s*=\s*['\"]",
                r"api_key\s*=\s*['\"]",
                r"secret\s*=\s*['\"]"
            ]
        }

    async def generate_code(
        self,
        purpose: str,
        language: CodeLanguage,
        context: Dict[str, Any],
        template_name: Optional[str] = None
    ) -> GeneratedCode:
        """
        Generate code based on purpose and context
        """
        if template_name and template_name in self.templates:
            code = self._generate_from_template(template_name, context)
        else:
            code = await self._generate_custom(purpose, language, context)

        # Generate tests
        test_code = await self._generate_tests(code, language, purpose)

        # Generate documentation
        documentation = self._generate_documentation(code, language, purpose)

        # Detect dependencies
        dependencies = self._detect_dependencies(code, language)

        generated = GeneratedCode(
            code=code,
            language=language,
            purpose=purpose,
            dependencies=dependencies,
            test_code=test_code,
            documentation=documentation
        )

        # Validate
        validation = await self.validate_code(generated)
        generated.validation = validation

        return generated

    def _generate_from_template(
        self,
        template_name: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate code from template"""
        template = self.templates[template_name]
        code = template.template

        # Replace variables
        for var in template.variables:
            value = context.get(var, f"# TODO: Implement {var}")
            placeholder = "{" + var + "}"
            code = code.replace(placeholder, str(value))

        return self._clean_code(code)

    async def _generate_custom(
        self,
        purpose: str,
        language: CodeLanguage,
        context: Dict[str, Any]
    ) -> str:
        """Generate custom code (would use LLM in production)"""
        # Simplified implementation - in production would use LLM
        if language == CodeLanguage.PYTHON:
            return f'''
"""
{purpose}
"""

async def implement_{purpose.lower().replace(" ", "_")}():
    """
    Implementation for: {purpose}
    """
    # TODO: Implement functionality
    pass
'''
        else:
            return f"// Implementation for: {purpose}\n// TODO: Add implementation"

    async def _generate_tests(
        self,
        code: str,
        language: CodeLanguage,
        purpose: str
    ) -> Optional[str]:
        """Generate test code"""
        if language != CodeLanguage.PYTHON:
            return None

        # Extract function names from code
        function_names = self._extract_function_names(code)

        test_code = f'''
"""
Tests for: {purpose}
"""

import pytest
from unittest.mock import Mock, patch

'''

        for func_name in function_names:
            test_code += f'''
@pytest.mark.asyncio
async def test_{func_name}_success():
    """Test {func_name} successful execution"""
    # Arrange
    # TODO: Set up test data

    # Act
    result = await {func_name}()

    # Assert
    assert result is not None

@pytest.mark.asyncio
async def test_{func_name}_error_handling():
    """Test {func_name} error handling"""
    # Arrange
    # TODO: Set up error condition

    # Act & Assert
    with pytest.raises(Exception):
        await {func_name}()

'''

        return test_code

    def _generate_documentation(
        self,
        code: str,
        language: CodeLanguage,
        purpose: str
    ) -> str:
        """Generate documentation"""
        return f"""
# {purpose}

## Overview
This module implements: {purpose}

## Usage
```{language.value}
{code[:200]}...
```

## API Reference
- See inline documentation in code

## Dependencies
- See imports in code

## Testing
- Run tests with: pytest tests/
"""

    def _detect_dependencies(self, code: str, language: CodeLanguage) -> List[str]:
        """Detect code dependencies"""
        dependencies = []

        if language == CodeLanguage.PYTHON:
            # Extract imports
            import_pattern = r"^(?:from|import)\s+([\w.]+)"
            matches = re.findall(import_pattern, code, re.MULTILINE)

            # Filter to third-party packages (simplified)
            third_party = []
            for match in matches:
                root_package = match.split('.')[0]
                if root_package not in ['typing', 'dataclasses', 'enum', 'datetime']:
                    if root_package not in third_party:
                        third_party.append(root_package)

            dependencies = third_party

        return dependencies

    def _extract_function_names(self, code: str) -> List[str]:
        """Extract function names from Python code"""
        try:
            tree = ast.parse(code)
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not node.name.startswith('_'):  # Skip private functions
                        functions.append(node.name)
            return functions
        except:
            return []

    async def validate_code(self, generated: GeneratedCode) -> ValidationResult:
        """
        Comprehensive code validation
        """
        errors = []
        warnings = []
        suggestions = []
        metrics = {}

        code = generated.code

        # Syntax validation
        if generated.language == CodeLanguage.PYTHON:
            syntax_valid, syntax_errors = self._validate_python_syntax(code)
            if not syntax_valid:
                errors.extend(syntax_errors)

        # Safety checks
        safety_issues = self._check_safety(code)
        if safety_issues:
            errors.extend(safety_issues)

        # Code quality metrics
        metrics = self._calculate_metrics(code, generated.language)

        # Generate suggestions
        suggestions = self._generate_suggestions(code, metrics)

        # Determine overall quality
        quality = self._determine_quality(errors, warnings, metrics)

        return ValidationResult(
            is_valid=len(errors) == 0,
            quality=quality,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            metrics=metrics
        )

    def _validate_python_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Validate Python syntax"""
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            return False, [f"Syntax error at line {e.lineno}: {e.msg}"]

    def _check_safety(self, code: str) -> List[str]:
        """Check for security issues"""
        issues = []

        # Check dangerous imports
        for pattern in self.safety_patterns["dangerous_imports"]:
            if pattern in code:
                issues.append(f"Potentially dangerous usage: {pattern}")

        # Check SQL injection patterns
        for pattern in self.safety_patterns["sql_injection_patterns"]:
            if re.search(pattern, code):
                issues.append(f"Potential SQL injection risk: {pattern}")

        # Check security issues
        for pattern in self.safety_patterns["security_issues"]:
            if re.search(pattern, code):
                issues.append(f"Security concern: {pattern}")

        return issues

    def _calculate_metrics(self, code: str, language: CodeLanguage) -> Dict[str, Any]:
        """Calculate code quality metrics"""
        lines = code.split('\n')

        metrics = {
            "total_lines": len(lines),
            "code_lines": len([l for l in lines if l.strip() and not l.strip().startswith('#')]),
            "comment_lines": len([l for l in lines if l.strip().startswith('#')]),
            "blank_lines": len([l for l in lines if not l.strip()]),
            "average_line_length": sum(len(l) for l in lines) / max(len(lines), 1)
        }

        if language == CodeLanguage.PYTHON:
            try:
                tree = ast.parse(code)
                metrics["functions"] = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
                metrics["classes"] = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            except:
                pass

        return metrics

    def _generate_suggestions(self, code: str, metrics: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []

        # Check line length
        if metrics.get("average_line_length", 0) > 100:
            suggestions.append("Consider breaking long lines for better readability")

        # Check comments
        code_lines = metrics.get("code_lines", 0)
        comment_lines = metrics.get("comment_lines", 0)
        if code_lines > 50 and comment_lines < code_lines * 0.1:
            suggestions.append("Consider adding more comments for complex logic")

        # Check function count
        if metrics.get("functions", 0) > 10:
            suggestions.append("Consider splitting into multiple modules")

        return suggestions

    def _determine_quality(
        self,
        errors: List[str],
        warnings: List[str],
        metrics: Dict[str, Any]
    ) -> CodeQuality:
        """Determine overall code quality"""
        if errors:
            return CodeQuality.POOR

        score = 100

        # Deduct for warnings
        score -= len(warnings) * 10

        # Deduct for poor metrics
        if metrics.get("average_line_length", 0) > 120:
            score -= 15

        code_lines = metrics.get("code_lines", 1)
        comment_lines = metrics.get("comment_lines", 0)
        comment_ratio = comment_lines / code_lines if code_lines > 0 else 0

        if comment_ratio < 0.05:
            score -= 10

        # Determine quality level
        if score >= 90:
            return CodeQuality.EXCELLENT
        elif score >= 75:
            return CodeQuality.GOOD
        elif score >= 60:
            return CodeQuality.ACCEPTABLE
        elif score >= 40:
            return CodeQuality.NEEDS_IMPROVEMENT
        else:
            return CodeQuality.POOR

    def _clean_code(self, code: str) -> str:
        """Clean and format generated code"""
        # Remove excessive blank lines
        lines = code.split('\n')
        cleaned = []
        prev_blank = False

        for line in lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            cleaned.append(line)
            prev_blank = is_blank

        return '\n'.join(cleaned).strip()

    async def optimize_code(self, code: str, language: CodeLanguage) -> str:
        """Optimize generated code"""
        # In production, would use AST transformations and LLM
        return code

    async def refactor_code(
        self,
        code: str,
        language: CodeLanguage,
        improvements: List[str]
    ) -> str:
        """Refactor code based on suggestions"""
        # In production, would apply specific refactorings
        return code
