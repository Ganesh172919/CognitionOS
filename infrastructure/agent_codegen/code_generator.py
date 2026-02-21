"""
Code Generator - Multi-language Code Generation Engine

Generates production-ready code from specifications using LLM and templates.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import json
import ast
import re

logger = logging.getLogger(__name__)


class Language(str, Enum):
    """Supported programming languages"""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CSHARP = "csharp"


class CodeStyle(str, Enum):
    """Code style conventions"""
    FUNCTIONAL = "functional"
    OBJECT_ORIENTED = "object_oriented"
    PROCEDURAL = "procedural"
    REACTIVE = "reactive"


@dataclass
class CodeContext:
    """Context for code generation"""
    language: Language
    style: CodeStyle
    framework: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)

    # Project structure
    module_name: Optional[str] = None
    package_name: Optional[str] = None

    # Code standards
    max_line_length: int = 88
    use_type_hints: bool = True
    include_docstrings: bool = True
    include_tests: bool = True

    # Existing codebase context
    existing_files: Dict[str, str] = field(default_factory=dict)
    code_patterns: List[str] = field(default_factory=list)
    naming_conventions: Dict[str, str] = field(default_factory=dict)


@dataclass
class GeneratedCode:
    """Generated code with metadata"""
    code: str
    language: Language
    file_path: str

    # Quality metrics
    estimated_quality: float = 0.0  # 0-1 score
    complexity_score: int = 0
    test_coverage: float = 0.0

    # Dependencies
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    # Documentation
    docstring: Optional[str] = None
    inline_comments: int = 0

    # Validation results
    syntax_valid: bool = False
    passes_linter: bool = False
    passes_type_check: bool = False

    # Metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    generation_time_ms: int = 0
    tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "file_path": self.file_path,
            "language": self.language.value,
            "quality_metrics": {
                "estimated_quality": self.estimated_quality,
                "complexity_score": self.complexity_score,
                "test_coverage": self.test_coverage,
                "syntax_valid": self.syntax_valid,
                "passes_linter": self.passes_linter,
                "passes_type_check": self.passes_type_check
            },
            "imports": self.imports,
            "dependencies": self.dependencies,
            "metadata": {
                "generated_at": self.generated_at.isoformat(),
                "generation_time_ms": self.generation_time_ms,
                "tokens_used": self.tokens_used
            }
        }


class LanguageSupport:
    """Language-specific code generation support"""

    TEMPLATES = {
        Language.PYTHON: {
            "class": '''class {class_name}:
    """
    {docstring}
    """

    def __init__(self{init_params}):
        {init_body}

    {methods}
''',
            "function": '''def {function_name}({params}){return_type}:
    """
    {docstring}
    """
    {body}
''',
            "test": '''def test_{test_name}():
    """Test {description}"""
    {test_body}
'''
        },
        Language.TYPESCRIPT: {
            "class": '''export class {class_name} {
    {properties}

    constructor({constructor_params}) {
        {constructor_body}
    }

    {methods}
}
''',
            "function": '''export function {function_name}({params}): {return_type} {
    {body}
}
''',
            "test": '''describe('{suite_name}', () => {
    it('{test_name}', () => {
        {test_body}
    });
});
'''
        }
    }

    @staticmethod
    def get_file_extension(language: Language) -> str:
        """Get file extension for language"""
        extensions = {
            Language.PYTHON: ".py",
            Language.TYPESCRIPT: ".ts",
            Language.JAVASCRIPT: ".js",
            Language.GO: ".go",
            Language.RUST: ".rs",
            Language.JAVA: ".java",
            Language.CSHARP: ".cs"
        }
        return extensions.get(language, ".txt")

    @staticmethod
    def get_import_statement(language: Language, module: str, items: List[str]) -> str:
        """Generate import statement"""
        if language == Language.PYTHON:
            if items:
                return f"from {module} import {', '.join(items)}"
            return f"import {module}"
        elif language in [Language.TYPESCRIPT, Language.JAVASCRIPT]:
            if items:
                return f"import {{ {', '.join(items)} }} from '{module}';"
            return f"import {module};"
        elif language == Language.GO:
            return f"import \"{module}\""
        return ""

    @staticmethod
    def get_docstring_format(language: Language, content: str) -> str:
        """Format docstring for language"""
        if language == Language.PYTHON:
            return f'"""\n{content}\n"""'
        elif language in [Language.TYPESCRIPT, Language.JAVASCRIPT]:
            lines = content.split('\n')
            return "/**\n" + "\n".join(f" * {line}" for line in lines) + "\n */"
        elif language == Language.JAVA:
            lines = content.split('\n')
            return "/**\n" + "\n".join(f" * {line}" for line in lines) + "\n */"
        return f"// {content}"


class CodeGenerator:
    """
    Autonomous code generator

    Generates production-ready code from specifications using LLM assistance
    and structured templates.
    """

    def __init__(self, llm_provider: Optional[Any] = None):
        self.llm_provider = llm_provider
        self.language_support = LanguageSupport()

    async def generate(
        self,
        specification: str,
        context: CodeContext,
        examples: Optional[List[str]] = None
    ) -> GeneratedCode:
        """
        Generate code from specification

        Args:
            specification: Natural language description of what to build
            context: Code generation context
            examples: Optional example code snippets

        Returns:
            Generated code with metadata
        """
        logger.info(f"Generating {context.language.value} code: {specification[:50]}...")

        start_time = datetime.utcnow()

        # Build generation prompt
        prompt = self._build_generation_prompt(specification, context, examples)

        # Generate code using LLM
        if self.llm_provider:
            generated_text = await self.llm_provider.generate(prompt)
            code = self._extract_code(generated_text, context.language)
        else:
            # Fallback: template-based generation
            code = self._generate_from_template(specification, context)

        # Post-process code
        code = self._post_process_code(code, context)

        # Extract metadata
        imports = self._extract_imports(code, context.language)
        complexity = self._calculate_complexity(code, context.language)

        # Create result
        file_ext = self.language_support.get_file_extension(context.language)
        file_path = f"{context.module_name or 'generated'}{file_ext}"

        generation_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        result = GeneratedCode(
            code=code,
            language=context.language,
            file_path=file_path,
            imports=imports,
            dependencies=context.dependencies,
            complexity_score=complexity,
            generation_time_ms=generation_time
        )

        return result

    def _build_generation_prompt(
        self,
        specification: str,
        context: CodeContext,
        examples: Optional[List[str]]
    ) -> str:
        """Build prompt for LLM code generation"""

        prompt_parts = [
            f"Generate production-ready {context.language.value} code.",
            f"Style: {context.style.value}",
            "",
            "Specification:",
            specification,
            ""
        ]

        if context.framework:
            prompt_parts.append(f"Framework: {context.framework}")

        if context.dependencies:
            prompt_parts.append(f"Dependencies: {', '.join(context.dependencies)}")

        # Code standards
        prompt_parts.extend([
            "",
            "Requirements:",
            f"- Max line length: {context.max_line_length}",
            f"- Type hints: {'Required' if context.use_type_hints else 'Optional'}",
            f"- Docstrings: {'Required' if context.include_docstrings else 'Optional'}",
            "- Follow language best practices",
            "- Include error handling",
            "- Use descriptive variable names"
        ])

        # Naming conventions
        if context.naming_conventions:
            prompt_parts.append("")
            prompt_parts.append("Naming conventions:")
            for pattern, convention in context.naming_conventions.items():
                prompt_parts.append(f"- {pattern}: {convention}")

        # Examples
        if examples:
            prompt_parts.append("")
            prompt_parts.append("Example code patterns:")
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(example)

        prompt_parts.extend([
            "",
            "Generate only the code, without explanations.",
            "Wrap code in ```language``` blocks."
        ])

        return "\n".join(prompt_parts)

    def _extract_code(self, generated_text: str, language: Language) -> str:
        """Extract code from LLM response"""
        # Look for code blocks
        pattern = f"```{language.value}\\n(.*?)```"
        matches = re.findall(pattern, generated_text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Fallback: look for any code block
        pattern = r"```\w*\n(.*?)```"
        matches = re.findall(pattern, generated_text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Last resort: return as-is
        return generated_text.strip()

    def _generate_from_template(
        self,
        specification: str,
        context: CodeContext
    ) -> str:
        """Generate code from template (fallback)"""

        # Very simple template-based generation
        templates = self.language_support.TEMPLATES.get(context.language, {})

        if "function" in specification.lower():
            template = templates.get("function", "")
            return template.format(
                function_name="generated_function",
                params="",
                return_type="",
                docstring=specification,
                body="    pass" if context.language == Language.PYTHON else "    // TODO: Implement"
            )
        elif "class" in specification.lower():
            template = templates.get("class", "")
            return template.format(
                class_name="GeneratedClass",
                init_params="",
                init_body="pass" if context.language == Language.PYTHON else "// TODO",
                methods="",
                docstring=specification
            )

        return f"# TODO: Implement\n# {specification}"

    def _post_process_code(self, code: str, context: CodeContext) -> str:
        """Post-process generated code"""

        if context.language == Language.PYTHON:
            # Format with standard conventions
            lines = code.split('\n')

            # Ensure proper spacing
            processed_lines = []
            prev_was_blank = False

            for line in lines:
                is_blank = not line.strip()

                # Avoid multiple consecutive blank lines
                if is_blank and prev_was_blank:
                    continue

                processed_lines.append(line)
                prev_was_blank = is_blank

            code = '\n'.join(processed_lines)

        return code

    def _extract_imports(self, code: str, language: Language) -> List[str]:
        """Extract imports from code"""
        imports = []

        if language == Language.PYTHON:
            # Parse Python imports
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
            except SyntaxError:
                # Fallback to regex
                import_pattern = r'(?:from\s+([\w.]+)\s+)?import\s+([\w,\s]+)'
                matches = re.findall(import_pattern, code)
                for match in matches:
                    if match[0]:
                        imports.append(match[0])

        elif language in [Language.TYPESCRIPT, Language.JAVASCRIPT]:
            # Extract ES6 imports
            pattern = r'import\s+.*?\s+from\s+[\'"](.+?)[\'"]'
            imports = re.findall(pattern, code)

        return list(set(imports))

    def _calculate_complexity(self, code: str, language: Language) -> int:
        """Calculate code complexity score"""

        # Simple cyclomatic complexity approximation
        complexity = 1  # Base complexity

        # Count decision points
        decision_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'case']

        for keyword in decision_keywords:
            # Simple word boundary search
            pattern = r'\b' + keyword + r'\b'
            complexity += len(re.findall(pattern, code))

        # Count functions (each adds to complexity)
        if language == Language.PYTHON:
            complexity += len(re.findall(r'\bdef\s+\w+', code))
        elif language in [Language.TYPESCRIPT, Language.JAVASCRIPT]:
            complexity += len(re.findall(r'\bfunction\s+\w+', code))
            complexity += len(re.findall(r'=>', code))

        return min(complexity, 100)  # Cap at 100

    async def refactor_code(
        self,
        existing_code: str,
        refactoring_goal: str,
        context: CodeContext
    ) -> GeneratedCode:
        """
        Refactor existing code

        Args:
            existing_code: Code to refactor
            refactoring_goal: What to improve
            context: Code context

        Returns:
            Refactored code
        """
        logger.info(f"Refactoring code: {refactoring_goal}")

        if self.llm_provider:
            prompt = f"""Refactor the following {context.language.value} code:

Goal: {refactoring_goal}

Original code:
```{context.language.value}
{existing_code}
```

Requirements:
- Maintain functionality
- Improve code quality
- Follow best practices
- Add type hints if missing
- Improve variable names

Return only the refactored code in a code block."""

            generated_text = await self.llm_provider.generate(prompt)
            refactored_code = self._extract_code(generated_text, context.language)
        else:
            # Simple refactoring: just add formatting
            refactored_code = self._post_process_code(existing_code, context)

        return GeneratedCode(
            code=refactored_code,
            language=context.language,
            file_path="refactored" + self.language_support.get_file_extension(context.language),
            imports=self._extract_imports(refactored_code, context.language),
            complexity_score=self._calculate_complexity(refactored_code, context.language)
        )

    async def generate_batch(
        self,
        specifications: List[tuple[str, CodeContext]],
        max_parallel: int = 3
    ) -> List[GeneratedCode]:
        """Generate multiple code files in parallel"""
        import asyncio

        results = []
        semaphore = asyncio.Semaphore(max_parallel)

        async def generate_one(spec: str, ctx: CodeContext):
            async with semaphore:
                return await self.generate(spec, ctx)

        tasks = [generate_one(spec, ctx) for spec, ctx in specifications]
        results = await asyncio.gather(*tasks)

        return results
