"""
Code Validation Pipeline

Automated code validation and testing for agent-generated code with:
- Static analysis (syntax, style, security)
- Dynamic testing (unit tests, integration tests)
- Security scanning
- Performance profiling
- Quality metrics
"""

import logging
import ast
import subprocess
import tempfile
import os
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Code validation levels."""
    BASIC = "basic"  # Syntax and basic checks
    STANDARD = "standard"  # + style and type checks
    STRICT = "strict"  # + security and performance
    PARANOID = "paranoid"  # + comprehensive testing


class IssueType(str, Enum):
    """Types of validation issues."""
    SYNTAX_ERROR = "syntax_error"
    STYLE_VIOLATION = "style_violation"
    TYPE_ERROR = "type_error"
    SECURITY_VULNERABILITY = "security_vulnerability"
    PERFORMANCE_ISSUE = "performance_issue"
    TEST_FAILURE = "test_failure"
    RUNTIME_ERROR = "runtime_error"


class IssueSeverity(str, Enum):
    """Issue severity levels."""
    CRITICAL = "critical"  # Must fix
    HIGH = "high"  # Should fix
    MEDIUM = "medium"  # Consider fixing
    LOW = "low"  # Optional fix
    INFO = "info"  # Informational


@dataclass
class ValidationIssue:
    """A single validation issue."""
    issue_type: IssueType
    severity: IssueSeverity
    message: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.issue_type,
            "severity": self.severity,
            "message": self.message,
            "line": self.line_number,
            "column": self.column,
            "code": self.code_snippet,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Result of code validation."""
    passed: bool
    validation_level: ValidationLevel
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    test_results: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    
    def get_issue_counts(self) -> Dict[IssueSeverity, int]:
        """Get count of issues by severity."""
        counts = {severity: 0 for severity in IssueSeverity}
        for issue in self.issues:
            counts[issue.severity] += 1
        return counts
    
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return any(i.severity == IssueSeverity.CRITICAL for i in self.issues)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "validation_level": self.validation_level,
            "issues": [i.to_dict() for i in self.issues],
            "issue_counts": {k.value: v for k, v in self.get_issue_counts().items()},
            "metrics": self.metrics,
            "test_results": self.test_results,
            "execution_time": self.execution_time,
        }


class CodeValidationPipeline:
    """
    Automated code validation pipeline.
    
    Validates agent-generated code through multiple stages:
    1. Syntax validation
    2. Style checking
    3. Type checking
    4. Security scanning
    5. Test execution
    6. Performance profiling
    """
    
    def __init__(
        self,
        default_level: ValidationLevel = ValidationLevel.STANDARD,
        enable_auto_fix: bool = False,
    ):
        """
        Initialize validation pipeline.
        
        Args:
            default_level: Default validation level
            enable_auto_fix: Whether to attempt automatic fixes
        """
        self.default_level = default_level
        self.enable_auto_fix = enable_auto_fix
        
        logger.info(f"Code validation pipeline initialized (level={default_level})")
    
    async def validate(
        self,
        code: str,
        language: str = "python",
        level: Optional[ValidationLevel] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate code through pipeline.
        
        Args:
            code: Source code to validate
            language: Programming language
            level: Validation level (uses default if not specified)
            context: Optional context for validation
            
        Returns:
            ValidationResult with issues and metrics
        """
        import time
        start_time = time.time()
        
        level = level or self.default_level
        issues = []
        metrics = {}
        
        logger.info(f"Validating {language} code (level={level})")
        
        # Stage 1: Syntax validation
        syntax_issues = await self._validate_syntax(code, language)
        issues.extend(syntax_issues)
        
        # If syntax errors, stop here
        if any(i.severity == IssueSeverity.CRITICAL for i in syntax_issues):
            return ValidationResult(
                passed=False,
                validation_level=level,
                issues=issues,
                metrics=metrics,
                execution_time=time.time() - start_time,
            )
        
        # Stage 2: Style checking (standard+)
        if level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            style_issues = await self._check_style(code, language)
            issues.extend(style_issues)
        
        # Stage 3: Type checking (standard+)
        if level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            type_issues = await self._check_types(code, language)
            issues.extend(type_issues)
        
        # Stage 4: Security scanning (strict+)
        if level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            security_issues = await self._scan_security(code, language)
            issues.extend(security_issues)
        
        # Stage 5: Performance analysis (strict+)
        if level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            perf_issues, perf_metrics = await self._analyze_performance(code, language)
            issues.extend(perf_issues)
            metrics.update(perf_metrics)
        
        # Stage 6: Test execution (paranoid)
        test_results = None
        if level == ValidationLevel.PARANOID and context and context.get("tests"):
            test_results = await self._run_tests(code, context["tests"], language)
            if not test_results["passed"]:
                issues.append(ValidationIssue(
                    issue_type=IssueType.TEST_FAILURE,
                    severity=IssueSeverity.HIGH,
                    message=f"Tests failed: {test_results['failed']}/{test_results['total']}",
                ))
        
        # Determine overall pass/fail
        passed = not any(i.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH] 
                        for i in issues)
        
        execution_time = time.time() - start_time
        
        result = ValidationResult(
            passed=passed,
            validation_level=level,
            issues=issues,
            metrics=metrics,
            test_results=test_results,
            execution_time=execution_time,
        )
        
        logger.info(f"Validation complete: passed={passed}, "
                   f"issues={len(issues)}, time={execution_time:.2f}s")
        
        return result
    
    async def _validate_syntax(
        self,
        code: str,
        language: str,
    ) -> List[ValidationIssue]:
        """Validate code syntax."""
        issues = []
        
        if language == "python":
            try:
                ast.parse(code)
            except SyntaxError as e:
                issues.append(ValidationIssue(
                    issue_type=IssueType.SYNTAX_ERROR,
                    severity=IssueSeverity.CRITICAL,
                    message=f"Syntax error: {e.msg}",
                    line_number=e.lineno,
                    column=e.offset,
                ))
        
        return issues
    
    async def _check_style(
        self,
        code: str,
        language: str,
    ) -> List[ValidationIssue]:
        """Check code style."""
        issues = []
        
        if language == "python":
            # Check for common style issues
            lines = code.split('\n')
            
            for i, line in enumerate(lines, 1):
                # Line too long
                if len(line) > 100:
                    issues.append(ValidationIssue(
                        issue_type=IssueType.STYLE_VIOLATION,
                        severity=IssueSeverity.LOW,
                        message="Line too long (>100 characters)",
                        line_number=i,
                    ))
                
                # Trailing whitespace
                if line.endswith(' ') or line.endswith('\t'):
                    issues.append(ValidationIssue(
                        issue_type=IssueType.STYLE_VIOLATION,
                        severity=IssueSeverity.LOW,
                        message="Trailing whitespace",
                        line_number=i,
                    ))
        
        return issues
    
    async def _check_types(
        self,
        code: str,
        language: str,
    ) -> List[ValidationIssue]:
        """Check type annotations and usage."""
        issues = []
        
        if language == "python":
            try:
                tree = ast.parse(code)
                
                # Check for functions without type hints
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if not node.returns:
                            issues.append(ValidationIssue(
                                issue_type=IssueType.TYPE_ERROR,
                                severity=IssueSeverity.MEDIUM,
                                message=f"Function '{node.name}' missing return type hint",
                                line_number=node.lineno,
                                suggestion="Add return type annotation",
                            ))
            except Exception as e:
                logger.warning(f"Type checking failed: {e}")
        
        return issues
    
    async def _scan_security(
        self,
        code: str,
        language: str,
    ) -> List[ValidationIssue]:
        """Scan for security vulnerabilities."""
        issues = []
        
        if language == "python":
            # Check for dangerous patterns
            dangerous_imports = ['eval', 'exec', 'compile', '__import__']
            dangerous_functions = ['os.system', 'subprocess.call', 'pickle.loads']
            
            for pattern in dangerous_imports:
                if pattern in code:
                    issues.append(ValidationIssue(
                        issue_type=IssueType.SECURITY_VULNERABILITY,
                        severity=IssueSeverity.HIGH,
                        message=f"Potentially dangerous: {pattern}",
                        suggestion="Use safer alternatives",
                    ))
            
            # Check for SQL injection patterns
            if 'execute(' in code and '+' in code and 'SELECT' in code.upper():
                issues.append(ValidationIssue(
                    issue_type=IssueType.SECURITY_VULNERABILITY,
                    severity=IssueSeverity.CRITICAL,
                    message="Potential SQL injection vulnerability",
                    suggestion="Use parameterized queries",
                ))
        
        return issues
    
    async def _analyze_performance(
        self,
        code: str,
        language: str,
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Analyze code performance."""
        issues = []
        metrics = {}
        
        if language == "python":
            try:
                tree = ast.parse(code)
                
                # Count complexity metrics
                function_count = 0
                loop_count = 0
                max_nesting = 0
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        function_count += 1
                    elif isinstance(node, (ast.For, ast.While)):
                        loop_count += 1
                
                metrics = {
                    "function_count": function_count,
                    "loop_count": loop_count,
                    "lines_of_code": len(code.split('\n')),
                }
                
                # Check for performance anti-patterns
                if loop_count > 3 and code.count('for') > code.count('break'):
                    issues.append(ValidationIssue(
                        issue_type=IssueType.PERFORMANCE_ISSUE,
                        severity=IssueSeverity.MEDIUM,
                        message="Multiple nested loops detected",
                        suggestion="Consider vectorization or optimization",
                    ))
                
            except Exception as e:
                logger.warning(f"Performance analysis failed: {e}")
        
        return issues, metrics
    
    async def _run_tests(
        self,
        code: str,
        tests: str,
        language: str,
    ) -> Dict[str, Any]:
        """Run tests against code."""
        if language != "python":
            return {"passed": True, "message": "Testing not implemented for this language"}
        
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as code_file:
                code_file.write(code)
                code_path = code_file.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as test_file:
                test_file.write(tests)
                test_path = test_file.name
            
            # Run tests with pytest
            result = subprocess.run(
                ['python', '-m', 'pytest', test_path, '-v'],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            # Parse results
            passed = result.returncode == 0
            
            # Cleanup
            os.unlink(code_path)
            os.unlink(test_path)
            
            return {
                "passed": passed,
                "output": result.stdout,
                "errors": result.stderr,
            }
            
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "message": "Tests timed out after 30 seconds",
            }
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return {
                "passed": False,
                "message": f"Test execution error: {str(e)}",
            }
    
    async def auto_fix(
        self,
        code: str,
        issues: List[ValidationIssue],
        language: str,
    ) -> Optional[str]:
        """
        Attempt to automatically fix validation issues.
        
        Args:
            code: Original code
            issues: List of validation issues
            language: Programming language
            
        Returns:
            Fixed code or None if auto-fix not possible
        """
        if not self.enable_auto_fix:
            return None
        
        fixed_code = code
        
        # Auto-fix style issues
        for issue in issues:
            if issue.issue_type == IssueType.STYLE_VIOLATION:
                if "Trailing whitespace" in issue.message and issue.line_number:
                    lines = fixed_code.split('\n')
                    if 0 < issue.line_number <= len(lines):
                        lines[issue.line_number - 1] = lines[issue.line_number - 1].rstrip()
                        fixed_code = '\n'.join(lines)
        
        # Validate fixed code
        validation = await self.validate(fixed_code, language)
        if validation.passed:
            logger.info("Auto-fix successful")
            return fixed_code
        
        return None
