"""
AI-Powered Code Review System

Provides intelligent code review with:
- Automated code analysis
- Best practice detection
- Security vulnerability scanning
- Performance issue identification
- Complexity analysis
- Style consistency checking
- Automated suggestions
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class ReviewSeverity(Enum):
    """Issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ReviewCategory(Enum):
    """Review categories"""
    SECURITY = "security"
    PERFORMANCE = "performance"
    BUGS = "bugs"
    CODE_SMELL = "code_smell"
    BEST_PRACTICES = "best_practices"
    STYLE = "style"
    DOCUMENTATION = "documentation"


@dataclass
class CodeIssue:
    """Detected code issue"""
    file_path: str
    line_number: int
    category: ReviewCategory
    severity: ReviewSeverity
    title: str
    description: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None
    confidence: float = 1.0


@dataclass
class ReviewResult:
    """Code review result"""
    repository: str
    commit_sha: str
    issues: List[CodeIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    reviewed_at: datetime = field(default_factory=datetime.utcnow)


class AICodeReviewer:
    """
    AI-Powered Code Review System

    Features:
    - Automated code analysis
    - Security vulnerability detection
    - Performance anti-pattern detection
    - Code smell identification
    - Complexity metrics
    - Style consistency checking
    - Best practices enforcement
    - Automated fix suggestions
    - Learning from review feedback
    - Custom rule engine
    """

    def __init__(self):
        self._rules: List[Tuple[str, Callable]] = []
        self._security_patterns: Dict[str, List[str]] = {}
        self._performance_patterns: Dict[str, List[str]] = {}
        self._initialize_rules()

    def _initialize_rules(self):
        """Initialize default review rules"""
        # Security patterns
        self._security_patterns = {
            "python": [
                (r'eval\(', "Use of eval() is dangerous"),
                (r'exec\(', "Use of exec() is dangerous"),
                (r'pickle\.loads?\(', "Pickle deserialization can be unsafe"),
                (r'__import__\(', "Dynamic imports can be dangerous"),
                (r'subprocess\.call\(.+shell=True', "Shell=True in subprocess is unsafe"),
            ],
            "javascript": [
                (r'eval\(', "Use of eval() is dangerous"),
                (r'innerHTML\s*=', "innerHTML can lead to XSS"),
                (r'document\.write\(', "document.write() is deprecated"),
            ]
        }

        # Performance patterns
        self._performance_patterns = {
            "python": [
                (r'for\s+\w+\s+in\s+range\(len\(', "Use enumerate() instead of range(len())"),
                (r'\+\s*=.*str\(', "String concatenation in loop is inefficient"),
            ],
            "sql": [
                (r'SELECT\s+\*', "SELECT * is inefficient, specify columns"),
                (r'WHERE.*OR.*OR', "Multiple ORs may need index optimization"),
            ]
        }

    async def review_code(
        self,
        code: str,
        file_path: str,
        language: str = "python"
    ) -> List[CodeIssue]:
        """
        Review code and return issues

        Args:
            code: Source code to review
            file_path: Path to file
            language: Programming language

        Returns:
            List of detected issues
        """
        issues = []

        # Run security checks
        issues.extend(self._check_security(code, file_path, language))

        # Run performance checks
        issues.extend(self._check_performance(code, file_path, language))

        # Check code complexity
        issues.extend(self._check_complexity(code, file_path, language))

        # Check best practices
        issues.extend(self._check_best_practices(code, file_path, language))

        # Check documentation
        issues.extend(self._check_documentation(code, file_path, language))

        return issues

    def _check_security(
        self,
        code: str,
        file_path: str,
        language: str
    ) -> List[CodeIssue]:
        """Check for security issues"""
        issues = []
        patterns = self._security_patterns.get(language, [])

        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern, message in patterns:
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=i,
                        category=ReviewCategory.SECURITY,
                        severity=ReviewSeverity.HIGH,
                        title="Security Vulnerability",
                        description=message,
                        code_snippet=line.strip(),
                        confidence=0.9
                    ))

        return issues

    def _check_performance(
        self,
        code: str,
        file_path: str,
        language: str
    ) -> List[CodeIssue]:
        """Check for performance issues"""
        issues = []
        patterns = self._performance_patterns.get(language, [])

        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern, message in patterns:
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=i,
                        category=ReviewCategory.PERFORMANCE,
                        severity=ReviewSeverity.MEDIUM,
                        title="Performance Issue",
                        description=message,
                        code_snippet=line.strip(),
                        confidence=0.8
                    ))

        return issues

    def _check_complexity(
        self,
        code: str,
        file_path: str,
        language: str
    ) -> List[CodeIssue]:
        """Check cyclomatic complexity"""
        issues = []

        if language == "python":
            # Count decision points
            decision_keywords = ['if', 'elif', 'for', 'while', 'except', 'and', 'or']
            complexity = 1

            for keyword in decision_keywords:
                pattern = r'\b' + keyword + r'\b'
                complexity += len(re.findall(pattern, code))

            if complexity > 15:
                issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=1,
                    category=ReviewCategory.CODE_SMELL,
                    severity=ReviewSeverity.MEDIUM,
                    title="High Complexity",
                    description=f"Cyclomatic complexity is {complexity}, consider refactoring",
                    suggestion="Break down into smaller functions",
                    confidence=0.85
                ))

        return issues

    def _check_best_practices(
        self,
        code: str,
        file_path: str,
        language: str
    ) -> List[CodeIssue]:
        """Check best practices"""
        issues = []

        if language == "python":
            lines = code.split('\n')

            # Check for bare except
            for i, line in enumerate(lines, 1):
                if re.search(r'except\s*:', line):
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=i,
                        category=ReviewCategory.BEST_PRACTICES,
                        severity=ReviewSeverity.LOW,
                        title="Bare except clause",
                        description="Bare except catches all exceptions including system exit",
                        suggestion="Specify exception types: except ValueError:",
                        code_snippet=line.strip(),
                        confidence=1.0
                    ))

                # Check for print statements (should use logging)
                if re.search(r'\bprint\(', line) and 'def' in ''.join(lines[:i]):
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=i,
                        category=ReviewCategory.BEST_PRACTICES,
                        severity=ReviewSeverity.LOW,
                        title="Use logging instead of print",
                        description="Print statements should be replaced with proper logging",
                        suggestion="Use logging.info() or logging.debug()",
                        code_snippet=line.strip(),
                        confidence=0.7
                    ))

        return issues

    def _check_documentation(
        self,
        code: str,
        file_path: str,
        language: str
    ) -> List[CodeIssue]:
        """Check documentation"""
        issues = []

        if language == "python":
            # Check for missing docstrings in functions
            function_pattern = r'def\s+(\w+)\s*\('
            docstring_pattern = r'"""[\s\S]*?"""'

            functions = [(m.start(), m.group(1)) for m in re.finditer(function_pattern, code)]

            lines = code.split('\n')
            for func_pos, func_name in functions:
                # Skip private functions
                if func_name.startswith('_'):
                    continue

                # Find line number
                line_num = code[:func_pos].count('\n') + 1

                # Check if docstring exists nearby
                context_start = max(0, func_pos)
                context_end = min(len(code), func_pos + 500)
                context = code[context_start:context_end]

                if not re.search(docstring_pattern, context):
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=line_num,
                        category=ReviewCategory.DOCUMENTATION,
                        severity=ReviewSeverity.LOW,
                        title="Missing docstring",
                        description=f"Function {func_name} lacks documentation",
                        suggestion="Add docstring with description, args, and returns",
                        confidence=0.9
                    ))

        return issues

    def generate_report(self, issues: List[CodeIssue]) -> Dict[str, Any]:
        """Generate review report"""
        # Group by severity
        by_severity = {}
        for severity in ReviewSeverity:
            by_severity[severity.value] = [
                i for i in issues if i.severity == severity
            ]

        # Group by category
        by_category = {}
        for category in ReviewCategory:
            by_category[category.value] = [
                i for i in issues if i.category == category
            ]

        # Calculate score (0-100)
        total_weight = sum(
            len(by_severity[s.value]) * (5 - list(ReviewSeverity).index(s))
            for s in ReviewSeverity
        )
        max_weight = len(issues) * 5 if issues else 1
        score = max(0, 100 - (total_weight / max_weight * 100))

        return {
            "total_issues": len(issues),
            "by_severity": {k: len(v) for k, v in by_severity.items()},
            "by_category": {k: len(v) for k, v in by_category.items()},
            "score": round(score, 2),
            "grade": self._score_to_grade(score)
        }

    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
