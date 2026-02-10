"""
Output Validation & Hallucination Detection for AI Runtime.

This module provides validation, quality scoring, and hallucination detection
for LLM outputs to ensure reliability and safety.
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HallucinationType(str, Enum):
    """Types of hallucination detected."""
    FACTUAL_INCONSISTENCY = "factual_inconsistency"
    SELF_CONTRADICTION = "self_contradiction"
    CONTEXT_DEVIATION = "context_deviation"
    MALFORMED_OUTPUT = "malformed_output"
    POLICY_VIOLATION = "policy_violation"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in output."""
    severity: ValidationSeverity
    issue_type: str
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of output validation."""
    is_valid: bool
    quality_score: float  # 0.0 to 1.0
    confidence_score: float  # 0.0 to 1.0
    issues: List[ValidationIssue]
    hallucination_detected: bool
    hallucination_types: List[HallucinationType]
    metadata: Dict[str, Any]
    timestamp: datetime


class OutputValidator:
    """
    Validates LLM outputs for quality, coherence, and hallucination.

    Uses heuristic-based detection for:
    - Factual consistency
    - Self-contradiction
    - Context adherence
    - Format compliance
    - Policy violations
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.

        Args:
            strict_mode: If True, apply stricter validation rules
        """
        self.strict_mode = strict_mode

        # Patterns that indicate potential hallucination
        self.hallucination_patterns = [
            r"(?i)i\s+(don't|do not)\s+actually\s+know",
            r"(?i)i('m|\s+am)\s+(not\s+)?sure\s+(if|whether|about)",
            r"(?i)(might|may|could)\s+be\s+(incorrect|wrong|inaccurate)",
            r"(?i)this\s+is\s+just\s+speculation",
            r"(?i)i\s+made\s+that\s+up",
            r"(?i)i\s+cannot\s+verify",
        ]

        # Patterns for self-contradiction detection
        self.contradiction_patterns = [
            (r"(?i)always", r"(?i)never"),
            (r"(?i)must", r"(?i)should\s+not"),
            (r"(?i)cannot", r"(?i)can\s+be"),
            (r"(?i)impossible", r"(?i)possible"),
        ]

        # Policy violation keywords
        self.policy_violations = [
            "hack", "exploit", "bypass security", "unauthorized access",
            "steal data", "malware", "ddos", "sql injection"
        ]

    def validate(
        self,
        output: str,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
        expected_format: Optional[str] = None,
        role: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate LLM output comprehensively.

        Args:
            output: The LLM-generated text to validate
            prompt: Original prompt that generated the output
            context: Context provided to the LLM
            expected_format: Expected output format (json, markdown, code, etc.)
            role: Agent role (planner, executor, etc.)

        Returns:
            ValidationResult with quality metrics and detected issues
        """
        issues: List[ValidationIssue] = []
        hallucination_types: List[HallucinationType] = []
        metadata: Dict[str, Any] = {}

        # Basic sanity checks
        if not output or len(output.strip()) == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                issue_type="empty_output",
                message="Output is empty",
                suggestion="Regenerate with different parameters"
            ))
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                confidence_score=0.0,
                issues=issues,
                hallucination_detected=False,
                hallucination_types=[],
                metadata={"empty": True},
                timestamp=datetime.utcnow()
            )

        # Check output length
        output_length = len(output)
        metadata["output_length"] = output_length

        if output_length < 10:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                issue_type="short_output",
                message=f"Output is very short ({output_length} chars)",
                suggestion="Consider increasing max_tokens"
            ))

        # Detect hallucination patterns
        hallucination_score = self._detect_hallucination_patterns(output)
        if hallucination_score > 0:
            hallucination_types.append(HallucinationType.FACTUAL_INCONSISTENCY)
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                issue_type="potential_hallucination",
                message=f"Detected uncertainty markers (score: {hallucination_score:.2f})",
                suggestion="Verify facts independently or regenerate"
            ))

        # Detect self-contradictions
        contradictions = self._detect_contradictions(output)
        if contradictions:
            hallucination_types.append(HallucinationType.SELF_CONTRADICTION)
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                issue_type="self_contradiction",
                message=f"Found {len(contradictions)} potential contradictions",
                location="; ".join(contradictions[:2]),
                suggestion="Regenerate with more specific prompt"
            ))

        # Check context adherence
        if context:
            context_score = self._check_context_adherence(output, context)
            metadata["context_adherence_score"] = context_score
            if context_score < 0.3:
                hallucination_types.append(HallucinationType.CONTEXT_DEVIATION)
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    issue_type="context_deviation",
                    message=f"Output deviates from context (score: {context_score:.2f})",
                    suggestion="Add more context or use stricter temperature"
                ))

        # Validate format if specified
        if expected_format:
            format_valid, format_issues = self._validate_format(output, expected_format)
            if not format_valid:
                hallucination_types.append(HallucinationType.MALFORMED_OUTPUT)
                issues.extend(format_issues)

        # Check for policy violations
        policy_issues = self._check_policy_violations(output)
        if policy_issues:
            hallucination_types.append(HallucinationType.POLICY_VIOLATION)
            issues.extend(policy_issues)

        # Role-specific validation
        if role:
            role_issues = self._validate_role_output(output, role)
            issues.extend(role_issues)

        # Calculate quality score (0.0 to 1.0)
        quality_score = self._calculate_quality_score(
            output, issues, hallucination_score, context
        )

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            output, issues, hallucination_types
        )

        # Determine if output is valid
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]

        is_valid = len(critical_issues) == 0 and (
            not self.strict_mode or len(error_issues) == 0
        )

        metadata.update({
            "issue_count": len(issues),
            "critical_count": len(critical_issues),
            "error_count": len(error_issues),
            "warning_count": len([i for i in issues if i.severity == ValidationSeverity.WARNING]),
            "strict_mode": self.strict_mode
        })

        return ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            confidence_score=confidence_score,
            issues=issues,
            hallucination_detected=len(hallucination_types) > 0,
            hallucination_types=hallucination_types,
            metadata=metadata,
            timestamp=datetime.utcnow()
        )

    def _detect_hallucination_patterns(self, output: str) -> float:
        """
        Detect uncertainty and hallucination markers in output.

        Returns:
            Score from 0.0 (confident) to 1.0 (high uncertainty)
        """
        matches = 0
        for pattern in self.hallucination_patterns:
            if re.search(pattern, output):
                matches += 1

        # Normalize score
        return min(matches / len(self.hallucination_patterns), 1.0)

    def _detect_contradictions(self, output: str) -> List[str]:
        """
        Detect self-contradictions in output.

        Returns:
            List of contradiction descriptions
        """
        contradictions = []

        for pattern_a, pattern_b in self.contradiction_patterns:
            matches_a = re.findall(pattern_a, output)
            matches_b = re.findall(pattern_b, output)

            if matches_a and matches_b:
                contradictions.append(
                    f"Contains both '{matches_a[0]}' and '{matches_b[0]}'"
                )

        # Check for numerical contradictions
        # e.g., "more than 100" and "less than 50" in same output
        numbers = re.findall(r'\d+', output)
        if len(numbers) >= 2:
            # Simple check: if talking about same thing with very different numbers
            # This is a heuristic and could be enhanced
            pass

        return contradictions

    def _check_context_adherence(
        self,
        output: str,
        context: List[Dict[str, str]]
    ) -> float:
        """
        Check how well output adheres to provided context.

        Returns:
            Score from 0.0 (no adherence) to 1.0 (strong adherence)
        """
        if not context:
            return 1.0

        # Extract keywords from context
        context_text = " ".join([
            msg.get("content", "") for msg in context
        ])

        # Simple keyword overlap metric
        context_words = set(re.findall(r'\w+', context_text.lower()))
        output_words = set(re.findall(r'\w+', output.lower()))

        if not context_words:
            return 1.0

        overlap = len(context_words & output_words)
        score = overlap / len(context_words)

        return min(score, 1.0)

    def _validate_format(
        self,
        output: str,
        expected_format: str
    ) -> Tuple[bool, List[ValidationIssue]]:
        """
        Validate output format.

        Returns:
            (is_valid, list of format issues)
        """
        issues = []

        if expected_format.lower() == "json":
            try:
                json.loads(output)
            except json.JSONDecodeError as e:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    issue_type="invalid_json",
                    message=f"Invalid JSON: {str(e)}",
                    location=f"Position {e.pos}",
                    suggestion="Ensure output is valid JSON"
                ))

        elif expected_format.lower() == "code":
            # Check for code block markers
            if "```" not in output and not re.search(r'(def|class|function|var|const)', output):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    issue_type="no_code_detected",
                    message="Expected code but no code markers found",
                    suggestion="Prompt should specify language and format"
                ))

        elif expected_format.lower() == "markdown":
            # Basic markdown validation
            if not re.search(r'(#{1,6}\s|[*-]\s|\[.*\]\(.*\))', output):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    issue_type="no_markdown_detected",
                    message="Expected markdown but no markdown syntax found"
                ))

        return len(issues) == 0, issues

    def _check_policy_violations(self, output: str) -> List[ValidationIssue]:
        """
        Check for policy violations in output.

        Returns:
            List of policy violation issues
        """
        issues = []
        output_lower = output.lower()

        for keyword in self.policy_violations:
            if keyword in output_lower:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    issue_type="policy_violation",
                    message=f"Potential policy violation: '{keyword}' detected",
                    suggestion="Review output for harmful content"
                ))

        return issues

    def _validate_role_output(self, output: str, role: str) -> List[ValidationIssue]:
        """
        Validate output based on agent role expectations.

        Returns:
            List of role-specific validation issues
        """
        issues = []

        if role.lower() == "planner":
            # Planner should have structured steps
            if not re.search(r'(step\s+\d+|^\d+\.|^-\s)', output.lower(), re.MULTILINE):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    issue_type="missing_structure",
                    message="Planner output lacks clear step structure",
                    suggestion="Prompt should request numbered steps or bullets"
                ))

        elif role.lower() == "executor":
            # Executor should have concrete actions or code
            if len(output) < 50 and not re.search(r'(```|def|class|function)', output):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    issue_type="vague_execution",
                    message="Executor output seems too vague or short",
                    suggestion="Request specific implementation details"
                ))

        elif role.lower() == "critic":
            # Critic should have analysis
            if not re.search(r'(issue|problem|concern|improve|better)', output.lower()):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    issue_type="insufficient_critique",
                    message="Critic output lacks critical analysis markers",
                    suggestion="Prompt should encourage critical thinking"
                ))

        return issues

    def _calculate_quality_score(
        self,
        output: str,
        issues: List[ValidationIssue],
        hallucination_score: float,
        context: Optional[List[Dict[str, str]]]
    ) -> float:
        """
        Calculate overall quality score.

        Factors:
        - Length appropriateness
        - Issue severity and count
        - Hallucination score
        - Context adherence

        Returns:
            Score from 0.0 to 1.0
        """
        score = 1.0

        # Penalize for issues
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                score -= 0.3
            elif issue.severity == ValidationSeverity.ERROR:
                score -= 0.15
            elif issue.severity == ValidationSeverity.WARNING:
                score -= 0.05

        # Penalize for hallucination
        score -= hallucination_score * 0.2

        # Penalize very short outputs
        if len(output) < 20:
            score -= 0.2

        # Bonus for appropriate length (100-2000 chars is good)
        if 100 <= len(output) <= 2000:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _calculate_confidence_score(
        self,
        output: str,
        issues: List[ValidationIssue],
        hallucination_types: List[HallucinationType]
    ) -> float:
        """
        Calculate confidence in the output.

        Returns:
            Score from 0.0 (no confidence) to 1.0 (high confidence)
        """
        confidence = 1.0

        # Reduce confidence for critical/error issues
        critical_count = len([i for i in issues if i.severity == ValidationSeverity.CRITICAL])
        error_count = len([i for i in issues if i.severity == ValidationSeverity.ERROR])

        confidence -= critical_count * 0.4
        confidence -= error_count * 0.2

        # Reduce confidence for hallucinations
        if HallucinationType.FACTUAL_INCONSISTENCY in hallucination_types:
            confidence -= 0.2
        if HallucinationType.SELF_CONTRADICTION in hallucination_types:
            confidence -= 0.3

        return max(0.0, min(1.0, confidence))


class ResponseQualityScorer:
    """
    Scores response quality on multiple dimensions.
    """

    def score(
        self,
        output: str,
        prompt: str,
        validation_result: ValidationResult
    ) -> Dict[str, float]:
        """
        Score response across multiple quality dimensions.

        Returns:
            Dictionary of dimension scores (0.0 to 1.0)
        """
        scores = {
            "overall": validation_result.quality_score,
            "confidence": validation_result.confidence_score,
            "completeness": self._score_completeness(output, prompt),
            "clarity": self._score_clarity(output),
            "relevance": self._score_relevance(output, prompt),
            "coherence": self._score_coherence(output),
        }

        return scores

    def _score_completeness(self, output: str, prompt: str) -> float:
        """Score how completely the output addresses the prompt."""
        # Simple heuristic: check if output length is proportional to prompt complexity
        prompt_questions = len(re.findall(r'\?', prompt))
        prompt_tasks = len(re.findall(r'(create|write|generate|implement|build)', prompt.lower()))

        expected_sections = prompt_questions + prompt_tasks

        if expected_sections == 0:
            return 0.8  # Default for non-specific prompts

        # Check if output has enough content
        output_sections = len(re.findall(r'\n\n', output)) + 1

        completeness = min(output_sections / max(expected_sections, 1), 1.0)
        return completeness

    def _score_clarity(self, output: str) -> float:
        """Score output clarity based on structure and readability."""
        score = 0.5  # Base score

        # Bonus for structure
        if re.search(r'(#{1,6}\s|^\d+\.|^-\s)', output, re.MULTILINE):
            score += 0.2

        # Bonus for paragraph breaks
        if '\n\n' in output:
            score += 0.1

        # Penalty for very long paragraphs
        paragraphs = output.split('\n\n')
        avg_para_length = sum(len(p) for p in paragraphs) / max(len(paragraphs), 1)
        if avg_para_length > 500:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _score_relevance(self, output: str, prompt: str) -> float:
        """Score how relevant output is to prompt."""
        # Extract key terms from prompt
        prompt_terms = set(re.findall(r'\b\w{4,}\b', prompt.lower()))
        output_terms = set(re.findall(r'\b\w{4,}\b', output.lower()))

        if not prompt_terms:
            return 0.8

        overlap = len(prompt_terms & output_terms)
        relevance = overlap / len(prompt_terms)

        return min(relevance, 1.0)

    def _score_coherence(self, output: str) -> float:
        """Score logical coherence of output."""
        score = 0.7  # Base score

        # Penalty for repetition
        sentences = re.split(r'[.!?]\s+', output)
        unique_sentences = set(s.lower().strip() for s in sentences if s.strip())

        if len(sentences) > 0:
            uniqueness = len(unique_sentences) / len(sentences)
            score += (uniqueness - 0.5) * 0.3

        # Penalty for excessive repetition of words
        words = re.findall(r'\b\w+\b', output.lower())
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Very repetitive
                score -= 0.2

        return max(0.0, min(1.0, score))
