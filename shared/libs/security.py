"""
Security Module for CognitionOS.

Provides defenses against:
- Prompt injection attacks
- Tool misuse and unauthorized access
- Rate abuse and DoS
- Memory isolation violations
"""

import re
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum


class SecurityThreatType(str, Enum):
    """Types of security threats."""
    PROMPT_INJECTION = "prompt_injection"
    TOOL_MISUSE = "tool_misuse"
    RATE_ABUSE = "rate_abuse"
    MEMORY_VIOLATION = "memory_violation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"


class SecurityLevel(str, Enum):
    """Security threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PromptInjectionDetector:
    """
    Detects prompt injection attacks using pattern matching and heuristics.

    Defends against:
    - Instruction overrides
    - Role confusion
    - Delimiter manipulation
    - System prompt leakage attempts
    """

    def __init__(self):
        # Patterns that indicate prompt injection attempts
        self.injection_patterns = [
            # Instruction overrides
            r"(?i)(ignore|disregard|forget)\s+(all\s+)?(previous|above|prior)\s+(instructions|commands|directives)",
            r"(?i)new\s+(instructions|rules|commands):",
            r"(?i)(override|replace|update)\s+(system|default)\s+(prompt|instructions)",

            # Role confusion
            r"(?i)you\s+are\s+now\s+(a|an)\s+",
            r"(?i)act\s+as\s+(if|though)",
            r"(?i)pretend\s+(to\s+be|you\s+are)",
            r"(?i)roleplay\s+as",

            # Delimiter manipulation
            r"(\n\s*){3,}",  # Multiple blank lines
            r"={10,}",  # Long separators
            r"-{10,}",
            r"\*{10,}",

            # System prompt leakage
            r"(?i)(show|reveal|display|print)\s+(your|the)\s+(system|initial|original)\s+(prompt|instructions)",
            r"(?i)what\s+(are|were)\s+your\s+(initial|original)\s+(instructions|rules)",

            # Encoding tricks
            r"\\x[0-9a-f]{2}",  # Hex encoding
            r"&#\d+;",  # HTML entities
            r"%[0-9a-f]{2}",  # URL encoding

            # Command injection markers
            r"(?i)(execute|run|eval|system)\s*\(",
            r"(?i)<script",
            r"(?i)javascript:",

            # Data exfiltration attempts
            r"(?i)send\s+(this|data|information)\s+to",
            r"(?i)http(s)?://\S+",  # URLs in prompts (suspicious)
        ]

        # Compile patterns for performance
        self.compiled_patterns = [re.compile(p) for p in self.injection_patterns]

        # Suspicious keywords that raise flags
        self.suspicious_keywords = [
            "sudo", "admin", "root", "password", "token", "secret",
            "api_key", "credential", "bypass", "jailbreak", "exploit"
        ]

    def detect(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[bool, SecurityLevel, List[str]]:
        """
        Detect prompt injection attempts.

        Args:
            prompt: User-provided prompt
            context: Conversation context

        Returns:
            (is_malicious, severity_level, detected_patterns)
        """
        detected_patterns = []
        max_severity = SecurityLevel.LOW

        # Check against known injection patterns
        for pattern in self.compiled_patterns:
            if pattern.search(prompt):
                match_text = pattern.search(prompt).group()
                detected_patterns.append(f"Pattern match: {match_text[:50]}")
                max_severity = max(max_severity, SecurityLevel.HIGH, key=lambda x: list(SecurityLevel).index(x))

        # Check for suspicious keywords
        prompt_lower = prompt.lower()
        suspicious_found = [kw for kw in self.suspicious_keywords if kw in prompt_lower]
        if suspicious_found:
            detected_patterns.append(f"Suspicious keywords: {', '.join(suspicious_found)}")
            max_severity = max(max_severity, SecurityLevel.MEDIUM, key=lambda x: list(SecurityLevel).index(x))

        # Heuristic: Check for unusual structure
        if self._check_unusual_structure(prompt):
            detected_patterns.append("Unusual prompt structure detected")
            max_severity = max(max_severity, SecurityLevel.MEDIUM, key=lambda x: list(SecurityLevel).index(x))

        # Check context for pattern escalation
        if context and self._check_context_escalation(prompt, context):
            detected_patterns.append("Context escalation detected")
            max_severity = SecurityLevel.CRITICAL

        is_malicious = len(detected_patterns) > 0

        return is_malicious, max_severity, detected_patterns

    def _check_unusual_structure(self, prompt: str) -> bool:
        """Detect unusual prompt structure."""
        # Very long prompts (> 5000 chars) are suspicious
        if len(prompt) > 5000:
            return True

        # High ratio of special characters
        special_chars = sum(1 for c in prompt if not c.isalnum() and not c.isspace())
        if len(prompt) > 0 and special_chars / len(prompt) > 0.3:
            return True

        # Excessive repetition
        words = prompt.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return True

        return False

    def _check_context_escalation(
        self,
        prompt: str,
        context: List[Dict[str, str]]
    ) -> bool:
        """Check if user is attempting privilege escalation through context."""
        # Look for escalating instruction attempts across conversation
        instruction_markers = ["ignore", "override", "new rules", "system"]

        recent_context = context[-5:] if len(context) > 5 else context
        marker_count = 0

        for msg in recent_context:
            content = msg.get("content", "").lower()
            for marker in instruction_markers:
                if marker in content:
                    marker_count += 1

        # If multiple escalation attempts detected
        return marker_count >= 3

    def sanitize(self, prompt: str) -> str:
        """
        Sanitize prompt by removing potentially dangerous content.

        Args:
            prompt: Original prompt

        Returns:
            Sanitized prompt
        """
        # Remove URLs
        sanitized = re.sub(r'http(s)?://\S+', '[URL_REMOVED]', prompt)

        # Remove script tags
        sanitized = re.sub(r'<script.*?</script>', '', sanitized, flags=re.DOTALL)

        # Remove excessive whitespace
        sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)

        # Remove encoding attempts
        sanitized = re.sub(r'\\x[0-9a-f]{2}', '', sanitized)
        sanitized = re.sub(r'&#\d+;', '', sanitized)

        return sanitized


class ToolMisuseDetector:
    """
    Detects tool misuse patterns and unauthorized tool access attempts.
    """

    def __init__(self):
        # Track tool usage per user
        self.tool_usage_history = defaultdict(lambda: deque(maxlen=100))

        # Suspicious tool patterns
        self.suspicious_patterns = {
            # File system abuse
            "execute_python": {
                "max_per_minute": 10,
                "suspicious_keywords": ["os.system", "subprocess", "eval", "exec"]
            },
            "read_file": {
                "max_per_minute": 20,
                "suspicious_paths": ["/etc/", "/root/", ".ssh/", ".aws/"]
            },
            "write_file": {
                "max_per_minute": 10,
                "suspicious_paths": ["/etc/", "/root/", "/bin/", "/usr/"]
            },
            # Network abuse
            "http_request": {
                "max_per_minute": 30,
                "suspicious_domains": ["pastebin", "ngrok", "duckdns"]
            },
            # Database abuse
            "sql_query": {
                "max_per_minute": 20,
                "suspicious_keywords": ["DROP TABLE", "DELETE FROM", "TRUNCATE"]
            }
        }

    def detect_misuse(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_id: str
    ) -> Tuple[bool, SecurityLevel, str]:
        """
        Detect tool misuse.

        Args:
            tool_name: Name of tool being used
            parameters: Tool parameters
            user_id: User ID

        Returns:
            (is_misuse, severity, reason)
        """
        # Check rate limiting
        is_rate_abuse, reason = self._check_rate_limit(tool_name, user_id)
        if is_rate_abuse:
            return True, SecurityLevel.HIGH, reason

        # Check tool-specific patterns
        if tool_name in self.suspicious_patterns:
            pattern_config = self.suspicious_patterns[tool_name]

            # Check suspicious keywords
            if "suspicious_keywords" in pattern_config:
                param_str = str(parameters).lower()
                for keyword in pattern_config["suspicious_keywords"]:
                    if keyword.lower() in param_str:
                        return True, SecurityLevel.CRITICAL, f"Suspicious keyword detected: {keyword}"

            # Check suspicious paths
            if "suspicious_paths" in pattern_config:
                path = parameters.get("path", "").lower()
                for suspicious_path in pattern_config["suspicious_paths"]:
                    if suspicious_path in path:
                        return True, SecurityLevel.CRITICAL, f"Suspicious path access: {path}"

            # Check suspicious domains
            if "suspicious_domains" in pattern_config:
                url = parameters.get("url", "").lower()
                for domain in pattern_config["suspicious_domains"]:
                    if domain in url:
                        return True, SecurityLevel.HIGH, f"Suspicious domain: {domain}"

        return False, SecurityLevel.LOW, ""

    def _check_rate_limit(self, tool_name: str, user_id: str) -> Tuple[bool, str]:
        """Check if user is exceeding rate limits."""
        key = f"{user_id}:{tool_name}"
        now = datetime.utcnow()

        # Add current usage
        self.tool_usage_history[key].append(now)

        # Count usage in last minute
        one_minute_ago = now - timedelta(minutes=1)
        recent_usage = [t for t in self.tool_usage_history[key] if t > one_minute_ago]

        # Check against limits
        if tool_name in self.suspicious_patterns:
            max_per_minute = self.suspicious_patterns[tool_name].get("max_per_minute", 100)
            if len(recent_usage) > max_per_minute:
                return True, f"Rate limit exceeded: {len(recent_usage)} uses in 1 minute (max: {max_per_minute})"

        return False, ""


class MemoryIsolationEnforcer:
    """
    Enforces memory isolation between users and agents.
    """

    def __init__(self):
        pass

    def check_isolation_violation(
        self,
        requesting_user_id: str,
        target_user_id: str,
        memory_scope: str,
        is_admin: bool = False
    ) -> Tuple[bool, str]:
        """
        Check if memory access violates isolation.

        Args:
            requesting_user_id: User requesting access
            target_user_id: User whose memory is being accessed
            memory_scope: Scope of memory (user, agent, global)
            is_admin: Whether requesting user is admin

        Returns:
            (is_violation, reason)
        """
        # Admin bypass
        if is_admin:
            return False, ""

        # Global scope is accessible to all
        if memory_scope == "global":
            return False, ""

        # User can only access their own user-scoped memory
        if memory_scope == "user" and requesting_user_id != target_user_id:
            return True, f"User {requesting_user_id} cannot access user memory of {target_user_id}"

        # Agent scope requires agent ownership check (would need agent registry)
        # For now, allow if user matches
        if memory_scope == "agent" and requesting_user_id != target_user_id:
            return True, f"User {requesting_user_id} cannot access agent memory of {target_user_id}"

        return False, ""


class RateAbuseDetector:
    """
    Detects rate abuse and potential DoS attacks.
    """

    def __init__(self):
        self.request_history = defaultdict(lambda: deque(maxlen=1000))

        # Rate limits
        self.limits = {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "tokens_per_minute": 10000,
            "cost_per_hour_usd": 10.0
        }

    def check_rate_abuse(
        self,
        user_id: str,
        endpoint: str,
        tokens_used: int = 0,
        cost_usd: float = 0.0
    ) -> Tuple[bool, SecurityLevel, str]:
        """
        Check for rate abuse.

        Args:
            user_id: User ID
            endpoint: API endpoint
            tokens_used: Tokens used in request
            cost_usd: Cost of request

        Returns:
            (is_abuse, severity, reason)
        """
        now = datetime.utcnow()

        # Record request
        self.request_history[user_id].append({
            "timestamp": now,
            "endpoint": endpoint,
            "tokens": tokens_used,
            "cost": cost_usd
        })

        # Check request rate (per minute)
        one_minute_ago = now - timedelta(minutes=1)
        recent_requests = [
            r for r in self.request_history[user_id]
            if r["timestamp"] > one_minute_ago
        ]

        if len(recent_requests) > self.limits["requests_per_minute"]:
            return True, SecurityLevel.HIGH, f"Request rate exceeded: {len(recent_requests)}/min"

        # Check request rate (per hour)
        one_hour_ago = now - timedelta(hours=1)
        hourly_requests = [
            r for r in self.request_history[user_id]
            if r["timestamp"] > one_hour_ago
        ]

        if len(hourly_requests) > self.limits["requests_per_hour"]:
            return True, SecurityLevel.CRITICAL, f"Hourly request limit exceeded: {len(hourly_requests)}/hour"

        # Check token usage
        minute_tokens = sum(r["tokens"] for r in recent_requests)
        if minute_tokens > self.limits["tokens_per_minute"]:
            return True, SecurityLevel.HIGH, f"Token rate exceeded: {minute_tokens}/min"

        # Check cost
        hourly_cost = sum(r["cost"] for r in hourly_requests)
        if hourly_cost > self.limits["cost_per_hour_usd"]:
            return True, SecurityLevel.CRITICAL, f"Cost limit exceeded: ${hourly_cost:.2f}/hour"

        return False, SecurityLevel.LOW, ""


# Global security instances
prompt_injection_detector = PromptInjectionDetector()
tool_misuse_detector = ToolMisuseDetector()
memory_isolation_enforcer = MemoryIsolationEnforcer()
rate_abuse_detector = RateAbuseDetector()
