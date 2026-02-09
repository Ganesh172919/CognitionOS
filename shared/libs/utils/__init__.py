"""
Common utility functions for CognitionOS.
"""

import hashlib
import secrets
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


def generate_id() -> UUID:
    """Generate a new UUID."""
    return uuid4()


def generate_random_string(length: int = 32) -> str:
    """
    Generate a cryptographically secure random string.

    Args:
        length: Length of the string

    Returns:
        Random string
    """
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def generate_token(length: int = 32) -> str:
    """
    Generate a secure random token.

    Args:
        length: Length in bytes (will be hex-encoded, so actual length is 2x)

    Returns:
        Hex-encoded token
    """
    return secrets.token_hex(length)


def hash_string(value: str, salt: Optional[str] = None) -> str:
    """
    Hash a string using SHA-256.

    Args:
        value: String to hash
        salt: Optional salt to add

    Returns:
        Hex-encoded hash
    """
    if salt:
        value = f"{salt}{value}"
    return hashlib.sha256(value.encode()).hexdigest()


def normalize_email(email: str) -> str:
    """
    Normalize email address.

    Args:
        email: Email address

    Returns:
        Normalized email (lowercase, trimmed)
    """
    return email.strip().lower()


def parse_duration(duration_str: str) -> timedelta:
    """
    Parse duration string to timedelta.

    Args:
        duration_str: Duration string (e.g., "1h", "30m", "5d")

    Returns:
        timedelta object

    Raises:
        ValueError: If format is invalid
    """
    unit = duration_str[-1]
    value = int(duration_str[:-1])

    if unit == 's':
        return timedelta(seconds=value)
    elif unit == 'm':
        return timedelta(minutes=value)
    elif unit == 'h':
        return timedelta(hours=value)
    elif unit == 'd':
        return timedelta(days=value)
    else:
        raise ValueError(f"Invalid duration unit: {unit}")


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks.

    Args:
        lst: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator between keys

    Returns:
        Flattened dictionary

    Example:
        {"a": {"b": 1}} -> {"a.b": 1}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def safe_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Safely get nested dictionary value.

    Args:
        d: Dictionary to search
        path: Dot-separated path (e.g., "a.b.c")
        default: Default value if not found

    Returns:
        Value at path or default
    """
    keys = path.split('.')
    value = d
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
            if value is None:
                return default
        else:
            return default
    return value


def truncate_string(s: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate string to maximum length.

    Args:
        s: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def is_valid_uuid(value: str) -> bool:
    """
    Check if string is a valid UUID.

    Args:
        value: String to check

    Returns:
        True if valid UUID
    """
    try:
        UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def calculate_exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0
) -> float:
    """
    Calculate exponential backoff delay.

    Args:
        attempt: Attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Delay in seconds
    """
    delay = base_delay * (2 ** attempt)
    return min(delay, max_delay)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing dangerous characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove path traversal attempts
    filename = filename.replace('..', '')

    # Keep only alphanumeric, dash, underscore, and dot
    allowed_chars = string.ascii_letters + string.digits + '-_.'
    return ''.join(c for c in filename if c in allowed_chars)


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries (later ones override earlier).

    Args:
        *dicts: Dictionaries to merge

    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


class RateLimiter:
    """
    Simple in-memory rate limiter using token bucket algorithm.
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize rate limiter.

        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = datetime.utcnow()

    def allow(self, tokens: int = 1) -> bool:
        """
        Check if request is allowed.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if allowed, False if rate limited
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self):
        """Refill tokens based on time elapsed."""
        now = datetime.utcnow()
        elapsed = (now - self.last_refill).total_seconds()
        new_tokens = elapsed * self.refill_rate

        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    """

    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Time to wait before attempting reset
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    def call(self, func, *args, **kwargs):
        """
        Call function with circuit breaker protection.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open
        """
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.last_failure_time:
            return True

        elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout_seconds
