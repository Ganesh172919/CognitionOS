"""
Input Validation and Sanitization

Provides utilities for validating and sanitizing user inputs to prevent
injection attacks and ensure data quality.
"""

import re
from typing import Any, Dict, List, Optional, Union
from html import escape
from urllib.parse import quote

# Regex patterns for common validations
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
SLUG_PATTERN = re.compile(r'^[a-z0-9-]+$')
ALPHANUMERIC_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')

# Dangerous patterns to detect potential injection attacks
SQL_INJECTION_PATTERNS = [
    r"(\bunion\b|\bselect\b|\binsert\b|\bupdate\b|\bdelete\b|\bdrop\b|\btruncate\b)",
    r"(--|;|\/\*|\*\/|xp_|sp_|exec|execute)",
]

XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"onerror\s*=",
    r"onload\s*=",
]


class ValidationError(Exception):
    """Exception raised for validation failures"""
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


def sanitize_html(text: str) -> str:
    """
    Sanitize HTML content to prevent XSS attacks.
    
    Args:
        text: Raw HTML text
        
    Returns:
        Sanitized text with HTML entities escaped
    """
    if not text:
        return ""
    
    return escape(text, quote=True)


def sanitize_sql_string(text: str) -> str:
    """
    Sanitize string for SQL queries (use parameterized queries instead when possible).
    
    Args:
        text: Input string
        
    Returns:
        Sanitized string
        
    Note:
        This is a defense-in-depth measure. Always use parameterized queries.
    """
    if not text:
        return ""
    
    # Remove common SQL injection patterns
    sanitized = text.replace("--", "").replace(";", "").replace("/*", "").replace("*/", "")
    
    return sanitized


def sanitize_url(url: str) -> str:
    """
    Sanitize URL to prevent injection attacks.
    
    Args:
        url: URL string
        
    Returns:
        Sanitized URL
    """
    if not url:
        return ""
    
    # Only allow http/https schemes
    if not url.startswith(("http://", "https://")):
        return ""
    
    # URL encode special characters
    return quote(url, safe=":/?#[]@!$&'()*+,;=")


def validate_email(email: str, field_name: str = "email") -> str:
    """
    Validate email format.
    
    Args:
        email: Email address
        field_name: Field name for error messages
        
    Returns:
        Normalized email (lowercase)
        
    Raises:
        ValidationError: If email is invalid
    """
    if not email:
        raise ValidationError(field_name, "Email is required")
    
    email = email.strip().lower()
    
    if len(email) > 255:
        raise ValidationError(field_name, "Email is too long (max 255 characters)")
    
    if not EMAIL_PATTERN.match(email):
        raise ValidationError(field_name, "Invalid email format")
    
    return email


def validate_uuid(uuid_str: str, field_name: str = "id") -> str:
    """
    Validate UUID format.
    
    Args:
        uuid_str: UUID string
        field_name: Field name for error messages
        
    Returns:
        Normalized UUID (lowercase)
        
    Raises:
        ValidationError: If UUID is invalid
    """
    if not uuid_str:
        raise ValidationError(field_name, "UUID is required")
    
    uuid_str = uuid_str.strip().lower()
    
    if not UUID_PATTERN.match(uuid_str):
        raise ValidationError(field_name, "Invalid UUID format")
    
    return uuid_str


def validate_slug(slug: str, field_name: str = "slug", max_length: int = 100) -> str:
    """
    Validate slug format (lowercase alphanumeric with hyphens).
    
    Args:
        slug: Slug string
        field_name: Field name for error messages
        max_length: Maximum allowed length
        
    Returns:
        Validated slug
        
    Raises:
        ValidationError: If slug is invalid
    """
    if not slug:
        raise ValidationError(field_name, "Slug is required")
    
    slug = slug.strip().lower()
    
    if len(slug) > max_length:
        raise ValidationError(field_name, f"Slug is too long (max {max_length} characters)")
    
    if not SLUG_PATTERN.match(slug):
        raise ValidationError(field_name, "Slug must contain only lowercase letters, numbers, and hyphens")
    
    return slug


def validate_alphanumeric(value: str, field_name: str = "value", max_length: int = 255) -> str:
    """
    Validate alphanumeric string with underscores and hyphens.
    
    Args:
        value: Input string
        field_name: Field name for error messages
        max_length: Maximum allowed length
        
    Returns:
        Validated string
        
    Raises:
        ValidationError: If string is invalid
    """
    if not value:
        raise ValidationError(field_name, f"{field_name} is required")
    
    value = value.strip()
    
    if len(value) > max_length:
        raise ValidationError(field_name, f"{field_name} is too long (max {max_length} characters)")
    
    if not ALPHANUMERIC_PATTERN.match(value):
        raise ValidationError(
            field_name,
            f"{field_name} must contain only letters, numbers, underscores, and hyphens"
        )
    
    return value


def detect_sql_injection(text: str) -> bool:
    """
    Detect potential SQL injection attempts.
    
    Args:
        text: Input text
        
    Returns:
        True if potential SQL injection detected
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False


def detect_xss(text: str) -> bool:
    """
    Detect potential XSS attempts.
    
    Args:
        text: Input text
        
    Returns:
        True if potential XSS detected
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    for pattern in XSS_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False


def sanitize_dict(data: Dict[str, Any], html_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Recursively sanitize dictionary values.
    
    Args:
        data: Input dictionary
        html_fields: List of fields that should have HTML sanitization
        
    Returns:
        Sanitized dictionary
    """
    html_fields = html_fields or []
    sanitized = {}
    
    for key, value in data.items():
        if isinstance(value, str):
            # Check for injection attempts
            if detect_sql_injection(value):
                raise ValidationError(key, "Potential SQL injection detected")
            
            if detect_xss(value):
                raise ValidationError(key, "Potential XSS attack detected")
            
            # Sanitize HTML fields
            if key in html_fields:
                sanitized[key] = sanitize_html(value)
            else:
                sanitized[key] = value.strip()
                
        elif isinstance(value, dict):
            sanitized[key] = sanitize_dict(value, html_fields)
            
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_dict(item, html_fields) if isinstance(item, dict)
                else sanitize_html(item) if isinstance(item, str) and key in html_fields
                else item
                for item in value
            ]
        else:
            sanitized[key] = value
    
    return sanitized


def validate_string_length(
    value: str,
    field_name: str,
    min_length: int = 0,
    max_length: int = 1000
) -> None:
    """
    Validate string length.
    
    Args:
        value: String value
        field_name: Field name for error messages
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        
    Raises:
        ValidationError: If length is invalid
    """
    if not value and min_length > 0:
        raise ValidationError(field_name, f"{field_name} is required")
    
    if value and len(value) < min_length:
        raise ValidationError(field_name, f"{field_name} must be at least {min_length} characters")
    
    if value and len(value) > max_length:
        raise ValidationError(field_name, f"{field_name} must not exceed {max_length} characters")


def validate_integer_range(
    value: int,
    field_name: str,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None
) -> None:
    """
    Validate integer is within range.
    
    Args:
        value: Integer value
        field_name: Field name for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Raises:
        ValidationError: If value is out of range
    """
    if min_value is not None and value < min_value:
        raise ValidationError(field_name, f"{field_name} must be at least {min_value}")
    
    if max_value is not None and value > max_value:
        raise ValidationError(field_name, f"{field_name} must not exceed {max_value}")
