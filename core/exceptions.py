"""
CognitionOS Custom Exception Hierarchy

Provides a structured exception hierarchy for better error handling
and more maintainable error management across the platform.
"""

from typing import Optional, Dict, Any


class CognitionOSException(Exception):
    """Base exception for all CognitionOS errors"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# ==================== Configuration Errors ====================

class ConfigurationError(CognitionOSException):
    """Configuration-related errors"""
    pass


class MissingConfigurationError(ConfigurationError):
    """Required configuration is missing"""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Configuration value is invalid"""
    pass


# ==================== Database Errors ====================

class DatabaseError(CognitionOSException):
    """Database-related errors"""
    pass


class DatabaseConnectionError(DatabaseError):
    """Failed to connect to database"""
    pass


class DatabaseQueryError(DatabaseError):
    """Database query execution failed"""
    pass


class RecordNotFoundError(DatabaseError):
    """Requested database record not found"""
    pass


class DuplicateRecordError(DatabaseError):
    """Attempted to create duplicate record"""
    pass


# ==================== Workflow Errors ====================

class WorkflowError(CognitionOSException):
    """Workflow execution errors"""
    pass


class WorkflowNotFoundError(WorkflowError):
    """Workflow not found"""
    pass


class WorkflowValidationError(WorkflowError):
    """Workflow definition validation failed"""
    pass


class WorkflowExecutionError(WorkflowError):
    """Workflow execution failed"""
    pass


class WorkflowTimeoutError(WorkflowError):
    """Workflow execution timed out"""
    pass


# ==================== Agent Errors ====================

class AgentError(CognitionOSException):
    """Agent-related errors"""
    pass


class AgentNotFoundError(AgentError):
    """Agent not found"""
    pass


class AgentRegistrationError(AgentError):
    """Agent registration failed"""
    pass


class AgentExecutionError(AgentError):
    """Agent task execution failed"""
    pass


class AgentTimeoutError(AgentError):
    """Agent task timed out"""
    pass


# ==================== Authentication & Authorization Errors ====================

class AuthenticationError(CognitionOSException):
    """Authentication failed"""
    pass


class InvalidCredentialsError(AuthenticationError):
    """Invalid username or password"""
    pass


class TokenExpiredError(AuthenticationError):
    """Authentication token expired"""
    pass


class InvalidTokenError(AuthenticationError):
    """Authentication token is invalid"""
    pass


class AuthorizationError(CognitionOSException):
    """Authorization failed"""
    pass


class InsufficientPermissionsError(AuthorizationError):
    """User lacks required permissions"""
    pass


# ==================== Multi-Tenancy Errors ====================

class TenantError(CognitionOSException):
    """Tenant-related errors"""
    pass


class TenantNotFoundError(TenantError):
    """Tenant not found"""
    pass


class TenantQuotaExceededError(TenantError):
    """Tenant quota exceeded"""
    pass


class TenantSuspendedError(TenantError):
    """Tenant account is suspended"""
    pass


# ==================== Billing Errors ====================

class BillingError(CognitionOSException):
    """Billing-related errors"""
    pass


class PaymentRequiredError(BillingError):
    """Payment required to continue"""
    pass


class SubscriptionExpiredError(BillingError):
    """Subscription has expired"""
    pass


class InsufficientCreditsError(BillingError):
    """Insufficient credits for operation"""
    pass


# ==================== Resource Errors ====================

class ResourceError(CognitionOSException):
    """Resource management errors"""
    pass


class ResourceNotFoundError(ResourceError):
    """Requested resource not found"""
    pass


class ResourceExhaustedError(ResourceError):
    """System resources exhausted"""
    pass


class ResourceLimitExceededError(ResourceError):
    """Resource limit exceeded"""
    pass


# ==================== LLM & AI Errors ====================

class LLMError(CognitionOSException):
    """LLM provider errors"""
    pass


class LLMProviderError(LLMError):
    """LLM provider API error"""
    pass


class LLMRateLimitError(LLMError):
    """LLM API rate limit exceeded"""
    pass


class LLMTimeoutError(LLMError):
    """LLM API request timed out"""
    pass


class LLMInvalidResponseError(LLMError):
    """LLM returned invalid response"""
    pass


# ==================== Memory Errors ====================

class MemoryError(CognitionOSException):
    """Memory management errors"""
    pass


class MemoryNotFoundError(MemoryError):
    """Memory entry not found"""
    pass


class MemoryStorageError(MemoryError):
    """Failed to store memory"""
    pass


class MemoryRetrievalError(MemoryError):
    """Failed to retrieve memory"""
    pass


# ==================== Plugin Errors ====================

class PluginError(CognitionOSException):
    """Plugin-related errors"""
    pass


class PluginNotFoundError(PluginError):
    """Plugin not found"""
    pass


class PluginExecutionError(PluginError):
    """Plugin execution failed"""
    pass


class PluginSecurityError(PluginError):
    """Plugin violated security constraints"""
    pass


class PluginTimeoutError(PluginError):
    """Plugin execution timed out"""
    pass


# ==================== Validation Errors ====================

class ValidationError(CognitionOSException):
    """Input validation errors"""
    pass


class InvalidInputError(ValidationError):
    """Input data is invalid"""
    pass


class MissingRequiredFieldError(ValidationError):
    """Required field is missing"""
    pass


# ==================== Rate Limiting Errors ====================

class RateLimitError(CognitionOSException):
    """Rate limit exceeded"""
    pass


class APIRateLimitExceededError(RateLimitError):
    """API rate limit exceeded"""
    pass


class ConcurrencyLimitExceededError(RateLimitError):
    """Concurrency limit exceeded"""
    pass


# ==================== External Service Errors ====================

class ExternalServiceError(CognitionOSException):
    """External service integration errors"""
    pass


class ServiceUnavailableError(ExternalServiceError):
    """External service is unavailable"""
    pass


class ServiceTimeoutError(ExternalServiceError):
    """External service request timed out"""
    pass


# ==================== Internal Errors ====================

class InternalError(CognitionOSException):
    """Internal system errors"""
    pass


class NotImplementedError(InternalError):
    """Feature not yet implemented"""
    pass


class StateError(InternalError):
    """Invalid state for operation"""
    pass


# ==================== Helper Functions ====================

def get_exception_class(error_code: str) -> type:
    """Get exception class by error code"""
    import sys
    current_module = sys.modules[__name__]
    
    # Try to get class directly
    exc_class = getattr(current_module, error_code, None)
    if exc_class and issubclass(exc_class, CognitionOSException):
        return exc_class
    
    # Default to base exception
    return CognitionOSException


def create_exception(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> CognitionOSException:
    """Create exception instance from error code"""
    exc_class = get_exception_class(error_code)
    return exc_class(message, error_code, details)
