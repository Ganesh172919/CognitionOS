"""
Retry Policy Implementation

Provides retry logic with exponential backoff for handling transient failures.
"""

import asyncio
import logging
from typing import Callable, TypeVar, Optional, Type, Tuple
from functools import wraps

from core.exceptions import ServiceUnavailableError, ExternalServiceError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryPolicy:
    """
    Retry policy with exponential backoff.
    
    Features:
    - Configurable retry attempts
    - Exponential backoff with jitter
    - Selective exception retry
    - Circuit breaker integration
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ):
        """
        Initialize retry policy.
        
        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before first retry
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff calculation
            jitter: Add random jitter to prevent thundering herd
            retry_exceptions: Tuple of exception types to retry on (None = all)
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_exceptions = retry_exceptions or (Exception,)
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt with exponential backoff.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
        
        return delay
    
    def _should_retry(self, exception: Exception) -> bool:
        """
        Determine if exception should trigger a retry.
        
        Args:
            exception: Exception that occurred
            
        Returns:
            True if should retry, False otherwise
        """
        return isinstance(exception, self.retry_exceptions)
    
    async def execute_async(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """
        Execute async function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                result = await func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(
                        f"Retry successful after {attempt} attempts for {func.__name__}"
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if not self._should_retry(e):
                    logger.warning(
                        f"Non-retryable exception in {func.__name__}: {type(e).__name__}"
                    )
                    raise
                
                if attempt < self.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_attempts} failed for {func.__name__}: "
                        f"{type(e).__name__}. Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_attempts} attempts exhausted for {func.__name__}",
                        exc_info=True
                    )
        
        # All retries exhausted
        raise ServiceUnavailableError(
            f"Operation failed after {self.max_attempts} attempts",
            details={
                "function": func.__name__,
                "last_error": str(last_exception),
                "attempts": self.max_attempts,
            }
        )


def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
):
    """
    Decorator to add retry logic to async functions.
    
    Usage:
        @with_retry(max_attempts=5, initial_delay=2.0)
        async def fetch_data():
            # Implementation
            pass
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff calculation
        jitter: Add random jitter to prevent thundering herd
        retry_exceptions: Tuple of exception types to retry on
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        policy = RetryPolicy(
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
            retry_exceptions=retry_exceptions,
        )
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await policy.execute_async(func, *args, **kwargs)
        
        return wrapper
    
    return decorator


# Pre-configured retry policies for common scenarios

# Fast retry for quick operations (network calls, cache)
FAST_RETRY = RetryPolicy(
    max_attempts=3,
    initial_delay=0.5,
    max_delay=5.0,
    exponential_base=2.0,
)

# Standard retry for normal operations
STANDARD_RETRY = RetryPolicy(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
)

# Aggressive retry for critical operations
AGGRESSIVE_RETRY = RetryPolicy(
    max_attempts=5,
    initial_delay=2.0,
    max_delay=60.0,
    exponential_base=2.0,
)

# Database retry - only retry on specific exceptions
DATABASE_RETRY = RetryPolicy(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=10.0,
    exponential_base=2.0,
    retry_exceptions=(
        asyncio.TimeoutError,
        ConnectionError,
        # Add database-specific exceptions here
    ),
)

# External service retry
EXTERNAL_SERVICE_RETRY = RetryPolicy(
    max_attempts=4,
    initial_delay=2.0,
    max_delay=30.0,
    exponential_base=2.0,
    retry_exceptions=(
        ExternalServiceError,
        ServiceUnavailableError,
        asyncio.TimeoutError,
        ConnectionError,
    ),
)
