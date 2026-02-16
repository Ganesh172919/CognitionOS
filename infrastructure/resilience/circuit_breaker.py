"""
Circuit Breaker Pattern for CognitionOS V4
Phase 5.3: Resilience & Intelligence
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, Any, Dict

from infrastructure.observability import get_logger


logger = get_logger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: int = 60
    half_open_max_requests: int = 3
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    jitter_factor: float = 0.1


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None
    state_changes: Dict[str, int] = field(default_factory=lambda: {
        "closed_to_open": 0,
        "open_to_half_open": 0,
        "half_open_to_closed": 0,
        "half_open_to_open": 0
    })


class CircuitBreaker:
    """Circuit Breaker implementation"""
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[Callable] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback = fallback
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.opened_at: Optional[datetime] = None
        self.half_open_requests = 0
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        self.metrics.total_requests += 1
        await self._update_state()
        
        if self.state == CircuitState.OPEN:
            self.metrics.rejected_requests += 1
            if self.fallback:
                return await self._execute_with_fallback(*args, **kwargs)
            raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        if self.state == CircuitState.HALF_OPEN:
            async with self._lock:
                if self.half_open_requests >= self.config.half_open_max_requests:
                    self.metrics.rejected_requests += 1
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is HALF_OPEN with max requests")
                self.half_open_requests += 1
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            if self.fallback:
                return await self._execute_with_fallback(*args, **kwargs)
            raise
        finally:
            if self.state == CircuitState.HALF_OPEN:
                async with self._lock:
                    self.half_open_requests -= 1
    
    async def _update_state(self):
        """Update circuit state based on time and metrics"""
        async with self._lock:
            if self.state == CircuitState.OPEN and self.opened_at:
                elapsed = (datetime.utcnow() - self.opened_at).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    await self._transition_to(CircuitState.HALF_OPEN)
    
    async def _on_success(self):
        """Handle successful request"""
        async with self._lock:
            self.metrics.successful_requests += 1
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = datetime.utcnow()
            
            if self.state == CircuitState.HALF_OPEN:
                if self.metrics.consecutive_successes >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
    
    async def _on_failure(self):
        """Handle failed request"""
        async with self._lock:
            self.metrics.failed_requests += 1
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            self.metrics.last_failure_time = datetime.utcnow()
            
            if self.state == CircuitState.CLOSED:
                if self.metrics.consecutive_failures >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)
            elif self.state == CircuitState.HALF_OPEN:
                await self._transition_to(CircuitState.OPEN)
    
    async def _transition_to(self, new_state: CircuitState):
        """Transition to new circuit state"""
        old_state = self.state
        self.state = new_state
        self.metrics.last_state_change = datetime.utcnow()
        
        transition_key = f"{old_state.value}_to_{new_state.value}"
        if transition_key in self.metrics.state_changes:
            self.metrics.state_changes[transition_key] += 1
        
        if new_state == CircuitState.OPEN:
            self.opened_at = datetime.utcnow()
            self.metrics.consecutive_failures = 0
        elif new_state == CircuitState.HALF_OPEN:
            self.half_open_requests = 0
            self.metrics.consecutive_successes = 0
        elif new_state == CircuitState.CLOSED:
            self.opened_at = None
            self.metrics.consecutive_failures = 0
            self.metrics.consecutive_successes = 0
        
        logger.info(f"Circuit breaker {self.name} state change: {old_state.value} -> {new_state.value}")
    
    async def _execute_with_fallback(self, *args, **kwargs) -> Any:
        """Execute fallback function"""
        if asyncio.iscoroutinefunction(self.fallback):
            return await self.fallback(*args, **kwargs)
        return self.fallback(*args, **kwargs)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        success_rate = 0.0
        if self.metrics.total_requests > 0:
            success_rate = self.metrics.successful_requests / self.metrics.total_requests
        
        return {
            "name": self.name,
            "state": self.state.value,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "rejected_requests": self.metrics.rejected_requests,
            "success_rate": success_rate,
            "consecutive_failures": self.metrics.consecutive_failures,
            "consecutive_successes": self.metrics.consecutive_successes,
            "state_changes": self.metrics.state_changes
        }


class ExponentialBackoff:
    """Exponential backoff with jitter for retry logic"""
    
    def __init__(self, initial_seconds: float = 1.0, max_seconds: float = 60.0, 
                 multiplier: float = 2.0, jitter_factor: float = 0.1):
        self.initial = initial_seconds
        self.max = max_seconds
        self.multiplier = multiplier
        self.jitter_factor = jitter_factor
    
    def calculate(self, attempt: int) -> float:
        """Calculate backoff time for given attempt"""
        base = min(self.initial * (self.multiplier ** attempt), self.max)
        jitter = base * self.jitter_factor * (2 * random.random() - 1)
        return max(0, base + jitter)
    
    async def wait(self, attempt: int):
        """Wait for calculated backoff time"""
        await asyncio.sleep(self.calculate(attempt))


class BulkheadIsolation:
    """Bulkhead isolation pattern for resource limiting"""
    
    def __init__(self, name: str, max_concurrent: int = 10):
        self.name = name
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.total_requests = 0
        self.active_requests = 0
        self.rejected_requests = 0
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with bulkhead isolation"""
        self.total_requests += 1
        
        try:
            async with self.semaphore:
                self.active_requests += 1
                try:
                    return await func(*args, **kwargs)
                finally:
                    self.active_requests -= 1
        except Exception as e:
            # If semaphore acquisition fails or function fails
            self.rejected_requests += 1
            if isinstance(e, BulkheadFullError):
                raise
            raise BulkheadFullError(f"Bulkhead {self.name} is full") from e
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get bulkhead metrics"""
        return {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "rejected_requests": self.rejected_requests,
            "utilization": self.active_requests / self.max_concurrent
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class BulkheadFullError(Exception):
    """Raised when bulkhead is full"""
    pass
