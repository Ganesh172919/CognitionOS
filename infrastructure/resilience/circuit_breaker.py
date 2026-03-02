"""
Circuit Breaker - Advanced Resilience Pattern

Production-grade circuit breaker with half-open state, sliding window
failure detection, metric collection, and configurable strategies.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class FailureStrategy(str, Enum):
    COUNT = "count"
    RATE = "rate"
    CONSECUTIVE = "consecutive"


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker instance."""

    name: str = "default"
    failure_threshold: int = 5
    failure_rate_threshold: float = 0.5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3
    sliding_window_size: int = 100
    sliding_window_seconds: float = 120.0
    strategy: FailureStrategy = FailureStrategy.COUNT
    excluded_exceptions: List[type] = field(default_factory=list)
    fallback: Optional[Callable] = None
    on_state_change: Optional[Callable] = None
    slow_call_threshold_seconds: float = 5.0
    slow_call_rate_threshold: float = 0.8


@dataclass
class CallResult:
    success: bool
    duration_ms: float
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.monotonic)
    slow: bool = False


class SlidingWindow:
    """Time-based sliding window for failure tracking."""

    def __init__(self, max_size: int = 100, window_seconds: float = 120.0):
        self._results: deque = deque(maxlen=max_size)
        self._window_seconds = window_seconds

    def record(self, result: CallResult) -> None:
        self._results.append(result)

    def _active_results(self) -> List[CallResult]:
        cutoff = time.monotonic() - self._window_seconds
        return [r for r in self._results if r.timestamp > cutoff]

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self._active_results() if not r.success)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self._active_results() if r.success)

    @property
    def total_count(self) -> int:
        return len(self._active_results())

    @property
    def failure_rate(self) -> float:
        total = self.total_count
        if total == 0:
            return 0.0
        return self.failure_count / total

    @property
    def slow_call_rate(self) -> float:
        results = self._active_results()
        if not results:
            return 0.0
        slow_count = sum(1 for r in results if r.slow)
        return slow_count / len(results)

    @property
    def avg_duration_ms(self) -> float:
        results = self._active_results()
        if not results:
            return 0.0
        return sum(r.duration_ms for r in results) / len(results)

    @property
    def consecutive_failures(self) -> int:
        count = 0
        for r in reversed(self._active_results()):
            if not r.success:
                count += 1
            else:
                break
        return count

    @property
    def consecutive_successes(self) -> int:
        count = 0
        for r in reversed(self._active_results()):
            if r.success:
                count += 1
            else:
                break
        return count

    def clear(self) -> None:
        self._results.clear()


class CircuitBreaker:
    """
    Production circuit breaker with sliding window analysis.

    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Circuit tripped, calls blocked (returns fallback or raises)
    - HALF_OPEN: Testing recovery, limited calls allowed

    Strategies:
    - COUNT: Trip after N failures in window
    - RATE: Trip when failure rate exceeds threshold
    - CONSECUTIVE: Trip after N consecutive failures
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._window = SlidingWindow(
            max_size=self._config.sliding_window_size,
            window_seconds=self._config.sliding_window_seconds,
        )
        self._opened_at: Optional[float] = None
        self._half_open_calls = 0
        self._state_change_count = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def name(self) -> str:
        return self._config.name

    async def call(self, fn: Callable[..., Coroutine], *args, **kwargs) -> Any:
        """Execute a function through the circuit breaker."""

        async with self._lock:
            if not self._can_execute():
                if self._config.fallback:
                    logger.warning(
                        "CircuitBreaker[%s] OPEN - using fallback", self.name
                    )
                    return await self._config.fallback(*args, **kwargs) if asyncio.iscoroutinefunction(self._config.fallback) else self._config.fallback(*args, **kwargs)
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is open",
                    breaker_name=self.name,
                    state=self._state,
                )

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1

        # Execute the call
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(fn(*args, **kwargs), timeout=60.0)
            elapsed_ms = (time.monotonic() - start) * 1000
            is_slow = elapsed_ms > self._config.slow_call_threshold_seconds * 1000

            call_result = CallResult(
                success=True, duration_ms=elapsed_ms, slow=is_slow
            )
            self._window.record(call_result)

            async with self._lock:
                await self._on_success()

            return result

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000

            # Check if exception is excluded
            if any(isinstance(exc, e) for e in self._config.excluded_exceptions):
                raise

            call_result = CallResult(
                success=False,
                duration_ms=elapsed_ms,
                error=str(exc),
            )
            self._window.record(call_result)

            async with self._lock:
                await self._on_failure(str(exc))

            raise

    def _can_execute(self) -> bool:
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if self._opened_at and (
                time.monotonic() - self._opened_at >= self._config.timeout_seconds
            ):
                self._transition_to(CircuitState.HALF_OPEN)
                self._half_open_calls = 0
                return True
            return False

        if self._state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self._config.half_open_max_calls

        return False

    async def _on_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            if self._window.consecutive_successes >= self._config.success_threshold:
                self._transition_to(CircuitState.CLOSED)

    async def _on_failure(self, error: str) -> None:
        should_trip = False

        if self._config.strategy == FailureStrategy.COUNT:
            should_trip = self._window.failure_count >= self._config.failure_threshold
        elif self._config.strategy == FailureStrategy.RATE:
            should_trip = (
                self._window.total_count >= self._config.sliding_window_size // 2
                and self._window.failure_rate >= self._config.failure_rate_threshold
            )
        elif self._config.strategy == FailureStrategy.CONSECUTIVE:
            should_trip = (
                self._window.consecutive_failures >= self._config.failure_threshold
            )

        # Also check slow call rate
        if (
            self._window.slow_call_rate >= self._config.slow_call_rate_threshold
            and self._window.total_count >= 10
        ):
            should_trip = True

        if should_trip or self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self._state_change_count += 1

        if new_state == CircuitState.OPEN:
            self._opened_at = time.monotonic()
        elif new_state == CircuitState.CLOSED:
            self._window.clear()

        logger.info(
            "CircuitBreaker[%s] %s -> %s", self.name, old_state.value, new_state.value
        )

        if self._config.on_state_change:
            try:
                self._config.on_state_change(self.name, old_state, new_state)
            except Exception:
                logger.exception("State change callback error")

    def reset(self) -> None:
        self._state = CircuitState.CLOSED
        self._window.clear()
        self._opened_at = None
        self._half_open_calls = 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self._state.value,
            "strategy": self._config.strategy.value,
            "failure_count": self._window.failure_count,
            "success_count": self._window.success_count,
            "total_calls": self._window.total_count,
            "failure_rate": round(self._window.failure_rate, 4),
            "slow_call_rate": round(self._window.slow_call_rate, 4),
            "avg_duration_ms": round(self._window.avg_duration_ms, 2),
            "consecutive_failures": self._window.consecutive_failures,
            "state_changes": self._state_change_count,
            "opened_at": self._opened_at,
        }


class CircuitOpenError(Exception):
    """Raised when a call is rejected by an open circuit breaker."""

    def __init__(
        self,
        message: str,
        breaker_name: str = "",
        state: CircuitState = CircuitState.OPEN,
    ):
        super().__init__(message)
        self.breaker_name = breaker_name
        self.state = state


class CircuitBreakerRegistry:
    """Manages multiple circuit breaker instances."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get_or_create(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        if name not in self._breakers:
            cfg = config or CircuitBreakerConfig(name=name)
            cfg.name = name
            self._breakers[name] = CircuitBreaker(cfg)
        return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        return self._breakers.get(name)

    def get_all_stats(self) -> Dict[str, Any]:
        return {name: cb.get_stats() for name, cb in self._breakers.items()}

    def reset_all(self) -> None:
        for cb in self._breakers.values():
            cb.reset()

    @property
    def breaker_count(self) -> int:
        return len(self._breakers)


# Module-level default registry
_default_registry = CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str, config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """Get or create a circuit breaker from the default registry."""
    return _default_registry.get_or_create(name, config)


def circuit_breaker(name: str, **config_kwargs):
    """Decorator to wrap an async function with a circuit breaker."""

    def decorator(fn: Callable):
        cfg = CircuitBreakerConfig(name=name, **config_kwargs)
        cb = get_circuit_breaker(name, cfg)

        async def wrapper(*args, **kwargs):
            return await cb.call(fn, *args, **kwargs)

        wrapper.__wrapped__ = fn
        wrapper.circuit_breaker = cb
        return wrapper

    return decorator
