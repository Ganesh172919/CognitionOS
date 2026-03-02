"""
Comprehensive Testing Framework — CognitionOS

Test infrastructure providing:
- Test case discovery and organization
- Fixture management
- Mock factory for all major services
- Assertion helpers for domain objects
- Integration test harness
- Performance test integration
- Test result collection and reporting
- Coverage tracking
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Dict, List, Optional, Set, Type

logger = logging.getLogger(__name__)


class TestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


class TestCategory(str, Enum):
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    SECURITY = "security"
    SMOKE = "smoke"
    REGRESSION = "regression"


@dataclass
class TestCase:
    test_id: str
    name: str
    description: str = ""
    category: TestCategory = TestCategory.UNIT
    tags: List[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    retries: int = 0
    dependencies: List[str] = field(default_factory=list)
    setup_fn: Optional[Callable] = None
    teardown_fn: Optional[Callable] = None
    test_fn: Optional[Callable] = None

    # Runtime
    status: TestStatus = TestStatus.PENDING
    duration_ms: float = 0
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    assertions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id, "name": self.name,
            "category": self.category.value,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 1),
            "error": self.error, "assertions": self.assertions,
            "tags": self.tags,
        }


@dataclass
class TestSuiteResult:
    suite_id: str
    name: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_ms: float = 0
    tests: List[TestCase] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0

    @property
    def success_rate(self) -> float:
        return (self.passed / max(self.total, 1)) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_id": self.suite_id, "name": self.name,
            "total": self.total, "passed": self.passed,
            "failed": self.failed, "skipped": self.skipped,
            "errors": self.errors,
            "success_rate_pct": round(self.success_rate, 1),
            "duration_ms": round(self.duration_ms, 1),
            "tests": [t.to_dict() for t in self.tests],
        }


# ── Assertion Helpers ──

class AssertionError(Exception):
    pass


class Assert:
    """Rich assertion library for domain testing."""

    @staticmethod
    def equal(actual: Any, expected: Any, msg: str = ""):
        if actual != expected:
            raise AssertionError(
                msg or f"Expected {repr(expected)}, got {repr(actual)}"
            )

    @staticmethod
    def not_equal(actual: Any, expected: Any, msg: str = ""):
        if actual == expected:
            raise AssertionError(
                msg or f"Expected {repr(actual)} to not equal {repr(expected)}"
            )

    @staticmethod
    def is_true(value: Any, msg: str = ""):
        if not value:
            raise AssertionError(msg or f"Expected truthy value, got {repr(value)}")

    @staticmethod
    def is_false(value: Any, msg: str = ""):
        if value:
            raise AssertionError(msg or f"Expected falsy value, got {repr(value)}")

    @staticmethod
    def is_none(value: Any, msg: str = ""):
        if value is not None:
            raise AssertionError(msg or f"Expected None, got {repr(value)}")

    @staticmethod
    def is_not_none(value: Any, msg: str = ""):
        if value is None:
            raise AssertionError(msg or "Expected non-None value")

    @staticmethod
    def is_instance(obj: Any, cls: Type, msg: str = ""):
        if not isinstance(obj, cls):
            raise AssertionError(
                msg or f"Expected instance of {cls.__name__}, got {type(obj).__name__}"
            )

    @staticmethod
    def contains(container: Any, item: Any, msg: str = ""):
        if item not in container:
            raise AssertionError(
                msg or f"Expected {repr(item)} in {repr(container)}"
            )

    @staticmethod
    def not_contains(container: Any, item: Any, msg: str = ""):
        if item in container:
            raise AssertionError(
                msg or f"Expected {repr(item)} not in {repr(container)}"
            )

    @staticmethod
    def raises(exc_type: Type[Exception], fn: Callable, *args, **kwargs):
        try:
            fn(*args, **kwargs)
            raise AssertionError(f"Expected {exc_type.__name__} to be raised")
        except exc_type:
            pass
        except Exception as exc:
            raise AssertionError(
                f"Expected {exc_type.__name__}, got {type(exc).__name__}: {exc}"
            )

    @staticmethod
    async def async_raises(exc_type: Type[Exception],
                            fn: Callable[..., Awaitable], *args, **kwargs):
        try:
            await fn(*args, **kwargs)
            raise AssertionError(f"Expected {exc_type.__name__} to be raised")
        except exc_type:
            pass
        except AssertionError:
            raise
        except Exception as exc:
            raise AssertionError(
                f"Expected {exc_type.__name__}, got {type(exc).__name__}: {exc}"
            )

    @staticmethod
    def greater_than(actual: float, expected: float, msg: str = ""):
        if not actual > expected:
            raise AssertionError(
                msg or f"Expected {actual} > {expected}"
            )

    @staticmethod
    def less_than(actual: float, expected: float, msg: str = ""):
        if not actual < expected:
            raise AssertionError(
                msg or f"Expected {actual} < {expected}"
            )

    @staticmethod
    def approx_equal(actual: float, expected: float, *,
                      tolerance: float = 0.01, msg: str = ""):
        if abs(actual - expected) > tolerance:
            raise AssertionError(
                msg or f"Expected {actual} ≈ {expected} (tolerance={tolerance})"
            )

    @staticmethod
    def has_key(d: Dict, key: str, msg: str = ""):
        if key not in d:
            raise AssertionError(msg or f"Expected key {repr(key)} in dict")

    @staticmethod
    def list_length(lst: List, expected_len: int, msg: str = ""):
        if len(lst) != expected_len:
            raise AssertionError(
                msg or f"Expected list length {expected_len}, got {len(lst)}"
            )

    @staticmethod
    def matches_schema(data: Dict, schema: Dict[str, Type], msg: str = ""):
        for key, expected_type in schema.items():
            if key not in data:
                raise AssertionError(
                    msg or f"Missing key {repr(key)} in data"
                )
            if not isinstance(data[key], expected_type):
                raise AssertionError(
                    msg or f"Key {repr(key)}: expected {expected_type.__name__}, "
                    f"got {type(data[key]).__name__}"
                )


# ── Mock Factory ──

class MockFactory:
    """Factory for creating mocks of platform services."""

    @staticmethod
    def create_mock_db_session():
        """Create a mock async database session."""
        from unittest.mock import AsyncMock, MagicMock
        session = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.close = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=[]))
        )))
        return session

    @staticmethod
    def create_mock_event_bus():
        from unittest.mock import AsyncMock
        bus = AsyncMock()
        bus.publish = AsyncMock(return_value=True)
        bus.subscribe = AsyncMock(return_value="sub_123")
        bus.get_metrics = AsyncMock(return_value={})
        return bus

    @staticmethod
    def create_mock_cache():
        from unittest.mock import AsyncMock
        cache = AsyncMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        cache.delete = AsyncMock()
        cache.exists = AsyncMock(return_value=False)
        return cache

    @staticmethod
    def create_mock_llm():
        from unittest.mock import AsyncMock
        llm = AsyncMock()
        llm.generate = AsyncMock(return_value="Generated response")
        llm.embed = AsyncMock(return_value=[0.1] * 32)
        return llm

    @staticmethod
    def create_mock_user(*, user_id: str = "", email: str = "test@example.com",
                          tenant_id: str = "tenant_001"):
        return {
            "user_id": user_id or uuid.uuid4().hex,
            "email": email,
            "tenant_id": tenant_id,
            "role": "admin",
            "created_at": time.time(),
        }

    @staticmethod
    def create_mock_tenant(*, tenant_id: str = "", name: str = "Test Corp",
                            tier: str = "pro"):
        return {
            "tenant_id": tenant_id or uuid.uuid4().hex,
            "name": name,
            "tier": tier,
            "created_at": time.time(),
            "settings": {},
        }


# ── Test Runner ──

class TestRunner:
    """Async test runner with fixture management and reporting."""

    def __init__(self, *, max_concurrent: int = 10,
                 default_timeout: float = 30.0):
        self._suites: Dict[str, List[TestCase]] = defaultdict(list)
        self._results: List[TestSuiteResult] = []
        self._max_concurrent = max_concurrent
        self._default_timeout = default_timeout
        self._global_setup: Optional[Callable] = None
        self._global_teardown: Optional[Callable] = None

    def register_test(self, suite: str, test: TestCase):
        self._suites[suite].append(test)

    def set_global_setup(self, fn: Callable):
        self._global_setup = fn

    def set_global_teardown(self, fn: Callable):
        self._global_teardown = fn

    async def run_suite(self, suite_name: str, *,
                         category: Optional[TestCategory] = None,
                         tags: Optional[List[str]] = None) -> TestSuiteResult:
        """Run a test suite."""
        tests = self._suites.get(suite_name, [])
        if category:
            tests = [t for t in tests if t.category == category]
        if tags:
            tests = [t for t in tests if any(tag in t.tags for tag in tags)]

        result = TestSuiteResult(
            suite_id=uuid.uuid4().hex[:12],
            name=suite_name,
            total=len(tests),
        )

        # Global setup
        if self._global_setup:
            try:
                if asyncio.iscoroutinefunction(self._global_setup):
                    await self._global_setup()
                else:
                    self._global_setup()
            except Exception as exc:
                logger.error("Global setup failed: %s", exc)
                return result

        start_time = time.time()
        semaphore = asyncio.Semaphore(self._max_concurrent)

        tasks = []
        for test in tests:
            tasks.append(self._run_test(test, semaphore))
        await asyncio.gather(*tasks)

        # Collect results
        for test in tests:
            result.tests.append(test)
            if test.status == TestStatus.PASSED:
                result.passed += 1
            elif test.status == TestStatus.FAILED:
                result.failed += 1
            elif test.status == TestStatus.SKIPPED:
                result.skipped += 1
            elif test.status == TestStatus.ERROR:
                result.errors += 1

        result.duration_ms = (time.time() - start_time) * 1000
        result.completed_at = time.time()

        # Global teardown
        if self._global_teardown:
            try:
                if asyncio.iscoroutinefunction(self._global_teardown):
                    await self._global_teardown()
                else:
                    self._global_teardown()
            except Exception:
                pass

        self._results.append(result)
        return result

    async def _run_test(self, test: TestCase, semaphore: asyncio.Semaphore):
        """Execute a single test case."""
        async with semaphore:
            test.status = TestStatus.RUNNING
            start = time.perf_counter()

            try:
                # Setup
                if test.setup_fn:
                    if asyncio.iscoroutinefunction(test.setup_fn):
                        await test.setup_fn()
                    else:
                        test.setup_fn()

                # Execute test
                if test.test_fn:
                    if asyncio.iscoroutinefunction(test.test_fn):
                        await asyncio.wait_for(
                            test.test_fn(), timeout=test.timeout_seconds
                        )
                    else:
                        test.test_fn()

                test.status = TestStatus.PASSED

            except asyncio.TimeoutError:
                test.status = TestStatus.TIMEOUT
                test.error = f"Test timed out after {test.timeout_seconds}s"
            except AssertionError as exc:
                test.status = TestStatus.FAILED
                test.error = str(exc)
                test.error_traceback = traceback.format_exc()
            except Exception as exc:
                test.status = TestStatus.ERROR
                test.error = str(exc)
                test.error_traceback = traceback.format_exc()
            finally:
                test.duration_ms = (time.perf_counter() - start) * 1000

                # Teardown
                if test.teardown_fn:
                    try:
                        if asyncio.iscoroutinefunction(test.teardown_fn):
                            await test.teardown_fn()
                        else:
                            test.teardown_fn()
                    except Exception as exc:
                        logger.warning("Teardown failed for %s: %s", test.name, exc)

    async def run_all(self, *,
                       category: Optional[TestCategory] = None) -> List[TestSuiteResult]:
        """Run all registered test suites."""
        results = []
        for suite_name in self._suites:
            result = await self.run_suite(suite_name, category=category)
            results.append(result)
        return results

    def get_results(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._results]

    def get_summary(self) -> Dict[str, Any]:
        total = sum(r.total for r in self._results)
        passed = sum(r.passed for r in self._results)
        failed = sum(r.failed for r in self._results)
        return {
            "suites": len(self._results),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "success_rate_pct": round(passed / max(total, 1) * 100, 1),
        }


# ── Test Registration Decorator ──

def test_case(suite: str = "default", *,
               category: TestCategory = TestCategory.UNIT,
               tags: Optional[List[str]] = None,
               timeout: float = 30.0):
    """Decorator to register a function as a test case."""
    def decorator(fn):
        tc = TestCase(
            test_id=uuid.uuid4().hex[:12],
            name=fn.__name__,
            description=fn.__doc__ or "",
            category=category,
            tags=tags or [],
            timeout_seconds=timeout,
            test_fn=fn,
        )
        # Store for later discovery
        if not hasattr(fn, "_test_cases"):
            fn._test_cases = []
        fn._test_cases.append((suite, tc))
        return fn
    return decorator


# ── Singleton ──
_runner: Optional[TestRunner] = None


def get_test_runner() -> TestRunner:
    global _runner
    if not _runner:
        _runner = TestRunner()
    return _runner
