"""
Background Task Queue — CognitionOS

Async task queue with:
- Priority-based scheduling
- Retry with exponential backoff
- Concurrency limits
- Task chaining and DAG support
- Dead letter handling
- Real-time status tracking
- Scheduled/cron tasks
- Task de-duplication
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskState(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    DEAD_LETTERED = "dead_lettered"


class TaskPriority(int, Enum):
    LOW = 10
    NORMAL = 50
    HIGH = 80
    CRITICAL = 100


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TaskDefinition:
    task_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    handler_name: str = ""
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff_factor: float = 2.0
    timeout_seconds: float = 300.0
    dedup_key: Optional[str] = None
    chain_next: Optional[str] = None  # handler name to chain
    metadata: Dict[str, Any] = field(default_factory=dict)
    scheduled_at: Optional[str] = None
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class TaskResult:
    task_id: str
    state: TaskState
    result: Any = None
    error: Optional[str] = None
    traceback_str: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: float = 0.0
    retry_count: int = 0
    handler_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "state": self.state.value,
            "result": str(self.result)[:500] if self.result else None,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "retry_count": self.retry_count,
            "handler_name": self.handler_name,
        }


@dataclass
class ScheduledTask:
    task_id: str
    handler_name: str
    cron_expression: str  # simplified: "interval:seconds" or "at:HH:MM"
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    last_run_at: Optional[str] = None
    next_run_at: Optional[str] = None
    run_count: int = 0


# ---------------------------------------------------------------------------
# Task Queue
# ---------------------------------------------------------------------------


class TaskQueue:
    """Async in-process background task queue."""

    def __init__(
        self,
        *,
        max_concurrency: int = 10,
        max_queue_size: int = 10000,
        enable_dedup: bool = True,
    ) -> None:
        self._handlers: Dict[str, Callable[..., Awaitable[Any]]] = {}
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._results: Dict[str, TaskResult] = {}
        self._max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._running = False
        self._workers: List[asyncio.Task] = []
        self._enable_dedup = enable_dedup
        self._active_dedup_keys: Set[str] = set()
        self._dead_letter: List[TaskResult] = []
        self._scheduled_tasks: Dict[str, ScheduledTask] = {}
        self._scheduler_task: Optional[asyncio.Task] = None
        self._metrics: Dict[str, int] = defaultdict(int)
        self._task_counter = 0

    # ----- handler registration -----

    def register(self, name: str, handler: Callable[..., Awaitable[Any]]) -> None:
        self._handlers[name] = handler
        logger.debug("Registered task handler: %s", name)

    def handler(self, name: Optional[str] = None) -> Callable:
        """Decorator for registering handlers."""
        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable:
            handler_name = name or func.__qualname__
            self.register(handler_name, func)
            return func
        return decorator

    # ----- lifecycle -----

    async def start(self, num_workers: int = 0) -> None:
        if self._running:
            return
        self._running = True
        worker_count = num_workers or self._max_concurrency
        for i in range(worker_count):
            task = asyncio.create_task(self._worker_loop(i))
            self._workers.append(task)
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("TaskQueue started with %d workers", worker_count)

    async def stop(self, timeout: float = 10.0) -> None:
        self._running = False
        for worker in self._workers:
            worker.cancel()
        if self._scheduler_task:
            self._scheduler_task.cancel()
        await asyncio.gather(*self._workers, self._scheduler_task or asyncio.sleep(0), return_exceptions=True)
        self._workers.clear()
        logger.info("TaskQueue stopped. Processed=%d Failed=%d", self._metrics["completed"], self._metrics["failed"])

    # ----- submit -----

    async def submit(
        self,
        handler_name: str,
        *args: Any,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        timeout_seconds: float = 300.0,
        dedup_key: Optional[str] = None,
        chain_next: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        if handler_name not in self._handlers:
            raise ValueError(f"Unknown handler: {handler_name}")

        task_def = TaskDefinition(
            name=handler_name,
            handler_name=handler_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            dedup_key=dedup_key,
            chain_next=chain_next,
            metadata=metadata or {},
        )

        # Dedup check
        if self._enable_dedup and dedup_key:
            if dedup_key in self._active_dedup_keys:
                self._metrics["deduplicated"] += 1
                logger.debug("Deduplicated task: %s", dedup_key)
                return ""
            self._active_dedup_keys.add(dedup_key)

        # Initialize result tracking
        self._results[task_def.task_id] = TaskResult(
            task_id=task_def.task_id,
            state=TaskState.QUEUED,
            handler_name=handler_name,
        )

        # Priority queue uses (negative_priority, counter, task) for ordering
        self._task_counter += 1
        await self._queue.put((-priority.value, self._task_counter, task_def))
        self._metrics["submitted"] += 1
        return task_def.task_id

    # ----- worker -----

    async def _worker_loop(self, worker_id: int) -> None:
        while self._running:
            try:
                _, _, task_def = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            async with self._semaphore:
                await self._execute_task(task_def)

    async def _execute_task(self, task_def: TaskDefinition, retry_count: int = 0) -> None:
        handler = self._handlers.get(task_def.handler_name)
        if not handler:
            self._metrics["handler_not_found"] += 1
            return

        result = self._results.get(task_def.task_id)
        if result:
            result.state = TaskState.RUNNING
            result.started_at = datetime.now(timezone.utc).isoformat()
            result.retry_count = retry_count

        start = time.monotonic()
        try:
            output = await asyncio.wait_for(
                handler(*task_def.args, **task_def.kwargs),
                timeout=task_def.timeout_seconds,
            )
            elapsed_ms = (time.monotonic() - start) * 1000

            if result:
                result.state = TaskState.COMPLETED
                result.result = output
                result.completed_at = datetime.now(timezone.utc).isoformat()
                result.duration_ms = elapsed_ms

            self._metrics["completed"] += 1

            # Clean dedup
            if task_def.dedup_key:
                self._active_dedup_keys.discard(task_def.dedup_key)

            # Chain next
            if task_def.chain_next and task_def.chain_next in self._handlers:
                await self.submit(task_def.chain_next, output, priority=task_def.priority)

        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - start) * 1000
            error_msg = f"Task timed out after {task_def.timeout_seconds}s"
            await self._handle_failure(task_def, error_msg, retry_count, elapsed_ms)

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            await self._handle_failure(task_def, str(exc), retry_count, elapsed_ms, traceback.format_exc())

    async def _handle_failure(
        self,
        task_def: TaskDefinition,
        error: str,
        retry_count: int,
        elapsed_ms: float,
        tb: Optional[str] = None,
    ) -> None:
        result = self._results.get(task_def.task_id)
        if retry_count < task_def.max_retries:
            # Retry with backoff
            delay = task_def.retry_delay_seconds * (task_def.retry_backoff_factor ** retry_count)
            logger.warning("Task %s failed (attempt %d/%d), retrying in %.1fs: %s",
                           task_def.task_id, retry_count + 1, task_def.max_retries, delay, error)
            if result:
                result.state = TaskState.RETRYING
            await asyncio.sleep(delay)
            await self._execute_task(task_def, retry_count + 1)
        else:
            # Dead letter
            logger.error("Task %s permanently failed after %d retries: %s", task_def.task_id, task_def.max_retries, error)
            if result:
                result.state = TaskState.DEAD_LETTERED
                result.error = error
                result.traceback_str = tb
                result.completed_at = datetime.now(timezone.utc).isoformat()
                result.duration_ms = elapsed_ms
                self._dead_letter.append(result)

            self._metrics["failed"] += 1
            if task_def.dedup_key:
                self._active_dedup_keys.discard(task_def.dedup_key)

    # ----- scheduled tasks -----

    def schedule(
        self,
        handler_name: str,
        cron_expression: str,
        *args: Any,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        tid = task_id or str(uuid4())
        scheduled = ScheduledTask(
            task_id=tid,
            handler_name=handler_name,
            cron_expression=cron_expression,
            args=args,
            kwargs=kwargs,
        )
        self._scheduled_tasks[tid] = scheduled
        logger.info("Scheduled task %s: %s [%s]", tid, handler_name, cron_expression)
        return tid

    def unschedule(self, task_id: str) -> bool:
        return self._scheduled_tasks.pop(task_id, None) is not None

    async def _scheduler_loop(self) -> None:
        while self._running:
            try:
                for scheduled in list(self._scheduled_tasks.values()):
                    if not scheduled.is_active:
                        continue
                    if self._should_run_scheduled(scheduled):
                        try:
                            await self.submit(
                                scheduled.handler_name,
                                *scheduled.args,
                                **scheduled.kwargs,
                            )
                            scheduled.last_run_at = datetime.now(timezone.utc).isoformat()
                            scheduled.run_count += 1
                        except Exception as e:
                            logger.error("Scheduled task %s error: %s", scheduled.task_id, e)
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break

    def _should_run_scheduled(self, scheduled: ScheduledTask) -> bool:
        expr = scheduled.cron_expression
        now = time.time()

        if expr.startswith("interval:"):
            try:
                interval = float(expr.split(":")[1])
            except (IndexError, ValueError):
                return False
            if scheduled.last_run_at:
                last = datetime.fromisoformat(scheduled.last_run_at).timestamp()
                return (now - last) >= interval
            return True

        return False

    # ----- query -----

    def get_result(self, task_id: str) -> Optional[TaskResult]:
        return self._results.get(task_id)

    def get_dead_letters(self, limit: int = 100) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._dead_letter[-limit:]]

    async def retry_dead_letter(self, task_id: str) -> bool:
        for i, result in enumerate(self._dead_letter):
            if result.task_id == task_id:
                self._dead_letter.pop(i)
                await self.submit(result.handler_name, priority=TaskPriority.HIGH)
                return True
        return False

    def cancel_task(self, task_id: str) -> bool:
        result = self._results.get(task_id)
        if result and result.state in (TaskState.QUEUED, TaskState.PENDING):
            result.state = TaskState.CANCELLED
            return True
        return False

    def get_metrics(self) -> Dict[str, Any]:
        return {
            **dict(self._metrics),
            "queue_size": self._queue.qsize(),
            "active_dedup_keys": len(self._active_dedup_keys),
            "dead_letter_count": len(self._dead_letter),
            "scheduled_count": len(self._scheduled_tasks),
            "total_tracked_results": len(self._results),
        }

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        return [
            r.to_dict() for r in self._results.values()
            if r.state in (TaskState.RUNNING, TaskState.RETRYING)
        ]

    def get_scheduled_tasks(self) -> List[Dict[str, Any]]:
        return [
            {
                "task_id": s.task_id,
                "handler": s.handler_name,
                "cron": s.cron_expression,
                "active": s.is_active,
                "run_count": s.run_count,
                "last_run": s.last_run_at,
            }
            for s in self._scheduled_tasks.values()
        ]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    global _queue
    if _queue is None:
        _queue = TaskQueue()
    return _queue


async def init_task_queue(**kwargs: Any) -> TaskQueue:
    global _queue
    _queue = TaskQueue(**kwargs)
    await _queue.start()
    return _queue
